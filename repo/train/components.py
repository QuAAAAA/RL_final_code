from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any

from .dataset import build_gold_control
from .global_intensity_reward import global_emotion_intensity_reward_json
from .http_client import JsonServiceClient
from .schemas import ControlPlan, ServiceSpec, TrainExample


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def control_to_json(control: ControlPlan) -> dict[str, Any]:
    return {
        "text": control.text,
        "tagged_text": control.tagged_text,
        "target_emotion": control.target_emotion,
        "emotion_vector": control.emotion_vector,
        "va": control.va,
        "va_01": control.va_01,
        "emphasis_targets": control.emphasis_targets,
        "quadruplets": control.quadruplets,
    }


def control_from_json(payload: dict[str, Any]) -> ControlPlan:
    va = payload.get("va")
    va_tuple = tuple(va) if isinstance(va, list) else va
    va_01 = payload.get("va_01")
    va_01_tuple = tuple(va_01) if isinstance(va_01, list) else va_01
    return ControlPlan(
        text=str(payload.get("text") or ""),
        tagged_text=str(payload.get("tagged_text") or payload.get("text") or ""),
        target_emotion=str(payload.get("target_emotion") or "neutral"),
        emotion_vector=list(payload.get("emotion_vector") or []),
        va=va_tuple,
        va_01=va_01_tuple,
        emphasis_targets=list(payload.get("emphasis_targets") or []),
        quadruplets=list(payload.get("quadruplets") or []),
    )


class ComponentHub:
    def __init__(
        self,
        specs: dict[str, ServiceSpec],
        output_dir: Path,
        prompt_audio: str | None = None,
        reward_options: dict[str, Any] | None = None,
    ) -> None:
        self.specs = specs
        self.output_dir = output_dir
        self.prompt_audio = prompt_audio
        self.reward_options = reward_options or {}
        self.enabled_rewards = set(
            self.reward_options.get(
                "enabled_rewards",
                ["asr", "emotion", "speaker", "local_emphasis", "vad", "global_intensity"],
            )
        )

    def predict_control(self, example: TrainExample) -> ControlPlan:
        spec = self.specs.get("asqp")
        if spec is None or spec.mock:
            return build_gold_control(example)
        result = JsonServiceClient(spec.url or "").post(
            "/predict",
            {"id": example.uid, "text": example.text, "quadruplets": example.quadruplets},
        )
        return control_from_json(result["control"])

    def synthesize(
        self,
        example: TrainExample,
        control: ControlPlan,
        sample_id: str | None = None,
        seed: int | None = None,
    ) -> str | None:
        spec = self.specs.get("tts")
        request_id = example.uid if sample_id is None else f"{example.uid}:{sample_id}"
        if spec is None or spec.mock:
            return self._write_mock_wav(request_id)
        result = JsonServiceClient(spec.url or "", timeout=600).post(
            "/synthesize",
            {
                "id": request_id,
                "text": control.text,
                "tagged_text": control.tagged_text,
                "target_emotion": control.target_emotion,
                "emotion_vector": control.emotion_vector,
                "emphasis_targets": control.emphasis_targets,
                "prompt_audio": self.prompt_audio,
                "output_dir": str(self.output_dir / "wav"),
                "seed": seed,
            },
        )
        return result.get("wav_path")

    def score(self, example: TrainExample, control: ControlPlan, wav_path: str | None) -> dict[str, Any]:
        rewards: dict[str, Any] = {}
        if not wav_path:
            for name in self.enabled_rewards:
                rewards[name] = 0.0
            rewards["vad_va"] = None
            return rewards

        vad_va = self._vad_va(control, wav_path)
        vad_vector = self._vad_vector(control, wav_path, vad_va)
        if self._is_enabled("asr"):
            rewards["asr"] = self._asr_reward(example, wav_path)
        if self._is_enabled("emotion"):
            rewards["emotion"] = self._emotion_reward(control, wav_path)
        if self._is_enabled("speaker"):
            rewards["speaker"] = self._speaker_reward(wav_path)
        if self._is_enabled("local_emphasis"):
            rewards["local_emphasis"] = self._local_emphasis_reward(control, wav_path)
        if self._is_enabled("vad"):
            rewards["vad"] = float(vad_va.get("reward", 0.0)) if vad_va else 0.0
        if self._is_enabled("global_intensity"):
            intensity = self._global_intensity_reward(vad_vector)
            rewards["global_intensity"] = float(intensity["reward"])
            rewards["global_intensity_details"] = intensity
        rewards["vad_va"] = vad_va
        rewards["vad_vector"] = vad_vector
        return rewards

    def _is_enabled(self, reward_name: str) -> bool:
        return reward_name in self.enabled_rewards

    def _asr_reward(self, example: TrainExample, wav_path: str) -> float:
        spec = self.specs.get("asr")
        if spec is None or spec.mock:
            return 1.0
        result = JsonServiceClient(spec.url or "").post("/wer", {"audio_path": wav_path, "label": example.text})
        wer = float(result.get("wer", 1.0))
        return max(0.0, 1.0 - wer)

    def _emotion_reward(self, control: ControlPlan, wav_path: str) -> float:
        spec = self.specs.get("emotion")
        if spec is None or spec.mock:
            return 5.0
        result = JsonServiceClient(spec.url or "").post(
            "/classify",
            {"audio_path": wav_path, "target_emotion": control.target_emotion},
        )
        if "reward" in result:
            return float(result["reward"])
        return 5.0 if result.get("pred_emotion") == control.target_emotion else -1.0

    def _speaker_reward(self, wav_path: str) -> float:
        spec = self.specs.get("speaker")
        if spec is None or spec.mock or not self.prompt_audio:
            return 1.0
        result = JsonServiceClient(spec.url or "").post(
            "/similarity",
            {"source_audio": self.prompt_audio, "target_audio": wav_path},
        )
        return float(result.get("similarity", 0.0))

    def _local_emphasis_reward(self, control: ControlPlan, wav_path: str) -> float:
        spec = self.specs.get("local_emphasis")
        if spec is None or spec.mock or not control.emphasis_targets:
            return 1.0 if control.emphasis_targets else 0.0
        result = JsonServiceClient(spec.url or "").post(
            "/score",
            {
                "audio_path": wav_path,
                "emphasis_targets": control.emphasis_targets,
                "text": control.text,
            },
        )
        return float(result.get("total", result.get("reward", 0.0)))

    def _vad_va(self, control: ControlPlan, wav_path: str) -> dict[str, Any] | None:
        spec = self.specs.get("vad")
        target = control.va_01
        if spec is None or target is None:
            return None
        if spec.mock:
            pred_valence, pred_arousal = target
        else:
            result = JsonServiceClient(spec.url or "").post("/va", {"audio_path": wav_path})
            pred_valence = float(result["valence"])
            pred_arousal = float(result["arousal"])

        target_valence, target_arousal = target
        valence_abs_error = abs(pred_valence - target_valence)
        arousal_abs_error = abs(pred_arousal - target_arousal)
        mean_abs_error = (valence_abs_error + arousal_abs_error) / 2.0
        reward = max(0.0, 1.0 - mean_abs_error)
        return {
            "predicted": {"valence": pred_valence, "arousal": pred_arousal},
            "target": {"valence": target_valence, "arousal": target_arousal},
            "valence_abs_error": valence_abs_error,
            "arousal_abs_error": arousal_abs_error,
            "mean_abs_error": mean_abs_error,
            "reward": reward,
        }

    def _vad_vector(
        self,
        control: ControlPlan,
        wav_path: str,
        vad_va: dict[str, Any] | None,
    ) -> list[float] | None:
        spec = self.specs.get("vad")
        if spec is None:
            return None
        if spec.mock:
            if control.va_01 is None:
                return None
            valence, arousal = control.va_01
            dominance = float(self.reward_options.get("global_intensity", {}).get("mock_dominance", 0.5))
            return [float(arousal), dominance, float(valence)]
        result = JsonServiceClient(spec.url or "").post("/vad", {"audio_path": wav_path})
        return [float(result["arousal"]), float(result["dominance"]), float(result["valence"])]

    def _global_intensity_reward(self, vad_vector: list[float] | None) -> dict[str, Any]:
        cfg = self.reward_options.get("global_intensity", {})
        if vad_vector is None:
            return {"reward": 0.0, "d_y_hat": None, "R_interval": 0.0, "R_gaussian": 0.0}
        return global_emotion_intensity_reward_json(
            vad_vector,
            cfg.get("mu_neutral", [0.5, 0.5, 0.5]),
            cfg.get("target_range", [0.4, 0.6]),
            sigma=float(cfg.get("sigma", 0.1)),
        )

    def _write_mock_wav(self, uid: str) -> str:
        wav_dir = self.output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        path = wav_dir / f"{stable_id(uid)}.wav"
        if path.exists():
            return str(path)
        sample_rate = 16000
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * int(sample_rate * 0.2))
        return str(path)

