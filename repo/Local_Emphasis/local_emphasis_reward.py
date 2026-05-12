"""Local emphasis control reward for TTS/RL training.

This module scores whether a target word is locally emphasized in generated
speech using pitch and energy features aligned to word boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Dict, List, Literal, Sequence, Tuple

import torch
import torch.nn.functional as F


WordBoundary = Tuple[str, float, float]
RewardValue = float | int | bool | str | List[float] | List[int]
PresetName = Literal["paper_baseline", "rl_robust"]
SoftMode = Literal["relative_mean", "zscore"]


@dataclass(frozen=True)
class LocalEmphasisRewardConfig:
    """Configuration for :class:`LocalEmphasisReward`.

    Hard rewards are sparse, so ``gamma`` and ``delta`` default to small values.
    """

    preset: PresetName = "rl_robust"
    n_fft: int = 1024
    hop_length: int = 256
    f0_min: float = 50.0
    f0_max: float = 500.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.1
    delta: float = 0.1
    eps: float = 1e-6
    use_log_energy: bool = True
    soft_mode: SoftMode = "zscore"
    min_voiced_ratio: float = 0.2
    lambda_pitch_rank: float = 0.2
    lambda_energy_rank: float = 0.2
    include_ranking_debug: bool = True

    @staticmethod
    def preset_values(preset: PresetName) -> Dict[str, float | int | bool | str]:
        """Return config values for a named reward preset."""

        if preset == "paper_baseline":
            return {
                "preset": "paper_baseline",
                "use_log_energy": False,
                "soft_mode": "relative_mean",
                "min_voiced_ratio": 0.0,
                "lambda_pitch_rank": 0.0,
                "lambda_energy_rank": 0.0,
            }
        if preset == "rl_robust":
            return {
                "preset": "rl_robust",
                "use_log_energy": True,
                "soft_mode": "zscore",
                "min_voiced_ratio": 0.2,
                "lambda_pitch_rank": 0.2,
                "lambda_energy_rank": 0.2,
            }
        raise ValueError(f"Unknown preset: {preset}")


class LocalEmphasisReward:
    """Compute local emphasis control reward from waveform and word timings.

    Args:
        config: Optional configuration object. Individual keyword arguments can
            also be supplied to override fields in the default config.

    Input:
        wav: ``torch.Tensor`` of shape ``[T]`` or ``[1, T]``.
        sr: Sampling rate in Hz.
        word_boundaries: Sequence of ``(word, start_sec, end_sec)`` tuples.
        target_idx: Index of the desired emphasized word.

    Returns:
        A dictionary containing total reward, hard/soft components, per-word
        pitch and energy features, and the target word. When
        ``include_ranking_debug`` is enabled, the dictionary also includes
        ``pitch_rank_soft`` and ``energy_rank_soft``.

    Raises:
        ValueError: If inputs are malformed or ``target_idx`` is invalid.
        RuntimeError: If neither librosa.pyin nor pyworld is available for F0.
    """

    def __init__(
        self,
        config: LocalEmphasisRewardConfig | None = None,
        **overrides: float | int | bool | str,
    ) -> None:
        preset = overrides.pop("preset", None)
        if config is None:
            preset_name = preset or LocalEmphasisRewardConfig().preset
            config_values = {
                **LocalEmphasisRewardConfig().__dict__,
                **LocalEmphasisRewardConfig.preset_values(preset_name),  # type: ignore[arg-type]
            }
            config = LocalEmphasisRewardConfig(**config_values)  # type: ignore[arg-type]
        elif preset is not None:
            config = LocalEmphasisRewardConfig(
                **{
                    **config.__dict__,
                    **LocalEmphasisRewardConfig.preset_values(preset),  # type: ignore[arg-type]
                }
            )
        if overrides:
            valid_fields = set(LocalEmphasisRewardConfig.__dataclass_fields__)
            unknown = set(overrides) - valid_fields
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown LocalEmphasisRewardConfig field(s): {names}")
            config = LocalEmphasisRewardConfig(
                **{**config.__dict__, **overrides}  # type: ignore[arg-type]
            )
        self.config = config

    def __call__(
        self,
        wav: torch.Tensor,
        sr: int,
        word_boundaries: Sequence[WordBoundary],
        target_idx: int,
    ) -> Dict[str, RewardValue]:
        return self.compute(wav, sr, word_boundaries, target_idx)

    def compute(
        self,
        wav: torch.Tensor,
        sr: int,
        word_boundaries: Sequence[WordBoundary],
        target_idx: int,
    ) -> Dict[str, RewardValue]:
        """Compute local emphasis reward for one utterance."""

        wav_1d = self._validate_inputs(wav, sr, word_boundaries, target_idx)
        boundaries = self._sanitize_boundaries(word_boundaries, wav_1d.numel(), sr)

        frame_pitch = self._extract_f0(wav_1d, sr)
        frame_energy = self._extract_energy(wav_1d)

        word_pitch, word_voiced_ratio = self._word_pitch_features(
            boundaries,
            frame_pitch,
            sr,
        )
        word_energy = self._word_energy_features(boundaries, frame_energy, sr)
        normalized_word_pitch, pitch_norm_valid = self._zscore_normalize(word_pitch)
        normalized_word_energy, energy_norm_valid = self._zscore_normalize(word_energy)

        cfg = self.config
        pitch_valid = (
            cfg.min_voiced_ratio <= 0.0
            or word_voiced_ratio[target_idx] >= cfg.min_voiced_ratio
        )
        pitch_confidence = 1.0 if cfg.min_voiced_ratio <= 0.0 else word_voiced_ratio[target_idx]
        pitch_soft, pitch_soft_valid = self._soft_reward(word_pitch, target_idx)
        energy_soft, energy_valid = self._soft_reward(word_energy, target_idx)
        pitch_hard = self._hard_max_reward(word_pitch, target_idx)
        energy_hard = self._hard_max_reward(word_energy, target_idx)
        pitch_rank_soft = self._ranking_soft_reward(normalized_word_pitch, target_idx)
        energy_rank_soft = self._ranking_soft_reward(normalized_word_energy, target_idx)
        energy_confidence = 0.5 if word_voiced_ratio[target_idx] == 0.0 else 1.0

        if not pitch_valid:
            pitch_soft = 0.0
            pitch_hard = 0.0
            pitch_rank_soft = 0.0
            pitch_confidence = 0.0
        else:
            pitch_soft *= pitch_confidence
            pitch_hard *= pitch_confidence
            pitch_rank_soft *= pitch_confidence
        energy_soft *= energy_confidence
        energy_rank_soft *= energy_confidence

        total = (
            cfg.alpha * pitch_soft
            + cfg.beta * energy_soft
            + cfg.gamma * pitch_hard
            + cfg.delta * energy_hard
            + cfg.lambda_pitch_rank * pitch_rank_soft
            + cfg.lambda_energy_rank * energy_rank_soft
        )

        result: Dict[str, RewardValue] = {
            "total": float(total),
            "pitch_soft": float(pitch_soft),
            "energy_soft": float(energy_soft),
            "pitch_hard": float(pitch_hard),
            "energy_hard": float(energy_hard),
            "word_pitch": [float(x) for x in word_pitch],
            "word_energy": [float(x) for x in word_energy],
            "word_voiced_ratio": [float(x) for x in word_voiced_ratio],
            "target_voiced_ratio": float(word_voiced_ratio[target_idx]),
            "pitch_confidence": float(pitch_confidence),
            "energy_confidence": float(energy_confidence),
            "normalized_word_pitch": [float(x) for x in normalized_word_pitch],
            "normalized_word_energy": [float(x) for x in normalized_word_energy],
            "pitch_rank_soft": float(pitch_rank_soft),
            "energy_rank_soft": float(energy_rank_soft),
            "energy_mode": "log" if cfg.use_log_energy else "raw",
            "soft_mode": cfg.soft_mode,
            "pitch_valid": bool(pitch_valid),
            "energy_valid": bool(energy_valid and energy_norm_valid),
            "target_word": str(boundaries[target_idx][0]),
        }

        return result

    def _validate_inputs(
        self,
        wav: torch.Tensor,
        sr: int,
        word_boundaries: Sequence[WordBoundary],
        target_idx: int,
    ) -> torch.Tensor:
        if not isinstance(wav, torch.Tensor):
            raise ValueError("wav must be a torch.Tensor.")
        if wav.ndim == 2:
            if wav.shape[0] != 1:
                raise ValueError("wav with shape [B, T] is only supported for B=1.")
            wav = wav.squeeze(0)
        elif wav.ndim != 1:
            raise ValueError("wav must have shape [T] or [1, T].")
        if wav.numel() == 0:
            raise ValueError("wav must be non-empty.")
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("sr must be a positive integer.")
        if not word_boundaries:
            raise ValueError("word_boundaries must contain at least one word.")
        if target_idx < 0 or target_idx >= len(word_boundaries):
            raise ValueError(
                f"target_idx={target_idx} is out of range for "
                f"{len(word_boundaries)} word boundaries."
            )
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be positive.")
        if self.config.n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        if self.config.soft_mode not in {"relative_mean", "zscore"}:
            raise ValueError("soft_mode must be 'relative_mean' or 'zscore'.")
        if self.config.min_voiced_ratio < 0.0 or self.config.min_voiced_ratio > 1.0:
            raise ValueError("min_voiced_ratio must be in [0, 1].")
        return wav.detach()

    def _sanitize_boundaries(
        self,
        word_boundaries: Sequence[WordBoundary],
        wav_len: int,
        sr: int,
    ) -> List[WordBoundary]:
        duration = wav_len / float(sr)
        clean: List[WordBoundary] = []
        for idx, boundary in enumerate(word_boundaries):
            if len(boundary) != 3:
                raise ValueError(
                    f"word_boundaries[{idx}] must be (word, start_sec, end_sec)."
                )
            word, start, end = boundary
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"word_boundaries[{idx}] has non-numeric start/end times."
                ) from exc

            start_f = max(0.0, min(duration, start_f))
            end_f = max(0.0, min(duration, end_f))
            if end_f < start_f:
                start_f, end_f = end_f, start_f
            clean.append((str(word), start_f, end_f))
        return clean

    def _extract_f0(self, wav: torch.Tensor, sr: int) -> List[float]:
        wav_np = wav.float().cpu().numpy()

        try:
            import librosa

            f0, _, _ = librosa.pyin(
                wav_np,
                fmin=self.config.f0_min,
                fmax=self.config.f0_max,
                sr=sr,
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length,
            )
            return [
                0.0 if value != value or value <= 0.0 else float(value)
                for value in f0
            ]
        except ImportError as librosa_error:
            try:
                import pyworld

                f0, _ = pyworld.dio(
                    wav_np.astype("float64"),
                    sr,
                    f0_floor=self.config.f0_min,
                    f0_ceil=self.config.f0_max,
                    frame_period=1000.0 * self.config.hop_length / sr,
                )
                return [0.0 if value <= 0.0 else float(value) for value in f0]
            except ImportError as pyworld_error:
                raise RuntimeError(
                    "F0 extraction requires either librosa or pyworld to be installed."
                ) from pyworld_error
            except Exception as exc:
                raise RuntimeError("pyworld F0 extraction failed.") from exc
        except Exception as exc:
            raise RuntimeError("librosa.pyin F0 extraction failed.") from exc

    def _extract_energy(self, wav: torch.Tensor) -> List[float]:
        cfg = self.config
        wav_for_stft = wav.float()
        if wav_for_stft.numel() < cfg.n_fft:
            wav_for_stft = F.pad(wav_for_stft, (0, cfg.n_fft - wav_for_stft.numel()))
        window = torch.hann_window(
            cfg.n_fft,
            dtype=wav_for_stft.dtype,
            device=wav_for_stft.device,
        )
        spec = torch.stft(
            wav_for_stft,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=window,
            return_complex=True,
        )
        energy = torch.linalg.vector_norm(spec.abs(), ord=2, dim=0)
        if cfg.use_log_energy:
            energy = torch.log(energy + cfg.eps)
        return [float(x) for x in energy.detach().cpu().tolist()]

    def _word_pitch_features(
        self,
        boundaries: Sequence[WordBoundary],
        frame_pitch: Sequence[float],
        sr: int,
    ) -> Tuple[List[float], List[float]]:
        features: List[float] = []
        voiced_ratios: List[float] = []
        for _, start, end in boundaries:
            start_idx, end_idx = self._time_to_frame_slice(start, end, sr, len(frame_pitch))
            raw_segment = frame_pitch[start_idx:end_idx]
            voiced_segment = [x for x in raw_segment if x > 0.0]
            if not raw_segment:
                voiced_ratios.append(0.0)
            else:
                voiced_ratios.append(float(len(voiced_segment) / len(raw_segment)))
            if not voiced_segment:
                features.append(0.0)
            else:
                features.append(float(log(max(voiced_segment))))
        return features, voiced_ratios

    def _word_energy_features(
        self,
        boundaries: Sequence[WordBoundary],
        frame_energy: Sequence[float],
        sr: int,
    ) -> List[float]:
        features: List[float] = []
        for _, start, end in boundaries:
            start_idx, end_idx = self._time_to_frame_slice(start, end, sr, len(frame_energy))
            segment = frame_energy[start_idx:end_idx]
            if not segment:
                features.append(0.0)
            else:
                features.append(float(sum(segment) / len(segment)))
        return features

    def _time_to_frame_slice(
        self,
        start_sec: float,
        end_sec: float,
        sr: int,
        feature_len: int,
    ) -> Tuple[int, int]:
        if feature_len <= 0:
            return 0, 0
        start_idx = int(start_sec * sr / self.config.hop_length)
        end_idx = int(end_sec * sr / self.config.hop_length)
        start_idx = max(0, min(feature_len, start_idx))
        end_idx = max(0, min(feature_len, end_idx))
        if end_idx <= start_idx and start_sec < end_sec:
            end_idx = min(feature_len, start_idx + 1)
        return start_idx, end_idx

    def _soft_reward(self, values: Sequence[float], target_idx: int) -> Tuple[float, bool]:
        if not values:
            return 0.0, False
        mean_value = sum(values) / len(values)
        if self.config.soft_mode == "relative_mean":
            if abs(mean_value) <= self.config.eps:
                return 0.0, False
            raw = (values[target_idx] - mean_value) / (mean_value + self.config.eps)
            return self._clip_unit(raw), True

        std_value = self._std(values, mean_value)
        if std_value <= self.config.eps:
            return 0.0, False
        raw = (values[target_idx] - mean_value) / (std_value + self.config.eps)
        return self._clip_unit(raw), True

    @staticmethod
    def _hard_max_reward(values: Sequence[float], target_idx: int) -> float:
        if not values:
            return 0.0
        return 1.0 if values[target_idx] >= max(values) else 0.0

    @staticmethod
    def _ranking_soft_reward(values: Sequence[float], target_idx: int) -> float:
        if len(values) <= 1:
            return 0.0
        target = float(values[target_idx])
        max_other = max(float(v) for i, v in enumerate(values) if i != target_idx)
        diff = torch.tensor(target - max_other, dtype=torch.float32)
        return float(torch.sigmoid(diff).item())

    def _zscore_normalize(self, values: Sequence[float]) -> Tuple[List[float], bool]:
        if not values:
            return [], False
        mean_value = sum(values) / len(values)
        std_value = self._std(values, mean_value)
        if std_value <= self.config.eps:
            return [0.0 for _ in values], False
        return [float((value - mean_value) / (std_value + self.config.eps)) for value in values], True

    @staticmethod
    def _std(values: Sequence[float], mean_value: float) -> float:
        if not values:
            return 0.0
        variance = sum((float(value) - mean_value) ** 2 for value in values) / len(values)
        return float(sqrt(max(variance, 0.0)))

    @staticmethod
    def _clip_unit(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))


if __name__ == "__main__":
    # Minimal example with synthetic audio and coarse word boundaries.
    sample_rate = 16_000
    t = torch.linspace(0.0, 1.2, int(sample_rate * 1.2))
    wav = 0.05 * torch.sin(2.0 * torch.pi * 180.0 * t)
    wav[int(0.4 * sample_rate) : int(0.8 * sample_rate)] *= 3.0

    boundaries = [
        ("I", 0.0, 0.35),
        ("really", 0.35, 0.85),
        ("care", 0.85, 1.2),
    ]

    reward_fn = LocalEmphasisReward()
    reward = reward_fn(wav, sample_rate, boundaries, target_idx=1)
    print(reward)
