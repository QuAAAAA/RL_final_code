#!/usr/bin/env python3
"""Direct batch inference using IndexTTS2 (loads model once, no Gradio server needed).
Generates all texts in a Kaldi text file or VA/tagged JSONL with emotion controls.
Prints wall time and total audio duration at the end.

cd /srv/RL_project/repo/index-tts
uv run python ../baseline/indexTTS_gen.py \
    --input ../baseline/text_origin \
    --gpt-checkpoint /srv/RL_project/repo/train/runs/try7_arc/top_reward_161-456.pth \
    --output-root ../outputs/wav_by_model/indextts_try7 \
    --manifest ../outputs/manifest_indextts_try7.json \
    --index-tts-dir .

# JSONL with TaggedText / Quadruplet VA:
uv run python ../baseline/indexTTS_gen.py \
    --input /srv/RL_project/repo/baseline/ensemble_tat_test_task3_least.jsonl \
    --input-format jsonl \
    --text-mode tagged \
    --emotion-mode auto \
    --output-root ../outputs/wav_by_model/indextts_va \
    --manifest ../outputs/manifest_indextts_va.json \
    --index-tts-dir .

Notes:
IndexTTS，耗時: wall=01:56:45 (7005.6s)
共2752筆資料，生成資料集總長: 09:21:57 (33717.5s)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from trim_silence import trim_wav as trim_silence
    TRIM_IMPORT_ERROR = None
except Exception as exc:
    trim_silence = None
    TRIM_IMPORT_ERROR = exc

logging.disable(logging.WARNING)
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")

EMOTIONS = ["angry", "happy", "fearful", "disgusted", "sad", "surprised", "neutral"]

# 8-dim vector: [joy, anger, sadness, fear, disgust, low_mood, surprise, neutral]
EMOTION_VECTORS: Dict[str, List[float]] = {
    "angry":     [0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "happy":     [1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fearful":   [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0],
    "disgusted": [0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0],
    "sad":       [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0],
    "neutral":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

JsonDict = Dict[str, Any]
Task = Dict[str, Any]
VaScaler = Callable[[List[float], Optional[tuple[float, float]]], tuple[List[float], Optional[float]]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Kaldi text file (utt_id text...) or VA/tagged JSONL.")
    p.add_argument("--input-format", choices=["auto", "kaldi", "jsonl"], default="auto")
    p.add_argument(
        "--text-mode",
        choices=["text", "tagged"],
        default="tagged",
        help="For JSONL input, generate from Text or TaggedText (default: tagged).",
    )
    p.add_argument("--output-root", default="./outputs/indextts")
    p.add_argument("--index-tts-dir", default="index-tts", help="Path to index-tts repo root.")
    p.add_argument("--config", default="checkpoints/config.yaml", help="Relative to --index-tts-dir.")
    p.add_argument("--gpt-checkpoint", default="/srv/RL_project/models/IndexTTS_trained/GPTs/trained_200hr_4e_step202000.pth")
    p.add_argument("--tokenizer", default="/srv/RL_project/models/IndexTTS_trained/checkpoints/bpe_extended.model")
    p.add_argument("--speaker", default="examples/prompts/FT0BAE.mp3", help="Relative to --index-tts-dir.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emotion-seed", type=int, default=None)
    p.add_argument(
        "--emotion-mode",
        choices=["auto", "random", "va"],
        default="auto",
        help="auto uses VA-derived emotion when JSONL VA exists, otherwise random.",
    )
    p.add_argument(
        "--disable-va-alpha",
        action="store_true",
        help="Do not scale emotion vectors with va_alpha_mlp if the checkpoint contains one.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N tasks.")
    p.add_argument("--retry-count", type=int, default=2)
    # generation params (mirrors run_inference.sh defaults)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-text-tokens", type=int, default=120)
    p.add_argument("--max-mel-tokens", type=int, default=1700)
    p.add_argument("--repetition-penalty", type=float, default=10.0)
    p.add_argument("--length-penalty", type=float, default=0.0)
    p.add_argument("--interval-silence", type=int, default=200)
    p.add_argument("--duration-seconds", type=float, default=10.0,
                   help="Target duration (s) for generated audio; enables the engine's internal "
                        "length planning (matches run_inference.sh). Use <=0 / None to disable.")
    p.add_argument("--model-name", default="indextts", help="Short model name recorded in manifest and filenames.")
    p.add_argument("--manifest", default=None, help="Path to manifest JSON. Defaults to <output-root>/manifest.json.")
    p.add_argument("--trim-top-db", type=float, default=40.0, help="Silence trim threshold in dB (default: 40).")
    p.add_argument("--trim-pad-ms", type=int, default=50, help="Padding kept on each side after trim (default: 50ms).")
    return p.parse_args()


def detect_input_format(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                json.loads(line)
                return "jsonl"
            except json.JSONDecodeError:
                pass
        return "kaldi"
    raise ValueError(f"No tasks from {path}")


def parse_va(value: object) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    match = re.match(r"\s*([0-9.]+)\s*#\s*([0-9.]+)\s*$", str(value))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def mean_va(quadruplets: List[JsonDict]) -> Optional[tuple[float, float]]:
    vals = [va for va in (parse_va(q.get("VA")) for q in quadruplets) if va is not None]
    if not vals:
        return None
    return sum(v for v, _ in vals) / len(vals), sum(a for _, a in vals) / len(vals)


def va_to_unit(va: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """Map dataset VA from 1-9 scale to the 0-1 scale used by GRPO training."""
    if va is None:
        return None
    valence, arousal = va
    return (valence - 1.0) / 8.0, (arousal - 1.0) / 8.0


def va_to_emotion(va: Optional[tuple[float, float]]) -> Optional[str]:
    """Same thresholds as repo/train/dataset.py, mapped to this script's labels."""
    if va is None:
        return None
    valence, arousal = va
    if valence >= 5.8 and arousal >= 5.8:
        return "happy"
    if valence <= 4.0 and arousal >= 5.4:
        return "angry"
    if valence <= 4.2 and arousal <= 4.8:
        return "sad"
    if valence <= 4.8 and arousal >= 6.2:
        return "fearful"
    if valence <= 4.5:
        return "disgusted"
    if arousal >= 6.5:
        return "surprised"
    return "neutral"


def emphasis_targets(quadruplets: List[JsonDict]) -> List[JsonDict]:
    targets: List[JsonDict] = []
    for item in quadruplets:
        opinion = str(item.get("Opinion") or "").strip()
        aspect = str(item.get("Aspect") or "").strip()
        if not opinion:
            continue
        va = parse_va(item.get("VA"))
        strength = "medium"
        if va is not None:
            valence, arousal = va
            if arousal >= 6.2 or abs(valence - 4.0) >= 2.0:
                strength = "strong"
            elif arousal <= 4.8:
                strength = "weak"
        targets.append({"word": opinion, "aspect": aspect, "strength": strength, "va": item.get("VA")})
    return targets


def message_text(item: JsonDict) -> str:
    for message in reversed(list(item.get("messages") or [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def load_kaldi_tasks(path: Path, limit: Optional[int]) -> List[Task]:
    tasks: List[Task] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if limit is not None and len(tasks) >= limit:
            break
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Line {idx} missing text: {line!r}")
        text = parts[1]
        tasks.append({
            "id": parts[0],
            "text": text,
            "tagged_text": text,
            "generation_text": text,
            "quadruplets": [],
            "va": None,
            "va_01": None,
            "emphasis_targets": [],
        })
    if not tasks:
        raise ValueError(f"No tasks from {path}")
    return tasks


def load_jsonl_tasks(path: Path, text_mode: str, limit: Optional[int]) -> List[Task]:
    tasks: List[Task] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if limit is not None and len(tasks) >= limit:
            break
        item = json.loads(line)
        text = str(item.get("Text") or message_text(item))
        tagged_text = str(item.get("TaggedText") or text)
        quadruplets = list(item.get("Quadruplet") or [])
        va = mean_va(quadruplets)
        va_01 = va_to_unit(va)
        generation_text = tagged_text if text_mode == "tagged" else text
        emotion = va_to_emotion(va)
        tasks.append({
            "id": str(item.get("ID") or f"line-{idx}"),
            "text": text,
            "tagged_text": tagged_text,
            "generation_text": generation_text,
            "quadruplets": quadruplets,
            "va": va,
            "va_01": va_01,
            "emotion": emotion,
            "emotion_source": "va" if emotion is not None else None,
            "emphasis_targets": emphasis_targets(quadruplets),
        })
    if not tasks:
        raise ValueError(f"No tasks from {path}")
    return tasks


def load_tasks(path: Path, input_format: str, text_mode: str, limit: Optional[int]) -> tuple[List[Task], str]:
    resolved_format = detect_input_format(path) if input_format == "auto" else input_format
    if resolved_format == "jsonl":
        return load_jsonl_tasks(path, text_mode=text_mode, limit=limit), resolved_format
    return load_kaldi_tasks(path, limit=limit), resolved_format


def assign_random_emotions(tasks: List[Task], rng: random.Random) -> None:
    pool = (EMOTIONS * math.ceil(len(tasks) / len(EMOTIONS)))[:len(tasks)]
    rng.shuffle(pool)
    for task, emotion in zip(tasks, pool):
        task["emotion"] = emotion
        task["emotion_source"] = "random"


def apply_emotion_mode(tasks: List[Task], mode: str, rng: random.Random) -> None:
    if mode == "random":
        assign_random_emotions(tasks, rng)
        return

    missing = [task for task in tasks if not task.get("emotion")]
    if mode == "va" and missing:
        missing_ids = ", ".join(str(task["id"]) for task in missing[:5])
        raise ValueError(f"--emotion-mode va requires VA for every task; missing: {missing_ids}")

    if missing:
        assign_random_emotions(missing, rng)

    for task in tasks:
        emotion = task.get("emotion")
        if emotion not in EMOTION_VECTORS:
            raise ValueError(f"Unsupported emotion {emotion!r} for task {task.get('id')}")


class VAAlphaMLP:
    """Tiny loader for GRPO's VAAlphaMLP state dict without importing training code."""

    def __init__(self, state_dict: JsonDict) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        stripped = {key.replace("net.", "", 1): value for key, value in state_dict.items()}
        self.model.load_state_dict(stripped)
        self.model.eval()

    def __call__(self, va_01: tuple[float, float]) -> float:
        with self.torch.no_grad():
            va_t = self.torch.tensor(list(va_01), dtype=self.torch.float32)
            return float(self.model(va_t).squeeze(-1).item())


def load_va_scaler(checkpoint_path: Path, disabled: bool) -> Optional[VaScaler]:
    if disabled:
        return None
    try:
        import torch

        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt.get("va_alpha_mlp") if isinstance(ckpt, dict) else None
        if not state:
            return None
        alpha_mlp = VAAlphaMLP(state)
    except Exception as exc:
        print(f"[WARN] could not load va_alpha_mlp from {checkpoint_path}: {exc}")
        return None

    def scale(vec: List[float], va_01: Optional[tuple[float, float]]) -> tuple[List[float], Optional[float]]:
        if va_01 is None or not any(vec):
            return list(vec), None
        alpha = alpha_mlp(va_01)
        return [value * alpha for value in vec], alpha

    return scale


def record_for_task(
    index: int,
    task: Task,
    model: str,
    file_name: str,
    duration_sec: Optional[float],
    emotion_vector: List[float],
    va_alpha: Optional[float],
    input_format: str,
) -> JsonDict:
    return {
        "index": index,
        "id": task["id"],
        "model": model,
        "input_format": input_format,
        "text": task["text"],
        "tagged_text": task.get("tagged_text"),
        "generation_text": task.get("generation_text", task["text"]),
        "emotion": task["emotion"],
        "emotion_source": task.get("emotion_source"),
        "emotion_vector": [round(value, 6) for value in emotion_vector],
        "va": task.get("va"),
        "va_01": task.get("va_01"),
        "va_alpha": round(va_alpha, 6) if va_alpha is not None else None,
        "quadruplets": task.get("quadruplets", []),
        "emphasis_targets": task.get("emphasis_targets", []),
        "file": file_name,
        "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
    }


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d} ({sec:.1f}s)"


def save_manifest(manifest_path: Path, records: list, model: str) -> None:
    existing = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing = [r for r in existing if r.get("model") != model]
    existing.extend(records)
    manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_seed(seed: Optional[int]) -> None:
    """Seed python / numpy / torch so sampling is reproducible (mirrors inference_script.py)."""
    if seed is None:
        return
    import numpy as np
    import torch

    py_seed = int(seed % (2**32))
    random.seed(py_seed)
    np.random.seed(py_seed)
    torch_seed = int(seed % (2**63 - 1))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    index_tts_dir = Path(args.index_tts_dir).expanduser().resolve()
    if not index_tts_dir.exists():
        raise FileNotFoundError(f"index-tts dir not found: {index_tts_dir}")

    # Make indextts importable
    if str(index_tts_dir) not in sys.path:
        sys.path.insert(0, str(index_tts_dir))

    from indextts.infer_v2_modded import IndexTTS2
    from omegaconf import OmegaConf

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    emotion_seed = args.emotion_seed if args.emotion_seed is not None else args.seed
    tasks, input_format = load_tasks(input_path, args.input_format, args.text_mode, args.limit)
    apply_emotion_mode(tasks, args.emotion_mode, random.Random(emotion_seed))
    va_count = sum(1 for task in tasks if task.get("va") is not None)
    tagged_count = sum(1 for task in tasks if task.get("tagged_text") != task.get("text"))
    print(f"[INFO] {len(tasks)} tasks from {input_path} ({input_format}, va={va_count}, tagged={tagged_count})")

    # Resolve paths relative to index_tts_dir
    cfg_path = (index_tts_dir / args.config).resolve()
    gpt_ckpt = (index_tts_dir / args.gpt_checkpoint).resolve()
    tokenizer = (index_tts_dir / args.tokenizer).resolve()
    speaker = (index_tts_dir / args.speaker).resolve()

    # Load config and apply overrides (same as inference_script.py)
    cfg = OmegaConf.load(cfg_path)
    cfg.gpt_checkpoint = str(gpt_ckpt)
    cfg.dataset["bpe_model"] = str(tokenizer)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        OmegaConf.save(cfg, tmp.name)
        tmp_cfg_path = tmp.name

    print(f"[INFO] loading model from {gpt_ckpt} ...")
    engine = IndexTTS2(
        cfg_path=tmp_cfg_path,
        model_dir=str(index_tts_dir / "checkpoints"),
        device=args.device,
        use_fp16=True,
        use_accel=True,
        use_cuda_kernel=False,
        gpt_checkpoint_path=str(gpt_ckpt),
        bpe_model_path=str(tokenizer),
    )
    va_scaler = load_va_scaler(gpt_ckpt, args.disable_va_alpha)
    if va_scaler is not None:
        print("[INFO] loaded va_alpha_mlp; VA will scale emotion vectors")
    if trim_silence is None:
        print(f"[WARN] trim_silence unavailable; generated wavs will not be trimmed: {TRIM_IMPORT_ERROR}")
    print("[INFO] model loaded, starting generation")

    generation_kwargs = {
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "max_mel_tokens": args.max_mel_tokens,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
    }

    total_audio_sec = 0.0
    fail_count = 0
    records = []
    wall_start = time.perf_counter()

    for i, task in enumerate(tasks, start=1):
        out_path = output_root / f"{i:06d}_{args.model_name}_{task['emotion']}.wav"
        base_vec = EMOTION_VECTORS[task["emotion"]]
        if va_scaler is None:
            vec, va_alpha = list(base_vec), None
        else:
            vec, va_alpha = va_scaler(base_vec, task.get("va_01"))
        va_msg = ""
        if task.get("va") is not None:
            va_msg = f"  va={task['va'][0]:.2f}#{task['va'][1]:.2f}"
        alpha_msg = f"  alpha={va_alpha:.3f}" if va_alpha is not None else ""

        if out_path.exists():
            dur = get_wav_duration(out_path)
            total_audio_sec += dur
            print(f"  [{i}/{len(tasks)}] {task['id']}  [SKIP existing]  {dur:.2f}s{va_msg}{alpha_msg}  cum={total_audio_sec:.1f}s")
            records.append(record_for_task(i, task, args.model_name, out_path.name, dur, vec, va_alpha, input_format))
            continue

        # Reseed per utterance so each one starts from the same RNG state, regardless of
        # batch position (matches batch_indexTTS.sh's per-process --seed; resume-independent).
        apply_seed(args.seed)

        success = False
        for attempt in range(1, args.retry_count + 2):
            try:
                engine.infer(
                    spk_audio_prompt=str(speaker),
                    text=task.get("generation_text", task["text"]),
                    output_path=str(out_path),
                    emo_vector=vec,
                    interval_silence=args.interval_silence,
                    duration_seconds=args.duration_seconds,
                    max_text_tokens_per_sentence=args.max_text_tokens,
                    skip_normalizer=True,
                    **generation_kwargs,
                )
                success = True
                break
            except Exception as exc:
                print(f"[WARN] {task['id']} attempt {attempt}: {exc}")

        if not success or not out_path.exists():
            fail_count += 1
            print(f"[SKIP] {task['id']}")
            records.append(record_for_task(i, task, args.model_name, out_path.name, None, vec, va_alpha, input_format))
            continue

        if trim_silence is not None:
            try:
                trim_silence(out_path, out_path, top_db=args.trim_top_db, pad_ms=args.trim_pad_ms)
            except Exception as exc:
                print(f"[WARN] trim failed for {task['id']}: {exc}")

        dur = get_wav_duration(out_path)
        total_audio_sec += dur
        print(f"  [{i}/{len(tasks)}] {task['id']}  {task['emotion']}  {dur:.2f}s{va_msg}{alpha_msg}  cum={total_audio_sec:.1f}s")
        records.append(record_for_task(i, task, args.model_name, out_path.name, dur, vec, va_alpha, input_format))

    wall_elapsed = time.perf_counter() - wall_start

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_root / "manifest.json"
    save_manifest(manifest_path, records, args.model_name)
    print(f"[INFO] manifest saved to {manifest_path}")

    print("\n" + "=" * 60)
    print(f"  wall={fmt(wall_elapsed)}  audio={fmt(total_audio_sec)}  failed={fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
