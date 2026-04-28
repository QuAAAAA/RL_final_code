#!/usr/bin/env python3
"""Direct batch inference using IndexTTS2 (loads model once, no Gradio server needed).
Generates all texts in text_origin once with random emotions.
Prints wall time and total audio duration at the end.

python baseline/indexTTS_gen.py \
  --input baseline/text_origin \
  --output-root outputs/wav \
  --manifest outputs/manifest_indextts.json

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
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Kaldi text file (utt_id text...).")
    p.add_argument("--output-root", default="./outputs/indextts")
    p.add_argument("--index-tts-dir", default="index-tts", help="Path to index-tts repo root.")
    p.add_argument("--config", default="checkpoints/config.yaml", help="Relative to --index-tts-dir.")
    p.add_argument("--gpt-checkpoint", default="/srv/RL_project/models/IndexTTS_trained/GPTs/trained_200hr_4e_step202000.pth")
    p.add_argument("--tokenizer", default="/srv/RL_project/models/IndexTTS_trained/checkpoints/bpe_extended.model")
    p.add_argument("--speaker", default="examples/prompts/FT0BAE.mp3", help="Relative to --index-tts-dir.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emotion-seed", type=int, default=None)
    p.add_argument("--retry-count", type=int, default=2)
    # generation params (mirrors run_inference.sh defaults)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-text-tokens", type=int, default=120)
    p.add_argument("--max-mel-tokens", type=int, default=1500)
    p.add_argument("--repetition-penalty", type=float, default=10.0)
    p.add_argument("--length-penalty", type=float, default=0.0)
    p.add_argument("--interval-silence", type=int, default=200)
    p.add_argument("--model-name", default="indextts", help="Short model name recorded in manifest and filenames.")
    p.add_argument("--manifest", default=None, help="Path to manifest JSON. Defaults to <output-root>/manifest.json.")
    return p.parse_args()


def load_tasks(path: Path) -> List[Dict[str, str]]:
    tasks = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Line {idx} missing text: {line!r}")
        tasks.append({"id": parts[0], "text": parts[1]})
    if not tasks:
        raise ValueError(f"No tasks from {path}")
    return tasks


def assign_emotions(tasks: List[Dict[str, str]], rng: random.Random) -> None:
    n = len(tasks)
    pool = (EMOTIONS * math.ceil(n / len(EMOTIONS)))[:n]
    rng.shuffle(pool)
    for task, emotion in zip(tasks, pool):
        task["emotion"] = emotion


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


def main() -> None:
    args = parse_args()

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

    tasks = load_tasks(input_path)
    assign_emotions(tasks, random.Random(args.emotion_seed))
    print(f"[INFO] {len(tasks)} tasks from {input_path}")

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
        vec = EMOTION_VECTORS[task["emotion"]]

        if out_path.exists():
            dur = get_wav_duration(out_path)
            total_audio_sec += dur
            print(f"  [{i}/{len(tasks)}] {task['id']}  [SKIP existing]  {dur:.2f}s  cum={total_audio_sec:.1f}s")
            records.append({"index": i, "model": args.model_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": round(dur, 3)})
            continue

        success = False
        for attempt in range(1, args.retry_count + 2):
            try:
                engine.infer(
                    spk_audio_prompt=str(speaker),
                    text=task["text"],
                    output_path=str(out_path),
                    emo_vector=vec,
                    interval_silence=args.interval_silence,
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
            records.append({"index": i, "model": args.model_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": None})
            continue

        dur = get_wav_duration(out_path)
        total_audio_sec += dur
        print(f"  [{i}/{len(tasks)}] {task['id']}  {task['emotion']}  {dur:.2f}s  cum={total_audio_sec:.1f}s")
        records.append({"index": i, "model": args.model_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": round(dur, 3)})

    wall_elapsed = time.perf_counter() - wall_start

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_root / "manifest.json"
    save_manifest(manifest_path, records, args.model_name)
    print(f"[INFO] manifest saved to {manifest_path}")

    print("\n" + "=" * 60)
    print(f"  wall={fmt(wall_elapsed)}  audio={fmt(total_audio_sec)}  failed={fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
