#!/usr/bin/env python3
"""IndexTTS inference using a GRPO-trained checkpoint.

改寫自 repo/baseline/indexTTS_gen.py:
  - 輸入:Kaldi text(每行 "utt_id 台羅拼音"),不再吃 jsonl,不需翻譯
  - 模型:先用 base GPT 起 IndexTTS2,再 overlay 訓練後的 GRPO checkpoint
            (結構 {"step", "model": gpt.state_dict(), "va_alpha_mlp": ...})
  - 輸出:wav + manifest.json,**不算 reward**

用法(從 /srv/RL_project):
    /srv/RL_project/repo/index-tts/.venv/bin/python \\
        -m repo.train.eval.indextts_grpo_eval \\
        --input TAT-Vol1/TAT-Vol1-test_manifest_enhanced/text \\
        --grpo-ckpt repo/train/runs/try1_arc/step_1169.pth \\
        --output-root repo/train/runs/eval/step1169_test_manifest/wav \\
        --index-tts-dir repo/index-tts
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

logging.disable(logging.WARNING)
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")

EMOTIONS = ["angry", "happy", "fearful", "disgusted", "sad", "surprised", "neutral"]

# 8-dim:[joy, anger, sadness, fear, disgust, low_mood, surprise, neutral]
EMOTION_VECTORS = {
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
    p.add_argument("--output-root", default="repo/train/runs/eval/step1169_test_manifest/wav")
    p.add_argument("--index-tts-dir", default="repo/index-tts", help="Path to index-tts repo root.")
    p.add_argument("--config", default="checkpoints/config.yaml", help="Relative to --index-tts-dir.")
    p.add_argument("--base-gpt",
                   default="/srv/RL_project/models/IndexTTS_trained/GPTs/trained_200hr_4e_step202000.pth",
                   help="Base GPT for IndexTTS2 init (raw state_dict).")
    p.add_argument("--grpo-ckpt",
                   default="/srv/RL_project/repo/train/runs/try1_arc/step_1169.pth",
                   help="GRPO training checkpoint to overlay (has 'model' + 'va_alpha_mlp' keys).")
    p.add_argument("--tokenizer",
                   default="/srv/RL_project/models/IndexTTS_trained/checkpoints/bpe_extended.model")
    p.add_argument("--speaker", default="examples/prompts/FT0BAE.mp3",
                   help="Relative to --index-tts-dir if not absolute.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--emotion-seed", type=int, default=42)
    p.add_argument("--retry-count", type=int, default=2)
    # generation params (mirror indexTTS_gen.py defaults)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--max-text-tokens", type=int, default=120)
    p.add_argument("--max-mel-tokens", type=int, default=1700)
    p.add_argument("--repetition-penalty", type=float, default=10.0)
    p.add_argument("--length-penalty", type=float, default=0.0)
    p.add_argument("--interval-silence", type=int, default=200)
    p.add_argument("--model-name", default="indextts_step1169",
                   help="Short name used in filenames and manifest.")
    p.add_argument("--manifest", default=None,
                   help="Path to manifest JSON. Defaults to <output-root>/manifest.json.")
    p.add_argument("--trim-top-db", type=float, default=30.0)
    p.add_argument("--trim-pad-ms", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="Only process first N lines (for smoke test).")
    return p.parse_args()


def load_tasks(path: Path, limit: int | None = None) -> list[dict]:
    tasks: list[dict] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if limit is not None and len(tasks) >= limit:
            break
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Line {idx} missing text: {line!r}")
        tasks.append({"id": parts[0], "text": parts[1]})
    if not tasks:
        raise ValueError(f"No tasks from {path}")
    return tasks


def assign_emotions(tasks: list[dict], rng: random.Random) -> None:
    n = len(tasks)
    pool = (EMOTIONS * math.ceil(n / len(EMOTIONS)))[:n]
    rng.shuffle(pool)
    for t, e in zip(tasks, pool):
        t["emotion"] = e


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d} ({sec:.1f}s)"


def save_manifest(manifest_path: Path, records: list, model: str) -> None:
    existing: list = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing = [r for r in existing if r.get("model") != model]
    existing.extend(records)
    manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    import torch
    args = parse_args()

    index_tts_dir = Path(args.index_tts_dir).expanduser().resolve()
    if not index_tts_dir.exists():
        raise FileNotFoundError(f"index-tts dir not found: {index_tts_dir}")
    if str(index_tts_dir) not in sys.path:
        sys.path.insert(0, str(index_tts_dir))

    from indextts.infer_v2_modded import IndexTTS2
    from omegaconf import OmegaConf

    # Optional silence trim from baseline/
    trim_silence = None
    try:
        baseline_dir = Path(__file__).resolve().parents[2] / "baseline"
        if str(baseline_dir) not in sys.path:
            sys.path.insert(0, str(baseline_dir))
        from trim_silence import trim_wav as _trim
        trim_silence = _trim
    except Exception as exc:
        print(f"[WARN] trim_silence unavailable: {exc}")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(input_path, limit=args.limit)
    assign_emotions(tasks, random.Random(args.emotion_seed))
    print(f"[INFO] {len(tasks)} tasks from {input_path}")

    cfg_path = (index_tts_dir / args.config).resolve()
    base_gpt = Path(args.base_gpt).expanduser().resolve()
    tokenizer = Path(args.tokenizer).expanduser().resolve()
    speaker = Path(args.speaker)
    if not speaker.is_absolute():
        speaker = (index_tts_dir / speaker).resolve()

    # Build a temp config that points at base GPT + tokenizer
    cfg = OmegaConf.load(cfg_path)
    cfg.gpt_checkpoint = str(base_gpt)
    cfg.dataset["bpe_model"] = str(tokenizer)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        OmegaConf.save(cfg, tmp.name)
        tmp_cfg_path = tmp.name

    print(f"[INFO] loading base GPT from {base_gpt}")
    engine = IndexTTS2(
        cfg_path=tmp_cfg_path,
        model_dir=str(index_tts_dir / "checkpoints"),
        device=args.device,
        use_fp16=True,
        use_accel=True,
        use_cuda_kernel=False,
        gpt_checkpoint_path=str(base_gpt),
        bpe_model_path=str(tokenizer),
    )

    # Overlay GRPO checkpoint onto engine.gpt
    grpo_ckpt = Path(args.grpo_ckpt).expanduser().resolve()
    if not grpo_ckpt.exists():
        raise FileNotFoundError(f"GRPO ckpt not found: {grpo_ckpt}")
    print(f"[INFO] overlaying GRPO checkpoint: {grpo_ckpt}")
    try:
        ckpt = torch.load(str(grpo_ckpt), map_location=args.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(grpo_ckpt), map_location=args.device)
    if "model" not in ckpt:
        raise KeyError(f"Expected key 'model' in {grpo_ckpt}, got: {list(ckpt.keys())}")
    missing, unexpected = engine.gpt.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] gpt overlay: step={ckpt.get('step')} missing={len(missing)} unexpected={len(unexpected)}")
    engine.gpt.eval()
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
    records: list = []
    wall_start = time.perf_counter()

    for i, task in enumerate(tasks, start=1):
        out_path = output_root / f"{i:06d}_{args.model_name}_{task['emotion']}.wav"
        vec = EMOTION_VECTORS[task["emotion"]]

        if out_path.exists():
            dur = get_wav_duration(out_path)
            total_audio_sec += dur
            print(f"  [{i}/{len(tasks)}] {task['id']}  [SKIP existing]  {dur:.2f}s  cum={total_audio_sec:.1f}s")
            records.append({"index": i, "id": task["id"], "model": args.model_name,
                            "text": task["text"], "emotion": task["emotion"],
                            "file": out_path.name, "duration_sec": round(dur, 3)})
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
            records.append({"index": i, "id": task["id"], "model": args.model_name,
                            "text": task["text"], "emotion": task["emotion"],
                            "file": out_path.name, "duration_sec": None})
            continue

        if trim_silence is not None:
            try:
                trim_silence(out_path, out_path, top_db=args.trim_top_db, pad_ms=args.trim_pad_ms)
            except Exception as exc:
                print(f"[WARN] trim failed for {task['id']}: {exc}")

        dur = get_wav_duration(out_path)
        total_audio_sec += dur
        print(f"  [{i}/{len(tasks)}] {task['id']}  {task['emotion']}  {dur:.2f}s  cum={total_audio_sec:.1f}s")
        records.append({"index": i, "id": task["id"], "model": args.model_name,
                        "text": task["text"], "emotion": task["emotion"],
                        "file": out_path.name, "duration_sec": round(dur, 3)})

    wall_elapsed = time.perf_counter() - wall_start
    manifest_path = (Path(args.manifest).expanduser().resolve()
                     if args.manifest else output_root / "manifest.json")
    save_manifest(manifest_path, records, args.model_name)
    print(f"[INFO] manifest saved to {manifest_path}")
    print("\n" + "=" * 60)
    print(f"  wall={fmt(wall_elapsed)}  audio={fmt(total_audio_sec)}  failed={fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
