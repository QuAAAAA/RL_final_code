#!/usr/bin/env python3
"""Generate all texts in text_origin once with A2, then once with A3.
Prints wall time and total audio duration per model at the end. No JSON output.

python3 baseline/batch_generate_ta.py \
  --input baseline/text_origin \
  --output-root outputs/wav \
  --manifest outputs/manifest_ta.json

Note:
A2  耗時=07:57:48 (28668.6s)  生成音檔時長=07:53:41 (28421.5s)  failed=39
A3  耗時=09:43:30 (35010.7s)  生成音檔時長=09:48:08 (35288.6s)  failed=5
總耗時 : 17:41:19 (63679.3s)
總生成音檔時長: 17:41:50 (63710.1s)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_AUDIO = "index-tts/examples/prompts/FT0BAE.mp3"
EMOTIONS = ["angry", "happy", "fearful", "disgusted", "sad", "surprised", "neutral"]
MODEL_A2 = "pretrained_For_Selection/台語A2模型"
MODEL_A3 = "pretrained_For_Selection/台語A3模型"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Kaldi text file (utt_id text...).")
    parser.add_argument("--output-root", default="./outputs/wav")
    parser.add_argument("--server-url", default="https://140.113.30.139:5003/")
    parser.add_argument("--model-a2", default=MODEL_A2)
    parser.add_argument("--model-a3", default=MODEL_A3)
    parser.add_argument("--model-name-a2", default="a2", help="Short name for A2 used in filenames and manifest.")
    parser.add_argument("--model-name-a3", default="a3", help="Short name for A3 used in filenames and manifest.")
    parser.add_argument("--mode-checkbox-group", default="自然語言控制")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--ssl-verify", action="store_true")
    parser.add_argument("--prompt-audio", default=PROMPT_AUDIO)
    parser.add_argument("--emotion-seed", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="Path to manifest JSON. Defaults to <output-root>/manifest.json.")
    return parser.parse_args()


def load_tasks(path: Path) -> List[Dict[str, str]]:
    tasks = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Line {idx} has no text after utt_id: {line!r}")
        tasks.append({"id": parts[0], "text": parts[1]})
    if not tasks:
        raise ValueError(f"No tasks loaded from {path}")
    return tasks


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


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


def assign_emotions(tasks: List[Dict[str, str]], rng: random.Random) -> None:
    n = len(tasks)
    pool = (EMOTIONS * math.ceil(n / len(EMOTIONS)))[:n]
    rng.shuffle(pool)
    for task, emotion in zip(tasks, pool):
        task["emotion"] = emotion


def run_model(
    model_name: str,
    short_name: str,
    tasks: List[Dict[str, str]],
    output_dir: Path,
    manifest_path: Path,
    client: Any,
    args: argparse.Namespace,
    prompt_wav_handle: Any,
    emotion_rng: random.Random,
) -> tuple[float, float, int]:
    """Run one model over all tasks. Returns (wall_sec, audio_sec, fail_count)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assign_emotions(tasks, emotion_rng)
    client.predict(model_name, api_name="/change_model")
    print(f"\n[MODEL] {model_name}  ({len(tasks)} utterances)")

    total_audio_sec = 0.0
    fail_count = 0
    records = []
    wall_start = time.perf_counter()

    for i, task in enumerate(tasks, start=1):
        out_path = output_dir / f"{i:06d}_{short_name}_{task['emotion']}.wav"

        if out_path.exists():
            dur = get_wav_duration(out_path)
            total_audio_sec += dur
            print(f"  [{i}/{len(tasks)}] {task['id']}  [SKIP existing]  {dur:.2f}s  cum={total_audio_sec:.1f}s")
            records.append({"index": i, "model": short_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": round(dur, 3)})
            continue

        generated: Optional[str] = None
        for attempt in range(1, args.retry_count + 2):
            try:
                generated = client.predict(
                    tts_text=task["text"],
                    mode_checkbox_group=args.mode_checkbox_group,
                    prompt_text=task["emotion"],
                    prompt_wav_upload=prompt_wav_handle,
                    prompt_wav_record=None,
                    seed=args.seed,
                    speed=args.speed,
                    enable_translation=False,
                    api_name="/generate",
                )
                break
            except Exception as exc:
                print(f"[WARN] {task['id']} attempt {attempt}: {exc}")

        if generated is None:
            fail_count += 1
            print(f"[SKIP] {task['id']}")
            records.append({"index": i, "model": short_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": None})
            continue

        shutil.copy(generated, out_path)
        dur = get_wav_duration(out_path)
        total_audio_sec += dur
        print(f"  [{i}/{len(tasks)}] {task['id']}  {task['emotion']}  {dur:.2f}s  cum={total_audio_sec:.1f}s")
        records.append({"index": i, "model": short_name, "text": task["text"], "emotion": task["emotion"], "file": out_path.name, "duration_sec": round(dur, 3)})

    wall_elapsed = time.perf_counter() - wall_start

    save_manifest(manifest_path, records, short_name)
    print(f"[INFO] manifest saved to {manifest_path}")

    return wall_elapsed, total_audio_sec, fail_count


def fmt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d} ({sec:.1f}s)"


def main() -> None:
    args = parse_args()
    from gradio_client import Client, handle_file

    prompt_audio_path = Path(args.prompt_audio).expanduser().resolve()
    if not prompt_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {prompt_audio_path}")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    tasks = load_tasks(input_path)
    print(f"[INFO] {len(tasks)} tasks from {input_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_root / "manifest.json"
    client = Client(args.server_url, ssl_verify=args.ssl_verify)
    prompt_wav_handle = handle_file(str(prompt_audio_path))
    emotion_rng = random.Random(args.emotion_seed)

    a2_wall, a2_audio, a2_fail = run_model(
        args.model_a2, args.model_name_a2, tasks, output_root, manifest_path, client, args, prompt_wav_handle, emotion_rng
    )
    a3_wall, a3_audio, a3_fail = run_model(
        args.model_a3, args.model_name_a3, tasks, output_root, manifest_path, client, args, prompt_wav_handle, emotion_rng
    )

    print("\n" + "=" * 60)
    print(f"  A2  wall={fmt(a2_wall)}  audio={fmt(a2_audio)}  failed={a2_fail}")
    print(f"  A3  wall={fmt(a3_wall)}  audio={fmt(a3_audio)}  failed={a3_fail}")
    print(f"  Total wall : {fmt(a2_wall + a3_wall)}")
    print(f"  Total audio: {fmt(a2_audio + a3_audio)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
