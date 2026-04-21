#!/usr/bin/env python3
"""Batch TTS generation script for the Gradio endpoint used in apiCall_ta.py.

Input task file supports:
1) JSON array: [{...}, {...}]
2) JSONL: one JSON object per line

Each task must include:
- text   (transcription / text to synthesise)
- audio  (path to reference wav for voice cloning)

Optional fields: id, speaker, language

python3 repo/baseline/batch_generate_ta.py \
  --input-json data/train_manifest.jsonl \
  --target-seconds 3600 \
  --output-dir outputs/train \
  --manifest outputs/train/manifest.jsonl \
  --emotion-seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

EMOTIONS = ["angry", "happy", "fearful", "disgusted", "sad", "surprised", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch TTS generation by target duration.")
    parser.add_argument("--input-json", required=True, help="Path to JSON array or JSONL task file.")
    parser.add_argument("--target-seconds", type=float, required=True, help="Stop when cumulative duration reaches this value.")
    parser.add_argument("--server-url", default="https://140.113.30.139:5003/", help="Gradio server URL.")
    parser.add_argument("--model", default="pretrained_For_Selection/台語模型", help="Default model for /change_model.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for generated wav files.")
    parser.add_argument("--manifest", default="./outputs/manifest.jsonl", help="JSONL manifest output path.")
    parser.add_argument("--mode-checkbox-group", default="3s極速覆刻", help="mode_checkbox_group value for /generate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Default synthesis speed.")
    parser.add_argument("--seed-start", type=int, default=51234, help="Base seed when task seed is not provided.")
    parser.add_argument("--enable-translation", action="store_true", help="Enable translation in /generate.")
    parser.add_argument(
        "--text-template",
        default="{instruct}\n{tts_text}",
        help="Template to compose final tts_text. Supports {instruct} and {tts_text}.",
    )
    parser.add_argument("--retry-count", type=int, default=2, help="Retries per task on API failure.")
    parser.add_argument("--no-cycle", action="store_true", help="Do not cycle tasks when input is exhausted.")
    parser.add_argument("--ssl-verify", action="store_true", help="Enable SSL cert verification (default disabled).")
    parser.add_argument("--emotion-seed", type=int, default=None, help="RNG seed for balanced emotion shuffle.")
    return parser.parse_args()


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Task file is empty: {path}")

    tasks: List[Dict[str, Any]] = []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("JSON input must be an array of objects.")
        tasks = parsed
    else:
        for idx, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {idx}: {exc}") from exc
            tasks.append(obj)

    if not tasks:
        raise ValueError("No valid tasks loaded from input.")

    required = ("text", "audio")
    for i, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"Task #{i} must be a JSON object.")
        missing = [key for key in required if key not in task]
        if missing:
            raise ValueError(f"Task #{i} missing required fields: {missing}")

    return tasks


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return safe or "task"


def get_wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            if sample_rate <= 0:
                return None
            return frame_count / float(sample_rate)
    except Exception:
        return None


def switch_model(client: Any, model: str) -> None:
    client.predict(model, api_name="/change_model")


def compose_tts_text(template: str, instruct: str, tts_text: str) -> str:
    try:
        return template.format(instruct=instruct, tts_text=tts_text)
    except KeyError as exc:
        raise ValueError(f"text-template contains unknown placeholder: {exc}") from exc


def append_manifest(manifest_path: Path, record: Dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def assign_emotions(tasks: List[Dict[str, Any]], seed: Optional[int] = None) -> None:
    n = len(tasks)
    reps = math.ceil(n / len(EMOTIONS))
    pool = (EMOTIONS * reps)[:n]
    rng = random.Random(seed)
    rng.shuffle(pool)
    for task, emotion in zip(tasks, pool):
        task["emotion"] = emotion


def main() -> None:
    args = parse_args()
    from gradio_client import Client, handle_file

    if args.target_seconds <= 0:
        raise ValueError("--target-seconds must be > 0")
    if args.retry_count < 0:
        raise ValueError("--retry-count must be >= 0")

    input_path = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(input_path)
    assign_emotions(tasks, seed=args.emotion_seed)
    counts = Counter(t["emotion"] for t in tasks)
    print(f"[INFO] emotion distribution: {dict(counts)}")

    client = Client(args.server_url, ssl_verify=args.ssl_verify)

    current_model = args.model
    switch_model(client, current_model)
    print(f"[INFO] model set: {current_model}")

    cumulative_sec = 0.0
    global_index = 0
    task_cursor = 0
    success_count = 0
    fail_count = 0
    no_progress_cycles = 0

    while cumulative_sec < args.target_seconds:
        if task_cursor >= len(tasks):
            if args.no_cycle:
                print("[WARN] input exhausted before reaching target duration.")
                break
            task_cursor = 0
            if success_count == 0:
                no_progress_cycles += 1
            else:
                no_progress_cycles = 0
            success_count = 0
            if no_progress_cycles >= 3:
                raise RuntimeError("No successful generation for 3 full cycles. Stopping to avoid infinite loop.")

        task = tasks[task_cursor]
        task_cursor += 1
        global_index += 1

        task_id = str(task.get("id", f"task_{global_index}"))
        instruct = str(task["emotion"])
        prompt_text = str(task["text"])
        raw_tts_text = str(task["text"])
        final_tts_text = compose_tts_text(args.text_template, instruct, raw_tts_text)

        audio_path = Path(task["audio"]).expanduser().resolve()
        if not audio_path.exists():
            fail_count += 1
            record_stub: Dict[str, Any] = {
                "index": global_index,
                "id": task_id,
                "status": "failed",
                "error": f"audio file not found: {audio_path}",
                "emotion": instruct,
            }
            append_manifest(manifest_path, record_stub)
            print(f"[WARN] task {task_id} skipped: audio not found: {audio_path}")
            continue
        prompt_wav_handle = handle_file(str(audio_path))

        task_model = str(task.get("model", current_model))
        if task_model != current_model:
            switch_model(client, task_model)
            current_model = task_model
            print(f"[INFO] model switched: {current_model}")

        mode_checkbox_group = str(task.get("mode_checkbox_group", args.mode_checkbox_group))
        speed = float(task.get("speed", args.speed))
        seed = int(task.get("seed", args.seed_start + global_index - 1))

        output_name = f"{global_index:06d}_{sanitize_name(task_id)}.wav"
        output_path = output_dir / output_name

        record: Dict[str, Any] = {
            "index": global_index,
            "id": task_id,
            "status": "failed",
            "error": None,
            "audio_path": str(output_path),
            "duration_sec": 0.0,
            "cum_duration_sec": cumulative_sec,
            "model": current_model,
            "seed": seed,
            "speed": speed,
            "emotion": instruct,
            "prompt_text": prompt_text,
            "tts_text": raw_tts_text,
            "instruct": instruct,
            "final_tts_text": final_tts_text,
        }

        total_attempts = args.retry_count + 1
        last_error: Optional[str] = None
        generated_audio: Optional[str] = None
        for attempt in range(1, total_attempts + 1):
            try:
                generated_audio = client.predict(
                    tts_text=final_tts_text,
                    mode_checkbox_group=mode_checkbox_group,
                    prompt_text=prompt_text,
                    prompt_wav_upload=prompt_wav_handle,
                    prompt_wav_record=None,
                    seed=seed,
                    speed=speed,
                    enable_translation=args.enable_translation,
                    api_name="/generate",
                )
                break
            except Exception as exc:
                last_error = f"attempt {attempt}/{total_attempts}: {exc}"
                print(f"[WARN] task {task_id} failed: {last_error}")

        if generated_audio is None:
            fail_count += 1
            record["error"] = last_error or "unknown error"
            append_manifest(manifest_path, record)
            continue

        shutil.copy(generated_audio, output_path)
        duration = get_wav_duration_seconds(output_path)
        if duration is None:
            fail_count += 1
            record["error"] = "generated file is not readable as wav"
            append_manifest(manifest_path, record)
            continue

        cumulative_sec += duration
        success_count += 1
        record["status"] = "success"
        record["duration_sec"] = round(duration, 6)
        record["cum_duration_sec"] = round(cumulative_sec, 6)
        append_manifest(manifest_path, record)

        print(
            f"[INFO] generated {output_name} "
            f"({duration:.2f}s, cumulative {cumulative_sec:.2f}/{args.target_seconds:.2f}s)"
        )

    print(
        "[DONE] "
        f"cumulative={cumulative_sec:.2f}s, target={args.target_seconds:.2f}s, "
        f"manifest={manifest_path}, output_dir={output_dir}, failed={fail_count}"
    )


if __name__ == "__main__":
    main()
