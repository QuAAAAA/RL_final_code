#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import wave
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "TAT-Vol1"
SUPPORTED_EXTENSIONS = {".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="計算 TAT 資料集中 condenser 資料夾音檔的總長度。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"TAT 資料集根目錄，預設為 {DEFAULT_ROOT}",
    )
    return parser.parse_args()


def find_condenser_audio_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "condenser" in path.parts
    )


def get_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        if sample_rate == 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return frame_count / sample_rate


def infer_split_name(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"TAT-Vol\d+-(.+)", part)
        if match:
            return match.group(1)
    return "unknown"


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"找不到資料夾: {root}")

    audio_files = find_condenser_audio_files(root)
    if not audio_files:
        raise SystemExit(f"在 {root} 下找不到 condenser 音檔。")

    split_totals: dict[str, float] = defaultdict(float)
    split_counts: dict[str, int] = defaultdict(int)
    total_seconds = 0.0

    for audio_file in audio_files:
        duration = get_wav_duration_seconds(audio_file)
        split = infer_split_name(audio_file)
        split_totals[split] += duration
        split_counts[split] += 1
        total_seconds += duration

    for split in sorted(split_totals):
        seconds = split_totals[split]
        print(
            f"{split:>5} | files: {split_counts[split]:>5} | "
            f"seconds: {seconds:>10.2f} | hours: {seconds / 3600:>7.2f} | "
            f"hms: {format_duration(seconds)}"
        )

    print("-" * 72)
    print(
        f"total | files: {len(audio_files):>5} | "
        f"seconds: {total_seconds:>10.2f} | hours: {total_seconds / 3600:>7.2f} | "
        f"hms: {format_duration(total_seconds)}"
    )


if __name__ == "__main__":
    main()
