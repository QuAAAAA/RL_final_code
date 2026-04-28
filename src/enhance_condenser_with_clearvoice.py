#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from clearvoice import ClearVoice

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 ClearVoice 對目錄中的 wav 音檔進行降噪，並輸出到新的資料夾。"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="輸入音檔資料夾，例如 /path/to/condenser",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="輸出資料夾；若未提供，預設為 input_dir 同層的 <資料夾名>_enhanced",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FRCRN_SE_16K",
        help="ClearVoice 使用的 speech enhancement 模型名稱",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若輸出檔已存在則跳過，不重跑",
    )
    return parser.parse_args()


def get_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_enhanced"


def iter_paths(paths: list[Path], desc: str):
    if tqdm is None:
        return paths
    return tqdm(paths, desc=desc)


def collect_wavs(input_dir: Path, output_dir: Path) -> list[Path]:
    wavs = []
    for wav_path in sorted(input_dir.rglob("*.wav")):
        if output_dir in wav_path.parents:
            continue
        wavs.append(wav_path)
    return wavs


def build_output_path(wav_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = wav_path.relative_to(input_dir)
    return output_dir / relative_path


def enhance_single_file(
    clearvoice: ClearVoice,
    input_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_wav = clearvoice(input_path=str(input_path), online_write=False)
    clearvoice.write(output_wav, str(output_path))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = get_output_dir(input_dir, args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"找不到輸入資料夾: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir 不是資料夾: {input_dir}")

    wav_list = collect_wavs(input_dir, output_dir)
    if not wav_list:
        raise SystemExit(f"在 {input_dir} 下找不到任何 .wav 檔案。")

    clearvoice = ClearVoice(
        task="speech_enhancement",
        model_names=[args.model_name],
    )

    processed = 0
    skipped = 0
    for wav_path in iter_paths(wav_list, desc="Enhancing WAV files"):
        output_path = build_output_path(wav_path, input_dir, output_dir)
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        enhance_single_file(clearvoice, wav_path, output_path)
        processed += 1

    print(f"input_dir : {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"processed : {processed}")
    print(f"skipped   : {skipped}")


if __name__ == "__main__":
    main()
