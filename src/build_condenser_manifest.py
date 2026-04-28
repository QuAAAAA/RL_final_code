#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


LOGGER = logging.getLogger(__name__)
VALID_ENDINGS = {".", "!", "?"}
REPLACE_WITH_PERIOD = {",", "、", ";", ":", " "}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="為 condenser 音檔建立 Kaldi-style manifest 檔案。"
    )
    parser.add_argument("--src_dir", type=Path, required=True, help="來源資料夾")
    parser.add_argument("--des_dir", type=Path, required=True, help="輸出資料夾")
    parser.add_argument(
        "--instruct",
        type=str,
        default="",
        help="若提供，會額外輸出 instruct 檔案並為每個 utterance 寫入相同內容",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        default="",
        help="為了相容舊指令保留，目前不使用",
    )
    parser.add_argument(
        "--json_field",
        type=str,
        default="台羅數字調",
        help="當來源為原始 TAT JSON 時，要讀取的文字欄位",
    )
    parser.add_argument(
        "--audio_dir_name",
        type=str,
        default="condenser",
        help="要讀取的音檔子資料夾名稱，例如 condenser 或 condenser_enhanced",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def find_wavs(src_dir: Path, audio_dir_name: str) -> tuple[list[Path], str]:
    audio_root = src_dir / audio_dir_name
    original_root = audio_root / "wav"
    if original_root.exists():
        return sorted(original_root.rglob("*.wav")), "tat_original"

    normalized_root = audio_root
    if normalized_root.exists():
        return sorted(normalized_root.rglob("*.wav")), "tatmoe_normalized"

    raise FileNotFoundError(f"找不到音檔資料夾: {audio_root}")


def read_text(text_path: Path) -> str:
    with text_path.open("r", encoding="utf-8") as file:
        return "".join(line.strip() for line in file)


def read_json_text(json_path: Path, json_field: str) -> str:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if json_field not in payload:
        raise KeyError(f"{json_path} 缺少欄位: {json_field}")

    return str(payload[json_field]).strip()


def normalize_sentence_ending(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    end = text[-1]
    if end in VALID_ENDINGS:
        return text
    if end in REPLACE_WITH_PERIOD:
        return text[:-1] + "."
    return text + "."


def split_original_tat_stem(stem: str) -> tuple[str, str]:
    parts = stem.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def resolve_metadata(
    wav_path: Path,
    src_dir: Path,
    layout: str,
    json_field: str,
) -> tuple[str, str, str]:
    spk = wav_path.parent.name
    raw_utt = wav_path.stem

    if layout == "tatmoe_normalized":
        text_path = wav_path.with_suffix(".normalized.txt")
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text file: {text_path}")
        return raw_utt, spk, normalize_sentence_ending(read_text(text_path))

    base_utt, _channel = split_original_tat_stem(raw_utt)
    json_path = src_dir / "json" / spk / f"{base_utt}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing json file: {json_path}")
    utt = f"condenser-{spk}_{raw_utt}"
    return utt, spk, normalize_sentence_ending(read_json_text(json_path, json_field))


def write_mapping(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for key, value in rows:
            file.write(f"{key} {value}\n")


def iter_wavs(wavs: list[Path]):
    if tqdm is None:
        return wavs
    return tqdm(wavs, desc="Scanning condenser wavs")


def main() -> None:
    args = parse_args()
    configure_logging()

    src_dir = args.src_dir.expanduser().resolve()
    des_dir = args.des_dir.expanduser().resolve()
    des_dir.mkdir(parents=True, exist_ok=True)

    wavs, layout = find_wavs(src_dir, args.audio_dir_name)
    if not wavs:
        raise SystemExit(f"在 {src_dir / args.audio_dir_name} 下找不到 wav 音檔。")

    utt2wav: dict[str, str] = {}
    utt2text: dict[str, str] = {}
    utt2spk: dict[str, str] = {}
    spk2utt: dict[str, list[str]] = {}
    missing_texts = 0

    for wav_path in iter_wavs(wavs):
        try:
            utt, spk, text = resolve_metadata(
                wav_path=wav_path,
                src_dir=src_dir,
                layout=layout,
                json_field=args.json_field,
            )
        except (FileNotFoundError, KeyError) as exc:
            LOGGER.warning("%s", exc)
            missing_texts += 1
            continue

        utt2wav[utt] = str(wav_path)
        utt2text[utt] = text
        utt2spk[utt] = spk
        spk2utt.setdefault(spk, []).append(utt)

    if not utt2wav:
        raise SystemExit("沒有可輸出的資料，請確認 wav 與文字檔是否成對存在。")

    ordered_utts = sorted(utt2wav)
    ordered_spks = sorted(spk2utt)

    write_mapping(
        des_dir / "wav.scp",
        [(utt, utt2wav[utt]) for utt in ordered_utts],
    )
    write_mapping(
        des_dir / "text",
        [(utt, utt2text[utt]) for utt in ordered_utts],
    )
    write_mapping(
        des_dir / "utt2spk",
        [(utt, utt2spk[utt]) for utt in ordered_utts],
    )
    write_mapping(
        des_dir / "spk2utt",
        [(spk, " ".join(sorted(spk2utt[spk]))) for spk in ordered_spks],
    )

    if args.instruct:
        write_mapping(
            des_dir / "instruct",
            [(utt, args.instruct) for utt in ordered_utts],
        )

    LOGGER.info(
        "Wrote %d utterances from %d speakers into %s (layout=%s)",
        len(ordered_utts),
        len(ordered_spks),
        des_dir,
        layout,
    )
    if missing_texts:
        LOGGER.info("Skipped %d wav files because text metadata was missing", missing_texts)


if __name__ == "__main__":
    main()
