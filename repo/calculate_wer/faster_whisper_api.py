import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "faster-whisper-taigi-pinyin-large-v7"
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parent / "MT0A2E.mp3"
DEFAULT_WAV_DIR = PROJECT_ROOT / "repo" / "outputs" / "wav"
DEFAULT_WAV_BY_MODEL_DIR = PROJECT_ROOT / "repo" / "outputs" / "wav_by_model"
DEFAULT_LABEL = "tiong1-san1 tsing3-kong1 loo7-khau2."


def normalize_text(text):
    text = re.sub(r"[-.,!?]", " ", text)
    return " ".join(text.split())


def calculate_wer(label, transcription):
    label_clean = normalize_text(label)
    transcription_clean = normalize_text(transcription) or " "
    measures = jiwer.process_words(label_clean.lower(), transcription_clean.lower())

    return {
        "label_clean": label_clean,
        "transcription_clean": transcription_clean,
        "wer": measures.wer,
        "reference_words": measures.hits + measures.substitutions + measures.deletions,
        "hits": measures.hits,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
    }


def load_whisper_model(model_path=DEFAULT_MODEL_PATH, device="cuda", compute_type=None):
    model_path = Path(model_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型目錄：{model_path}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    from faster_whisper import WhisperModel

    print(f"正在載入模型：{model_path}")
    return WhisperModel(str(model_path), device=device, compute_type=compute_type), model_path


def transcribe_audio(
    audio_path,
    model_path=DEFAULT_MODEL_PATH,
    device="cuda",
    compute_type=None,
    beam_size=5,
    language="zh",
    whisper_model=None,
    verbose=True,
):
    audio_path = Path(audio_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")

    if whisper_model is None:
        whisper_model, model_path = load_whisper_model(model_path, device, compute_type)

    if verbose:
        print(f"正在推論音檔：{audio_path}")

    segments, info = whisper_model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
    )
    transcription = " ".join(segment.text.strip() for segment in segments).strip()

    return {
        "filename": audio_path.name,
        "audio_path": str(audio_path),
        "model_path": str(model_path),
        "transcription": transcription,
        "language": info.language,
        "language_probability": info.language_probability,
    }


def load_manifest(manifest_path, limit=None):
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{manifest_path} 應該是 JSON list")
    if limit is not None:
        return data[:limit]
    return data


def resolve_manifest_audio(item, wav_dir, wav_by_model_dir):
    filename = item.get("file")
    model_name = item.get("model")
    if not filename:
        return None

    filename_path = Path(filename).expanduser()
    candidates = []
    if filename_path.is_absolute():
        candidates.append(filename_path)

    candidates.extend(
        [
            Path(wav_dir).expanduser() / filename,
            Path(wav_by_model_dir).expanduser() / str(model_name) / filename,
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def summarize_by_model(rows):
    stats = defaultdict(
        lambda: {
            "evaluated": 0,
            "missing_audio": 0,
            "errors": 0,
            "wer_sum": 0.0,
            "hits": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 0,
        }
    )

    for row in rows:
        model_name = row.get("model", "")
        item = stats[model_name]
        status = row.get("status")
        if status == "missing_audio":
            item["missing_audio"] += 1
            continue
        if status and status.startswith("error"):
            item["errors"] += 1
            continue
        if status != "ok":
            continue

        item["evaluated"] += 1
        item["wer_sum"] += float(row["wer"])
        item["hits"] += int(row["hits"])
        item["substitutions"] += int(row["substitutions"])
        item["deletions"] += int(row["deletions"])
        item["insertions"] += int(row["insertions"])
        item["reference_words"] += int(row["reference_words"])

    summary = {}
    for model_name, item in sorted(stats.items()):
        evaluated = item["evaluated"]
        reference_words = item["reference_words"]
        total_edits = item["substitutions"] + item["deletions"] + item["insertions"]
        summary[model_name] = {
            **item,
            "average_wer": item["wer_sum"] / evaluated if evaluated else 0.0,
            "corpus_wer": total_edits / reference_words if reference_words else 0.0,
        }
    return summary


CSV_FIELDS = [
    "index",
    "model",
    "file",
    "audio_path",
    "label",
    "transcription",
    "label_clean",
    "transcription_clean",
    "wer",
    "reference_words",
    "hits",
    "substitutions",
    "deletions",
    "insertions",
    "language",
    "language_probability",
    "status",
]


def resolve_output_format(output_path, output_format):
    if output_format != "auto":
        return output_format

    suffix = Path(output_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "txt"
    raise ValueError("無法從副檔名判斷輸出格式，請使用 .csv/.txt 或指定 --output-format")


def ensure_output_parent(output_path):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_csv_rows(rows, output_path):
    output_path = ensure_output_parent(output_path)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def single_result_to_row(result):
    row = {
        "index": "",
        "model": Path(result.get("model_path", "")).name,
        "file": result.get("filename", ""),
        "audio_path": result.get("audio_path", ""),
        "label": result.get("label", ""),
        "transcription": result.get("transcription", ""),
        "status": "ok",
    }
    for field in CSV_FIELDS:
        if field in result:
            row[field] = result[field]
    return row


def format_manifest_report(result):
    lines = [
        "[每個模型平均 WER]",
        "| Model | Evaluated | Avg WER | Corpus WER | Missing | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, item in result["by_model"].items():
        lines.append(
            f"| {model_name} | {item['evaluated']} | {item['average_wer']:.4f} | "
            f"{item['corpus_wer']:.4f} | {item['missing_audio']} | {item['errors']} |"
        )

    lines.extend(["", "[逐筆結果]"])
    for row in result["rows"]:
        lines.append(f"-- {row.get('model', '')} {row.get('file', '')}")
        lines.append(f"   Status: {row.get('status', '')}")
        lines.append(f"   Audio: {row.get('audio_path', '')}")
        lines.append(f"   Label: {row.get('label', '')}")
        lines.append(f"   Transcription: {row.get('transcription', '')}")
        if row.get("status") == "ok":
            lines.append(f"   WER: {float(row.get('wer', 0.0)):.4f}")
            lines.append(
                "   "
                f"Hits: {row.get('hits', '')}, "
                f"Substitutions: {row.get('substitutions', '')}, "
                f"Deletions: {row.get('deletions', '')}, "
                f"Insertions: {row.get('insertions', '')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_single_report(result):
    lines = [
        "-" * 40,
        f"檔案名稱：{result['filename']}",
        f"音檔路徑：{result['audio_path']}",
        f"模型路徑：{result['model_path']}",
        f"偵測語言：{result['language']} (機率: {result['language_probability']:.4f})",
        "轉錄結果：",
        result["transcription"],
    ]

    if "wer" in result:
        lines.extend(
            [
                "",
                "[WER 評估結果]",
                f"清理後辨識結果：{result['transcription_clean']}",
                f"清理後真實標籤：{result['label_clean']}",
                f"WER          ：{result['wer']:.4f}",
                f"Substitutions：{result['substitutions']}",
                f"Deletions    ：{result['deletions']}",
                f"Insertions   ：{result['insertions']}",
            ]
        )

    lines.append("-" * 40)
    return "\n".join(lines) + "\n"


def write_output(result, output_path, output_format):
    output_format = resolve_output_format(output_path, output_format)
    output_path = ensure_output_parent(output_path)

    if "rows" in result:
        if output_format == "csv":
            write_csv_rows(result["rows"], output_path)
        else:
            output_path.write_text(format_manifest_report(result), encoding="utf-8")
    elif output_format == "csv":
        write_csv_rows([single_result_to_row(result)], output_path)
    else:
        output_path.write_text(format_single_report(result), encoding="utf-8")

    return output_path, output_format


def log_batch_progress(message, progress_bar):
    if progress_bar is not None and tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def iter_manifest_entries(entries, quiet):
    if quiet or tqdm is None:
        return enumerate(entries, start=1), None
    progress_bar = tqdm(entries, desc="Transcribing", unit="wav")
    return enumerate(progress_bar, start=1), progress_bar


def transcribe_manifest(args):
    entries = load_manifest(args.manifest, args.limit)
    whisper_model, model_path = load_whisper_model(args.model, args.device, args.compute_type)

    rows = []
    iter_entries, progress_bar = iter_manifest_entries(entries, args.quiet)
    for pos, item in iter_entries:
        model_name = str(item.get("model", ""))
        label = item.get("text", "")
        filename = item.get("file", "")
        audio_path = resolve_manifest_audio(item, args.wav_dir, args.wav_by_model_dir)
        row = {
            "index": item.get("index", ""),
            "model": model_name,
            "file": filename,
            "audio_path": str(audio_path) if audio_path else "",
            "label": label,
            "transcription": "",
            "status": "ok",
        }

        if audio_path is None or not audio_path.exists():
            row["status"] = "missing_audio"
            rows.append(row)
            if not args.quiet:
                log_batch_progress(f"[{pos}/{len(entries)}] MISS {model_name} {filename}", progress_bar)
            continue

        try:
            result = transcribe_audio(
                audio_path=audio_path,
                model_path=model_path,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                language=args.language,
                whisper_model=whisper_model,
                verbose=False,
            )
            metrics = calculate_wer(label, result["transcription"])
        except Exception as exc:
            row["status"] = f"error: {exc}"
            rows.append(row)
            if not args.quiet:
                log_batch_progress(f"[{pos}/{len(entries)}] ERR  {model_name} {filename} {exc}", progress_bar)
            continue

        row.update(
            {
                "transcription": result["transcription"],
                "language": result["language"],
                "language_probability": result["language_probability"],
                **metrics,
            }
        )
        rows.append(row)

        if not args.quiet:
            log_batch_progress(
                f"[{pos}/{len(entries)}] OK   {model_name} {filename} WER={metrics['wer']:.4f}",
                progress_bar,
            )

    summary = summarize_by_model(rows)
    return {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "model_path": str(model_path),
        "evaluated": sum(item["evaluated"] for item in summary.values()),
        "by_model": summary,
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="使用本地 faster-whisper 模型做單檔推論")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH, help="要辨識的音檔路徑")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="faster-whisper 模型目錄")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推論裝置")
    parser.add_argument("--compute-type", default=None, help="例如 cuda 用 float16，cpu 用 int8")
    parser.add_argument("--beam-size", type=int, default=5, help="beam search 大小")
    parser.add_argument("--language", default="zh", help="Whisper language 參數")
    parser.add_argument("--label", default=None, help="提供答案標籤時會一併計算 WER")
    parser.add_argument("--manifest", type=Path, default=None, help="批次評估 manifest JSON，需包含 model/file/text 欄位")
    parser.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR, help="manifest 批次模式的 wav 目錄")
    parser.add_argument("--wav-by-model-dir", type=Path, default=DEFAULT_WAV_BY_MODEL_DIR, help="manifest 批次模式的分模型 wav 目錄")
    parser.add_argument("--limit", type=int, default=None, help="批次模式只跑前 N 筆")
    parser.add_argument("--quiet", action="store_true", help="批次模式不逐筆列印")
    parser.add_argument("--json", action="store_true", help="用 JSON 格式輸出結果")
    parser.add_argument("--output", type=Path, default=None, help="將結果輸出到 .csv 或 .txt 檔案")
    parser.add_argument(
        "--output-format",
        default="auto",
        choices=["auto", "csv", "txt"],
        help="輸出格式；auto 會依 --output 副檔名判斷",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.manifest is not None:
        result = transcribe_manifest(args)
        if args.output is not None:
            output_path, output_format = write_output(result, args.output, args.output_format)
            print(f"結果已輸出成 {output_format.upper()}：{output_path}", file=sys.stderr)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        print("\n[每個模型平均 WER]")
        print("| Model | Evaluated | Avg WER | Corpus WER | Missing | Errors |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for model_name, item in result["by_model"].items():
            print(
                f"| {model_name} | {item['evaluated']} | {item['average_wer']:.4f} | "
                f"{item['corpus_wer']:.4f} | {item['missing_audio']} | {item['errors']} |"
            )
        return

    result = transcribe_audio(
        audio_path=args.audio,
        model_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
    )

    if args.label is not None:
        result["label"] = args.label
        result.update(calculate_wer(args.label, result["transcription"]))

    if args.output is not None:
        output_path, output_format = write_output(result, args.output, args.output_format)
        print(f"結果已輸出成 {output_format.upper()}：{output_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("-" * 40)
    print(f"檔案名稱：{result['filename']}")
    print(f"偵測語言：{result['language']} (機率: {result['language_probability']:.4f})")
    print("轉錄結果：")
    print(result["transcription"])

    if args.label is not None:
        print("\n[WER 評估結果]")
        print(f"清理後辨識結果：{result['transcription_clean']}")
        print(f"清理後真實標籤：{result['label_clean']}")
        print(f"WER          ：{result['wer']:.4f}")
        print(f"Substitutions：{result['substitutions']}")
        print(f"Deletions    ：{result['deletions']}")
        print(f"Insertions   ：{result['insertions']}")
    print("-" * 40)


if __name__ == "__main__":
    main()
