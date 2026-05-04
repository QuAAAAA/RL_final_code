import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = SCRIPT_DIR / "source.mp3"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "manifest_all.json"
DEFAULT_AUDIO_DIR = REPO_ROOT / "outputs" / "wav"
DEFAULT_SAVEDIR = WORKSPACE_ROOT / "models" / "speechbrain-spkrec-ecapa-voxceleb"
DEFAULT_MODEL = DEFAULT_SAVEDIR
EMOTION_COLUMNS = [
    ("neutral", "Neutral"),
    ("angry", "Angry"),
    ("happy", "Happy"),
    ("sad", "Sad"),
    ("surprised", "Surprise"),
    ("fearful", "Fear"),
    ("disgusted", "Disgust"),
]


def load_audio(audio_path, sample_rate=16000):
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")

    waveform, original_sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0, keepdim=True)

    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=original_sample_rate,
            new_freq=sample_rate,
        )

    return waveform


def load_classifier(model_source=DEFAULT_MODEL, savedir=DEFAULT_SAVEDIR, device="cpu"):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=str(model_source),
        savedir=str(savedir),
        run_opts={"device": device},
    )


def encode_audio(classifier, audio_path, device="cpu"):
    wav = load_audio(audio_path).to(device)

    with torch.no_grad():
        return classifier.encode_batch(wav).squeeze()


def speaker_similarity(source_audio, target_audio, model_source=DEFAULT_MODEL, savedir=DEFAULT_SAVEDIR, device="cpu"):
    classifier = load_classifier(model_source=model_source, savedir=savedir, device=device)
    emb1 = encode_audio(classifier, source_audio, device=device)
    emb2 = encode_audio(classifier, target_audio, device=device)

    similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
    return {
        "source_audio": str(Path(source_audio).expanduser().resolve()),
        "target_audio": str(Path(target_audio).expanduser().resolve()),
        "model": str(model_source),
        "similarity": similarity,
    }


def resolve_audio_path(file_name, audio_dir):
    audio_path = Path(file_name).expanduser()
    if not audio_path.is_absolute():
        audio_path = Path(audio_dir).expanduser() / audio_path
    return audio_path.resolve()


def summarize_by_model(rows, missing_counts, error_counts):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row["similarity"])

    summary = {}
    for model_name in sorted(set(grouped) | set(missing_counts) | set(error_counts)):
        scores = grouped.get(model_name, [])
        summary[model_name] = {
            "evaluated": len(scores),
            "missing": missing_counts.get(model_name, 0),
            "errors": error_counts.get(model_name, 0),
            "mean_similarity": statistics.mean(scores) if scores else None,
            "std_similarity": statistics.pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            "min_similarity": min(scores) if scores else None,
            "max_similarity": max(scores) if scores else None,
        }
    return summary


def summarize_by_emotion(rows, missing_counts, error_counts):
    grouped = {}
    for row in rows:
        key = (row["model"], row.get("emotion") or "unknown")
        grouped.setdefault(key, []).append(row["similarity"])

    summary = []
    keys = sorted(set(grouped) | set(missing_counts) | set(error_counts))
    for model_name, emotion in keys:
        scores = grouped.get((model_name, emotion), [])
        summary.append(
            {
                "model": model_name,
                "emotion": emotion,
                "evaluated": len(scores),
                "mean_similarity": statistics.mean(scores) if scores else None,
                "missing": missing_counts.get((model_name, emotion), 0),
                "errors": error_counts.get((model_name, emotion), 0),
            }
        )
    return summary


def format_score(score):
    return f"{score:.4f}" if score is not None else "n/a"


def render_summary_table(result):
    lines = [
        "[每個模型平均 Speaker Similarity]",
        "| Model | Evaluated | Avg SS | Missing | Errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, summary in result["summary_by_model"].items():
        lines.append(
            f"| {model_name} "
            f"| {summary['evaluated']} "
            f"| {format_score(summary['mean_similarity'])} "
            f"| {summary['missing']} "
            f"| {summary['errors']} |"
        )
    return "\n".join(lines)


def render_emotion_table(result):
    emotion_scores = {}
    for row in result["summary_by_emotion"]:
        emotion_scores[(row["model"], row["emotion"])] = row["mean_similarity"]

    headers = ["Model", "Mean"] + [label for _, label in EMOTION_COLUMNS]
    lines = [
        "[依情緒分類 Speaker Similarity]",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name, summary in result["summary_by_model"].items():
        values = [model_name, format_score(summary["mean_similarity"])]
        values.extend(format_score(emotion_scores.get((model_name, emotion))) for emotion, _ in EMOTION_COLUMNS)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_text(path, text):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_details(path, details):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        fieldnames = ["index", "model", "emotion", "file", "similarity"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details)
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


def speaker_similarity_manifest(
    source_audio,
    manifest_path=DEFAULT_MANIFEST,
    audio_dir=DEFAULT_AUDIO_DIR,
    model_source=DEFAULT_MODEL,
    savedir=DEFAULT_SAVEDIR,
    device="cpu",
    limit=None,
    progress=True,
):
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest：{manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    classifier = load_classifier(model_source=model_source, savedir=savedir, device=device)
    source_audio = Path(source_audio).expanduser().resolve()
    source_emb = encode_audio(classifier, source_audio, device=device)

    rows = []
    missing = []
    errors = []
    missing_counts = {}
    error_counts = {}
    missing_emotion_counts = {}
    error_emotion_counts = {}

    items = manifest[:limit]
    iterator = tqdm(items, desc="Speaker similarity", unit="wav", disable=not progress)

    for item in iterator:
        model_name = item.get("model", "unknown")
        emotion = item.get("emotion") or "unknown"
        target_audio = resolve_audio_path(item["file"], audio_dir)

        if not target_audio.exists():
            missing.append(
                {
                    "index": item.get("index"),
                    "model": model_name,
                    "emotion": emotion,
                    "file": str(target_audio),
                }
            )
            missing_counts[model_name] = missing_counts.get(model_name, 0) + 1
            missing_emotion_counts[(model_name, emotion)] = missing_emotion_counts.get((model_name, emotion), 0) + 1
            continue

        try:
            target_emb = encode_audio(classifier, target_audio, device=device)
            similarity = F.cosine_similarity(source_emb, target_emb, dim=0).item()
        except Exception as exc:
            errors.append(
                {
                    "index": item.get("index"),
                    "model": model_name,
                    "emotion": emotion,
                    "file": str(target_audio),
                    "error": str(exc),
                }
            )
            error_counts[model_name] = error_counts.get(model_name, 0) + 1
            error_emotion_counts[(model_name, emotion)] = error_emotion_counts.get((model_name, emotion), 0) + 1
            continue

        rows.append(
            {
                "index": item.get("index"),
                "model": model_name,
                "emotion": emotion,
                "file": str(target_audio),
                "similarity": similarity,
            }
        )

    return {
        "source_audio": str(source_audio),
        "manifest": str(manifest_path),
        "audio_dir": str(Path(audio_dir).expanduser().resolve()),
        "speaker_model": str(model_source),
        "summary_by_model": summarize_by_model(rows, missing_counts, error_counts),
        "summary_by_emotion": summarize_by_emotion(rows, missing_emotion_counts, error_emotion_counts),
        "details": rows,
        "missing": missing,
        "errors": errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="計算語者相似度")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="來源語者音檔")
    parser.add_argument("--target", type=Path, default=None, help="要比較的單一目標音檔；不指定時會讀 manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="批次評估 manifest")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR, help="manifest file 欄位所在的音檔資料夾")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SpeechBrain speaker model 或本地模型路徑")
    parser.add_argument("--savedir", type=Path, default=DEFAULT_SAVEDIR, help="SpeechBrain 模型快取目錄")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推論裝置")
    parser.add_argument("--json", action="store_true", help="用 JSON 格式輸出結果")
    parser.add_argument("--details", action="store_true", help="JSON 輸出時包含每筆音檔分數")
    parser.add_argument("--limit", type=int, default=None, help="只評估前 N 筆，方便快速測試")
    parser.add_argument("--no-progress", action="store_true", help="不要顯示 tqdm 進度列")
    parser.add_argument("--summary-output", type=Path, default=None, help="把 summary Markdown 表格寫到指定檔案")
    parser.add_argument("--details-output", type=Path, default=None, help="把每筆分數寫到指定檔案；副檔名 .csv 會輸出 CSV，其他輸出 JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target:
        result = speaker_similarity(
            source_audio=args.source,
            target_audio=args.target,
            model_source=args.model,
            savedir=args.savedir,
            device=args.device,
        )

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        print("-" * 40)
        print(f"來源音檔：{result['source_audio']}")
        print(f"目標音檔：{result['target_audio']}")
        print(f"語者相似度：{result['similarity']:.4f}")
        print("-" * 40)
        return

    result = speaker_similarity_manifest(
        source_audio=args.source,
        manifest_path=args.manifest,
        audio_dir=args.audio_dir,
        model_source=args.model,
        savedir=args.savedir,
        device=args.device,
        limit=args.limit,
        progress=not args.no_progress,
    )

    if args.json:
        if not args.details:
            result = {k: v for k, v in result.items() if k != "details"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary_table = render_summary_table(result)
    emotion_table = render_emotion_table(result)
    if args.summary_output:
        write_text(args.summary_output, f"{summary_table}\n\n{emotion_table}")
    if args.details_output:
        write_details(args.details_output, result["details"])

    print("-" * 72)
    print(f"來源音檔：{result['source_audio']}")
    print(f"Manifest：{result['manifest']}")
    print("-" * 72)
    print(summary_table)
    print()
    print(emotion_table)
    print("-" * 72)


if __name__ == "__main__":
    main()
