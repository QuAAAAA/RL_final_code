import argparse
import json
import re
from pathlib import Path

import jiwer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "faster-whisper-taigi-pinyin-large-v7"
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parent / "MT0A2E.mp3"
DEFAULT_LABEL = "tiong1-san1 tsing3-kong1 loo7-khau2."


def normalize_text(text):
    text = re.sub(r"[-.,!?]", " ", text)
    return " ".join(text.split())


def calculate_mer(label, transcription):
    label_clean = normalize_text(label)
    transcription_clean = normalize_text(transcription) or " "
    measures = jiwer.process_words(label_clean.lower(), transcription_clean.lower())

    return {
        "label_clean": label_clean,
        "transcription_clean": transcription_clean,
        "mer": measures.wer,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
    }


def transcribe_audio(
    audio_path,
    model_path=DEFAULT_MODEL_PATH,
    device="cuda",
    compute_type=None,
    beam_size=5,
    language="zh",
):
    audio_path = Path(audio_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型目錄：{model_path}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    from faster_whisper import WhisperModel

    print(f"正在載入模型：{model_path}")
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    print(f"正在推論音檔：{audio_path}")

    segments, info = model.transcribe(
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


def parse_args():
    parser = argparse.ArgumentParser(description="使用本地 faster-whisper 模型做單檔推論")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH, help="要辨識的音檔路徑")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="faster-whisper 模型目錄")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推論裝置")
    parser.add_argument("--compute-type", default=None, help="例如 cuda 用 float16，cpu 用 int8")
    parser.add_argument("--beam-size", type=int, default=5, help="beam search 大小")
    parser.add_argument("--language", default="zh", help="Whisper language 參數")
    parser.add_argument("--label", default=None, help="提供答案標籤時會一併計算 MER/WER")
    parser.add_argument("--json", action="store_true", help="用 JSON 格式輸出結果")
    return parser.parse_args()


def main():
    args = parse_args()
    result = transcribe_audio(
        audio_path=args.audio,
        model_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
    )

    if args.label is not None:
        result.update(calculate_mer(args.label, result["transcription"]))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("-" * 40)
    print(f"檔案名稱：{result['filename']}")
    print(f"偵測語言：{result['language']} (機率: {result['language_probability']:.4f})")
    print("轉錄結果：")
    print(result["transcription"])

    if args.label is not None:
        print("\n[MER 評估結果]")
        print(f"清理後辨識結果：{result['transcription_clean']}")
        print(f"清理後真實標籤：{result['label_clean']}")
        print(f"MER (WER)    ：{result['mer']:.4f}")
        print(f"Substitutions：{result['substitutions']}")
        print(f"Deletions    ：{result['deletions']}")
        print(f"Insertions   ：{result['insertions']}")
    print("-" * 40)


if __name__ == "__main__":
    main()
