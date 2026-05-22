from __future__ import annotations

import argparse
import sys
from pathlib import Path

from repo.train.service_http import JsonApiServer


REPO_ROOT = Path(__file__).resolve().parents[2]
ASR_SRC = REPO_ROOT / "ASR_sys" / "src"
DEFAULT_ASR_MODEL = REPO_ROOT / "ASR_sys" / "ASR" / "whisper-large-v3-turbo-finetuned-pinyin"
DEFAULT_PROCESSOR = DEFAULT_ASR_MODEL

if str(ASR_SRC) not in sys.path:
    sys.path.insert(0, str(ASR_SRC))

from eval import transcribe_batch  # noqa: E402
from strip_tones import strip_tl_tones  # noqa: E402


app = JsonApiServer()
MODEL = None
PROCESSOR = None
MODEL_PATH = str(DEFAULT_ASR_MODEL)
PROCESSOR_PATH = str(DEFAULT_PROCESSOR)
DEVICE = "cpu"
BATCH_SIZE = 1
LANGUAGE = "zh"
TRIM_TOP_DB = 30.0
REMOVE_INTERNAL_SILENCE = True


def _levenshtein(a: str, b: str) -> int:
    try:
        from Levenshtein import distance as levenshtein_distance

        return int(levenshtein_distance(a, b))
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            for j, cb in enumerate(b, start=1):
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + (0 if ca == cb else 1),
                    )
                )
            prev = curr
        return prev[-1]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import wer as compute_wer

        return float(compute_wer(reference, hypothesis))
    except ImportError:
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        prev = list(range(len(hyp_words) + 1))
        for i, ref in enumerate(ref_words, start=1):
            curr = [i]
            for j, hyp in enumerate(hyp_words, start=1):
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + (0 if ref == hyp else 1),
                    )
                )
            prev = curr
        return float(prev[-1] / len(ref_words))


def _normalize_label(text: str, strip_tones: bool = True) -> str:
    text = str(text or "").strip()
    if strip_tones:
        text = strip_tl_tones(text)
    return " ".join(text.split())


def _transcribe_one(audio_path: str) -> str:
    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("ASR model is not loaded")
    trim_top_db = TRIM_TOP_DB if TRIM_TOP_DB and TRIM_TOP_DB > 0 else None
    predictions = transcribe_batch(
        [audio_path],
        MODEL,
        PROCESSOR,
        DEVICE,
        BATCH_SIZE,
        LANGUAGE,
        trim_top_db=trim_top_db,
        remove_internal_silence=REMOVE_INTERNAL_SILENCE,
    )
    return str(predictions[0]).strip() if predictions else ""


@app.route("/transcribe")
def transcribe(payload: dict) -> dict:
    audio_path = str(Path(payload["audio_path"]).expanduser().resolve())
    transcription = _transcribe_one(audio_path)
    return {
        "ok": True,
        "filename": Path(audio_path).name,
        "audio_path": audio_path,
        "transcription": transcription,
        "model_path": MODEL_PATH,
        "processor_path": PROCESSOR_PATH,
        "language": LANGUAGE,
    }


@app.route("/wer")
def wer(payload: dict) -> dict:
    result = transcribe(payload)
    strip_tones = bool(payload.get("strip_tones", True))
    label_clean = _normalize_label(payload.get("label", ""), strip_tones=strip_tones)
    transcription_clean = _normalize_label(result["transcription"], strip_tones=strip_tones)
    wer_value = _word_error_rate(label_clean, transcription_clean)
    lev_distance = _levenshtein(transcription_clean, label_clean)
    return {
        "ok": True,
        **result,
        "label_clean": label_clean,
        "transcription_clean": transcription_clean,
        "wer": wer_value,
        "lev_distance": lev_distance,
        "reference_chars": len(label_clean),
        "prediction_chars": len(transcription_clean),
    }


def main() -> None:
    global MODEL, PROCESSOR, MODEL_PATH, PROCESSOR_PATH, DEVICE, BATCH_SIZE, LANGUAGE, TRIM_TOP_DB, REMOVE_INTERNAL_SILENCE
    parser = argparse.ArgumentParser(description="ASR_sys Whisper HTTP service for reward scoring")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=str(DEFAULT_ASR_MODEL))
    parser.add_argument("--processor", default=str(DEFAULT_PROCESSOR))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--trim-top-db", type=float, default=30.0)
    parser.add_argument("--remove-internal-silence", action="store_true", default=True)
    parser.add_argument("--no-remove-internal-silence", dest="remove_internal_silence", action="store_false")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    DEVICE = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if DEVICE == "auto":
        DEVICE = "cpu"
    MODEL_PATH = str(Path(args.model).expanduser().resolve())
    PROCESSOR_PATH = str(Path(args.processor).expanduser().resolve()) if Path(args.processor).expanduser().exists() else args.processor
    BATCH_SIZE = args.batch_size
    LANGUAGE = args.language
    TRIM_TOP_DB = args.trim_top_db
    REMOVE_INTERNAL_SILENCE = args.remove_internal_silence

    print(f"[asr] device={DEVICE}")
    print(f"[asr] loading processor from {PROCESSOR_PATH}")
    PROCESSOR = AutoProcessor.from_pretrained(PROCESSOR_PATH)
    print(f"[asr] loading model from {MODEL_PATH}")
    MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    MODEL.eval()

    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()
