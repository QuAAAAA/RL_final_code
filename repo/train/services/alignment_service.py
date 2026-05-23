"""Alignment HTTP service — Whisper word-level timestamps.

API:
    POST /align  {audio_path: str, text: str}
    → {ok: true, boundaries: [[word, start, end], ...], target_indices: [int, ...]}

`text` should be the tagged text with "" around emphasis words.
`target_indices` are the indices into `boundaries` that were inside quotes.

Uses faster-whisper with word_timestamps=True to get actual word-level timing,
then maps to ground-truth words by position. No ASR transcription is used —
only the timing information from Whisper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from repo.train.service_http import JsonApiServer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = str(REPO_ROOT / "models" / "faster-whisper-taigi-pinyin-large-v7")

app = JsonApiServer()

_MODEL = None
MODEL_PATH: str = DEFAULT_MODEL
DEVICE: str = "cpu"

QUOTE_CHARS = {'"', "“", "”"}
PUNCT = {".", ",", "?", "!", ":", ";", "。", "，"}


def _token_has_quote(token: str) -> bool:
    return any(c in token for c in QUOTE_CHARS)


def _strip_quotes(token: str) -> str:
    for c in QUOTE_CHARS:
        token = token.replace(c, "")
    return token


def _preprocess_tagged_text(text: str) -> tuple[list[str], list[int]]:
    """Strip quote markers; return (clean_words, target_word_indices)."""
    tokens = text.split()
    clean_words: list[str] = []
    target_indices: list[int] = []
    inside_quote = False

    for token in tokens:
        has_quote = _token_has_quote(token)
        spoken = _strip_quotes(token).strip()

        if spoken and spoken not in PUNCT:
            if inside_quote:
                target_indices.append(len(clean_words))
            clean_words.append(spoken)

        if has_quote:
            inside_quote = not inside_quote

    return clean_words, target_indices


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        compute = "int8" if DEVICE == "cpu" else "float16"
        _MODEL = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=compute)
    return _MODEL


def _map_to_ref(ref_words: list[str], whisper_times: list[tuple[float, float]]) -> list[list]:
    """Map ground-truth words to Whisper word timestamps by position.

    If counts match, use 1:1 mapping. Otherwise fall back to proportional
    distribution within Whisper's detected time span.
    """
    if not whisper_times:
        return []

    n = len(ref_words)
    m = len(whisper_times)

    if n == m:
        return [[ref_words[i], whisper_times[i][0], whisper_times[i][1]] for i in range(n)]

    # Count mismatch: distribute evenly within Whisper's actual time span
    t_start = whisper_times[0][0]
    t_end = whisper_times[-1][1]
    duration = max(t_end - t_start, 0.01)
    step = duration / n
    return [[w, t_start + i * step, t_start + (i + 1) * step] for i, w in enumerate(ref_words)]


@app.route("/align")
def align(payload: dict) -> dict:
    audio_path = str(Path(payload["audio_path"]).expanduser().resolve())
    text = str(payload["text"])

    ref_words, target_indices = _preprocess_tagged_text(text)
    if not ref_words:
        return {"ok": True, "boundaries": [], "target_indices": []}

    model = _get_model()
    segments, _ = model.transcribe(
        audio_path,
        language="zh",
        initial_prompt=" ".join(ref_words),
        word_timestamps=True,
    )

    whisper_times: list[tuple[float, float]] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                whisper_times.append((float(w.start), float(w.end)))

    if not whisper_times:
        return {"ok": True, "boundaries": [], "target_indices": []}

    boundaries = _map_to_ref(ref_words, whisper_times)
    target_indices = [i for i in target_indices if i < len(boundaries)]
    return {"ok": True, "boundaries": boundaries, "target_indices": target_indices}


def main() -> None:
    global MODEL_PATH, DEVICE
    parser = argparse.ArgumentParser(description="Alignment HTTP service (Whisper word timestamps)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    MODEL_PATH = str(Path(args.model).expanduser().resolve())
    DEVICE = args.device
    app.serve(args.host, args.port)


if __name__ == "__main__":
    main()
