"""Faster-Whisper Taigi alignment HTTP service.

API:
    POST /align  {audio_path: str, text: str}
    → {ok: true, boundaries: [[word, start, end], ...], target_indices: [int, ...]}

`text` should be the tagged text with "" around emphasis words.
`target_indices` are the indices into `boundaries` that were inside quotes.
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


def _preprocess_tagged_text(text: str) -> tuple[str, list[int]]:
    """Strip quote markers; return (clean_text, target_word_indices)."""
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

    return " ".join(clean_words), target_indices


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        compute = "int8" if DEVICE == "cpu" else "float16"
        _MODEL = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=compute)
    return _MODEL


def _distribute_words(
    seg_chunks: list[dict],
    ref_words: list[str],
) -> list[list]:
    """Distribute reference words across segment timestamps proportionally."""
    if not ref_words or not seg_chunks:
        return []

    total_dur = sum(
        max(float(c["end"]) - float(c["start"]), 0.0)
        for c in seg_chunks
    )
    if total_dur <= 0:
        t0 = float(seg_chunks[0]["start"])
        t1 = float(seg_chunks[-1]["end"])
        dur = max(t1 - t0, 0.01)
        step = dur / len(ref_words)
        return [[w, t0 + i * step, t0 + (i + 1) * step] for i, w in enumerate(ref_words)]

    n = len(ref_words)
    seg_counts: list[int] = []
    for c in seg_chunks:
        seg_dur = max(float(c["end"]) - float(c["start"]), 0.0)
        seg_counts.append(max(1, round(n * seg_dur / total_dur)))

    while sum(seg_counts) > n:
        seg_counts[seg_counts.index(max(seg_counts))] -= 1
    while sum(seg_counts) < n:
        seg_counts[seg_counts.index(min(seg_counts))] += 1

    boundaries: list[list] = []
    word_idx = 0
    for chunk, count in zip(seg_chunks, seg_counts):
        t0 = float(chunk["start"])
        t1 = float(chunk["end"])
        step = max(t1 - t0, 0.01) / count
        for j in range(count):
            if word_idx >= n:
                break
            boundaries.append([ref_words[word_idx], t0 + j * step, t0 + (j + 1) * step])
            word_idx += 1

    return boundaries


@app.route("/align")
def align(payload: dict) -> dict:
    audio_path = str(Path(payload["audio_path"]).expanduser().resolve())
    text = str(payload["text"])

    clean_text, target_indices = _preprocess_tagged_text(text)
    ref_words = [w for w in clean_text.split() if w.strip()]

    if not ref_words:
        return {"ok": True, "boundaries": [], "target_indices": []}

    model = _get_model()
    segments, _ = model.transcribe(
        audio_path,
        language="zh",
        initial_prompt=clean_text,
    )

    seg_chunks = [
        {"start": seg.start, "end": seg.end}
        for seg in segments
    ]

    if not seg_chunks:
        return {"ok": True, "boundaries": [], "target_indices": []}

    boundaries = _distribute_words(seg_chunks, ref_words)
    target_indices = [i for i in target_indices if i < len(boundaries)]
    return {"ok": True, "boundaries": boundaries, "target_indices": target_indices}


def main() -> None:
    global MODEL_PATH, DEVICE
    parser = argparse.ArgumentParser(description="Taigi faster-whisper alignment HTTP service")
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
