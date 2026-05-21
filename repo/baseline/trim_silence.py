#!/usr/bin/env python3
"""Trim leading/trailing silence from a wav file (in place by default).

Usage:
    python baseline/trim_silence.py <input.wav> [output.wav] [--top-db 40] [--pad-ms 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def trim_wav(path: Path, out: Path, top_db: float, pad_ms: int) -> tuple[float, float]:
    y, sr = librosa.load(str(path), sr=None, mono=False)
    if y.ndim == 1:
        mono = y
    else:
        mono = np.mean(y, axis=0)

    _, idx = librosa.effects.trim(mono, top_db=top_db)
    start, end = int(idx[0]), int(idx[1])

    pad = int(sr * pad_ms / 1000)
    start = max(0, start - pad)
    end = min(mono.shape[-1], end + pad)

    if y.ndim == 1:
        trimmed = y[start:end]
    else:
        trimmed = y[:, start:end]

    orig_s = mono.shape[-1] / sr
    new_s = (end - start) / sr
    sf.write(str(out), trimmed.T if y.ndim > 1 else trimmed, sr)
    return orig_s, new_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, nargs="?", default=None)
    ap.add_argument("--top-db", type=float, default=40.0,
                    help="Below reference by this many dB is silence (default: 40).")
    ap.add_argument("--pad-ms", type=int, default=50,
                    help="Keep this much silence padding on each side (default: 50ms).")
    args = ap.parse_args()
    out = args.output or args.input
    orig, new = trim_wav(args.input, out, args.top_db, args.pad_ms)
    print(f"[trim] {args.input.name}  {orig:.2f}s -> {new:.2f}s")


if __name__ == "__main__":
    main()
