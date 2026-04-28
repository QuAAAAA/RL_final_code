#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
AUDIO_PATH="${AUDIO_PATH:-./MT0A2E.mp3}"
MODEL_PATH="${MODEL_PATH:-../../models/faster-whisper-taigi-pinyin-large-v7}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"
BEAM_SIZE="${BEAM_SIZE:-5}"
LANGUAGE="${LANGUAGE:-zh}"
LABEL="${LABEL:-tiong1-san1 tsing3-kong1 loo7-khau2.}"

"$PYTHON_BIN" faster_whisper_api.py \
  --audio "$AUDIO_PATH" \
  --model "$MODEL_PATH" \
  --device "$DEVICE" \
  --compute-type "$COMPUTE_TYPE" \
  --beam-size "$BEAM_SIZE" \
  --language "$LANGUAGE" \
  --label "$LABEL" \
  "$@"
