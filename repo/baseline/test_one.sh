#!/bin/bash
# Generate one utterance from text_origin using IndexTTS2.
#
# Usage:
#   bash baseline/test_one.sh <row_number_or_utt_id> [OUTPUT_WAV]
#
# Examples:
#   bash baseline/test_one.sh 5
#   bash baseline/test_one.sh condenser-IU_IUF0005_0014-1.5-03
#   bash baseline/test_one.sh 5 outputs/test/sample5.wav

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash baseline/test_one.sh <row_number_or_utt_id> [OUTPUT_WAV]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
INDEX_TTS_DIR="$REPO_DIR/index-tts"
INPUT="$SCRIPT_DIR/text_origin"

SELECTOR="$1"
OUTPUT="${2:-}"

# paths relative to INDEX_TTS_DIR (same as batch_indexTTS.sh)
SPEAKER="examples/prompts/FT0BAE.mp3"
GPT_CKPT="/srv/RL_project/models/IndexTTS_trained/GPTs/trained_200hr_4e_step202000.pth"
TOKENIZER="/srv/RL_project/models/IndexTTS_trained/checkpoints/bpe_extended.model"
CONFIG="checkpoints/config.yaml"
DEVICE="cuda:0"
SEED=42
# neutral emotion vector [joy anger sadness fear disgust low_mood surprise calm]
EMO_VEC="0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4"

[ -f "$INPUT" ]         || { echo "[ERROR] Input not found: $INPUT"; exit 1; }
[ -d "$INDEX_TTS_DIR" ] || { echo "[ERROR] index-tts dir not found: $INDEX_TTS_DIR"; exit 1; }

if command -v uv &>/dev/null; then
    UV="uv"
elif [ -x "$REPO_DIR/.venv/bin/uv" ]; then
    UV="$REPO_DIR/.venv/bin/uv"
else
    echo "[ERROR] 'uv' not found."
    exit 1
fi

# ── resolve selector to (id, text) ───────────────────────────────────────────
if [[ "$SELECTOR" =~ ^[0-9]+$ ]]; then
    LINE=$(awk -v n="$SELECTOR" 'NF>=2 {i++; if (i==n) {print; exit}}' "$INPUT")
    DEFAULT_TAG="row${SELECTOR}"
else
    LINE=$(awk -v id="$SELECTOR" '$1==id {print; exit}' "$INPUT")
    DEFAULT_TAG="$SELECTOR"
fi

[ -n "$LINE" ] || { echo "[ERROR] No matching utterance for: $SELECTOR"; exit 1; }

UTT_ID=$(awk '{print $1}' <<< "$LINE")
TEXT=$(awk '{$1=""; sub(/^ /,""); print}' <<< "$LINE")

if [ -z "$OUTPUT" ]; then
    OUTPUT="$REPO_DIR/outputs/test/${DEFAULT_TAG}.wav"
fi
mkdir -p "$(dirname "$OUTPUT")"

echo "[INFO] id   : $UTT_ID"
echo "[INFO] text : $TEXT"
echo "[INFO] out  : $OUTPUT"
echo ""

cd "$INDEX_TTS_DIR"

"$UV" run python inference_script.py \
    --config         "$CONFIG" \
    --speaker        "$SPEAKER" \
    --text           "$TEXT" \
    --output         "$OUTPUT" \
    --device         "$DEVICE" \
    --gpt-checkpoint "$GPT_CKPT" \
    --tokenizer      "$TOKENIZER" \
    --fp16 \
    --skip-normalizer \
    --seed           "$SEED" \
    --emo-vector     $EMO_VEC

# trim leading/trailing silence
"$UV" run python "$SCRIPT_DIR/trim_silence.py" "$OUTPUT" --top-db 40 --pad-ms 50
