#!/bin/bash
# Batch TTS generation using IndexTTS2 via `uv run python inference_script.py`.
# Processes every utterance in a Kaldi-format text file.
# Skips utterances whose output .wav already exists (resume-safe).
#
# Usage:
#   bash baseline/batch_indexTTS.sh [INPUT] [OUTPUT_ROOT]
#
# Defaults:
#   INPUT       = baseline/text_origin
#   OUTPUT_ROOT = outputs/indextts

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
INDEX_TTS_DIR="$REPO_DIR/index-tts"

INPUT="${1:-$SCRIPT_DIR/text_origin}"
OUTPUT_ROOT="${2:-$REPO_DIR/outputs/indextts}"

# paths relative to INDEX_TTS_DIR (same defaults as run_inference.sh)
SPEAKER="examples/prompts/FT0BAE.mp3"
GPT_CKPT="/srv/RL_project/models/IndexTTS_trained/GPTs/trained_200hr_4e_step202000.pth"
TOKENIZER="/srv/RL_project/models/IndexTTS_trained/checkpoints/bpe_extended.model"
CONFIG="checkpoints/config.yaml"
DEVICE="cuda:0"
SEED=42

# silence trimming (librosa.effects.trim)
TRIM_TOP_DB=40       # below ref by this many dB = silence
TRIM_PAD_MS=50       # keep this much padding on each side

# ── emotion vectors [joy anger sadness fear disgust low_mood surprise calm] ──
declare -A EMO_VECS
EMO_VECS[angry]="0.0 1.4 0.0 0.0 0.0 0.0 0.0 0.0"
EMO_VECS[happy]="1.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
EMO_VECS[fearful]="0.0 0.0 0.0 1.4 0.0 0.0 0.0 0.0"
EMO_VECS[disgusted]="0.0 0.0 0.0 0.0 1.4 0.0 0.0 0.0"
EMO_VECS[sad]="0.0 0.0 1.4 0.0 0.0 0.0 0.0 0.0"
EMO_VECS[surprised]="0.0 0.0 0.0 0.0 0.0 0.0 1.4 0.0"
EMO_VECS[neutral]="0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4"

# ── locate uv ────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    UV="uv"
elif [ -x "$REPO_DIR/.venv/bin/uv" ]; then
    UV="$REPO_DIR/.venv/bin/uv"
else
    echo "[ERROR] 'uv' not found. Install it:"
    echo "    $REPO_DIR/.venv/bin/pip install uv"
    exit 1
fi

[ -f "$INPUT" ]         || { echo "[ERROR] Input not found: $INPUT"; exit 1; }
[ -d "$INDEX_TTS_DIR" ] || { echo "[ERROR] index-tts dir not found: $INDEX_TTS_DIR"; exit 1; }

mkdir -p "$OUTPUT_ROOT"

# ── load tasks ───────────────────────────────────────────────────────────────
mapfile -t TASK_IDS   < <(awk 'NF>=2 {print $1}' "$INPUT")
mapfile -t TASK_TEXTS < <(awk 'NF>=2 {$1=""; sub(/^ /,""); print}' "$INPUT")
N="${#TASK_IDS[@]}"

[ "$N" -gt 0 ] || { echo "[ERROR] No tasks in $INPUT"; exit 1; }

# Assign emotions: seeded shuffle via system python3 (no extra deps needed)
EMOTION_SEED=42
mapfile -t EMOTIONS < <(python3 - <<EOF
import math, random
n = $N
emos = ["angry","happy","fearful","disgusted","sad","surprised","neutral"]
pool = (emos * math.ceil(n / len(emos)))[:n]
random.Random($EMOTION_SEED).shuffle(pool)
print("\n".join(pool))
EOF
)

echo "[INFO] $N tasks  ->  $OUTPUT_ROOT"
echo ""

# ── helpers ──────────────────────────────────────────────────────────────────
wav_duration() {
    python3 -c "import wave; wf=wave.open('$1','rb'); print(wf.getnframes()/wf.getframerate())"
}

fmt_sec() {
    python3 -c "
s=int($1); h,r=divmod(s,3600); m,sec=divmod(r,60)
print(f'{h:02d}:{m:02d}:{sec:02d} ({$1:.1f}s)')"
}

# ── main loop ─────────────────────────────────────────────────────────────────
TOTAL_AUDIO=0
FAIL_COUNT=0
WALL_START=$(date +%s%N)

cd "$INDEX_TTS_DIR"

for i in "${!TASK_IDS[@]}"; do
    idx=$(( i + 1 ))
    id="${TASK_IDS[$i]}"
    text="${TASK_TEXTS[$i]}"
    emotion="${EMOTIONS[$i]}"
    vec="${EMO_VECS[$emotion]}"
    out_path="$OUTPUT_ROOT/$(printf '%06d' "$idx").wav"

    # ── skip if already generated ─────────────────────────────────────────
    if [ -f "$out_path" ]; then
        dur=$(wav_duration "$out_path")
        TOTAL_AUDIO=$(python3 -c "print($TOTAL_AUDIO + $dur)")
        echo "  [$idx/$N] $id  [skip]  ${dur}s  cum=${TOTAL_AUDIO}s"
        continue
    fi

    # ── generate (with retry) ─────────────────────────────────────────────
    success=0
    for attempt in 1 2 3; do
        if "$UV" run python inference_script.py \
                --config      "$CONFIG" \
                --speaker     "$SPEAKER" \
                --text        "$text" \
                --output      "$out_path" \
                --device      "$DEVICE" \
                --gpt-checkpoint "$GPT_CKPT" \
                --tokenizer   "$TOKENIZER" \
                --fp16 \
                --skip-normalizer \
                --seed        "$SEED" \
                --emo-vector  $vec \
                2>&1; then
            success=1
            break
        fi
        echo "[WARN] $id attempt $attempt failed"
    done

    if [ "$success" -eq 0 ] || [ ! -f "$out_path" ]; then
        (( FAIL_COUNT++ )) || true
        echo "[SKIP] $id"
        continue
    fi

    # trim leading/trailing silence in place
    "$UV" run python "$SCRIPT_DIR/trim_silence.py" "$out_path" \
        --top-db "$TRIM_TOP_DB" --pad-ms "$TRIM_PAD_MS" >/dev/null || \
        echo "[WARN] trim failed for $id"

    dur=$(wav_duration "$out_path")
    TOTAL_AUDIO=$(python3 -c "print($TOTAL_AUDIO + $dur)")
    echo "  [$idx/$N] $id  $emotion  ${dur}s  cum=${TOTAL_AUDIO}s"
done

WALL_END=$(date +%s%N)
WALL=$(python3 -c "print(($WALL_END - $WALL_START) / 1e9)")

echo ""
echo "============================================================"
echo "  wall=$(fmt_sec "$WALL")  audio=$(fmt_sec "$TOTAL_AUDIO")  failed=$FAIL_COUNT"
echo "============================================================"
