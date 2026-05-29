#!/bin/bash
<<'EOF'
cd /srv/RL_project/repo/calculate_ss
SOURCE_AUDIO=/srv/RL_project/repo/index-tts/examples/prompts/FT0BAE.mp3 \
MANIFEST=/srv/RL_project/repo/outputs/manifest_indextts.json \
AUDIO_DIR=/srv/RL_project/repo/outputs/wav_by_model/indextts_try5 \
./run_cal_ss.sh \
  --summary-output /srv/RL_project/repo/outputs/wav_by_model/ss_summary_indextts_try5.md \
  --details-output /srv/RL_project/repo/outputs/wav_by_model/ss_eval_indextts_try5.csv


EOF

set -euo pipefail

cd "$(dirname "$0")"

PROJECT_ROOT="$(cd ../.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
SOURCE_AUDIO="${SOURCE_AUDIO:-./source.mp3}"
TARGET_AUDIO="${TARGET_AUDIO:-}"
MANIFEST="${MANIFEST:-../outputs/manifest_all.json}"
AUDIO_DIR="${AUDIO_DIR:-../outputs/wav}"
SAVEDIR="${SAVEDIR:-$PROJECT_ROOT/models/speechbrain-spkrec-ecapa-voxceleb}"
MODEL="${MODEL:-$SAVEDIR}"
DEVICE="${DEVICE:-cpu}"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/models/cache}"

mkdir -p "$SAVEDIR" "$HF_HUB_CACHE" "$XDG_CACHE_HOME"

CMD=(
  "$PYTHON_BIN" cal_ss.py
  --source "$SOURCE_AUDIO"
  --model "$MODEL"
  --savedir "$SAVEDIR"
  --device "$DEVICE"
)

if [[ -n "$TARGET_AUDIO" ]]; then
  CMD+=(--target "$TARGET_AUDIO")
else
  CMD+=(--manifest "$MANIFEST" --audio-dir "$AUDIO_DIR")
fi

"${CMD[@]}" "$@"
