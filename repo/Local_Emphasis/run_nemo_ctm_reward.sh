#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/repo/NeMo/.venv/bin/python}"
ALIGN_SCRIPT="${ALIGN_SCRIPT:-${PROJECT_ROOT}/repo/NeMo/tools/nemo_forced_aligner/run_align.sh}"

AUDIO_PATH="${1:-${PROJECT_ROOT}/repo/NeMo/tools/nemo_forced_aligner/asset/test.wav}"
TEXT_PATH="${2:-${PROJECT_ROOT}/repo/NeMo/tools/nemo_forced_aligner/asset/test.txt}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/repo/NeMo/tools/nemo_forced_aligner/asset/output}"
MANIFEST_PATH="${4:-${PROJECT_ROOT}/repo/NeMo/tools/nemo_forced_aligner/asset/sample_manifest.jsonl}"
QUOTE_INDEX="${QUOTE_INDEX:-all}"
PRESET="${PRESET:-rl_robust}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -x "${ALIGN_SCRIPT}" && ! -f "${ALIGN_SCRIPT}" ]]; then
  echo "Align script not found: ${ALIGN_SCRIPT}" >&2
  exit 1
fi

bash "${ALIGN_SCRIPT}" "${AUDIO_PATH}" "${TEXT_PATH}" "${OUTPUT_DIR}" "${MANIFEST_PATH}"

AUDIO_STEM="$(basename -- "${AUDIO_PATH}")"
AUDIO_STEM="${AUDIO_STEM%.*}"
CTM_PATH="${OUTPUT_DIR}/ctm/words/${AUDIO_STEM}.ctm"

if [[ ! -f "${CTM_PATH}" ]]; then
  echo "Word CTM not found after alignment: ${CTM_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/nemo_ctm_reward_example.py" \
  --audio "${AUDIO_PATH}" \
  --ctm "${CTM_PATH}" \
  --quote-index "${QUOTE_INDEX}" \
  --preset "${PRESET}"
