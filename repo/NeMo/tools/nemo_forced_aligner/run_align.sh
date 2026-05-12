#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/repo/NeMo/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/stt_en_fastconformer_hybrid_large_pc.nemo}"
AUDIO_PATH="${1:-${SCRIPT_DIR}/asset/test.wav}"
TEXT_PATH="${2:-${SCRIPT_DIR}/asset/test.txt}"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/asset/output}"
MANIFEST_PATH="${4:-${SCRIPT_DIR}/asset/sample_manifest.jsonl}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${AUDIO_PATH}" ]]; then
  echo "Audio not found: ${AUDIO_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TEXT_PATH}" ]]; then
  echo "Text not found: ${TEXT_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${MANIFEST_PATH}")" "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${AUDIO_PATH}" "${TEXT_PATH}" "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path

audio_path, text_path, manifest_path = map(Path, sys.argv[1:])
text = " ".join(text_path.read_text(encoding="utf-8-sig").split())
record = {
    "audio_filepath": str(audio_path.resolve()),
    "text": text,
}
manifest_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/align.py" \
  "model_path=${MODEL_PATH}" \
  "manifest_filepath=${MANIFEST_PATH}" \
  "output_dir=${OUTPUT_DIR}" \
  transcribe_device=cpu \
  viterbi_device=cpu \
  batch_size=1 \
  "save_output_file_formats=[ctm]" \
  use_local_attention=false

echo "Done. Output written to: ${OUTPUT_DIR}"
