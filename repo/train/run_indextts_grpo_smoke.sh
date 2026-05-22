#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/srv/RL_project}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/repo/index-tts/.venv/bin/python}"
CONFIG="${CONFIG:-${ROOT_DIR}/repo/train/config.indextts_grpo.json}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
echo "[run] ${PYTHON_BIN} -m repo.train.indextts_grpo_train ${CONFIG}"
"${PYTHON_BIN}" -m repo.train.indextts_grpo_train "${CONFIG}"
