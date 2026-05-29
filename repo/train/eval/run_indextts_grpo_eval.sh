#!/usr/bin/env bash
# IndexTTS 推理(使用 GRPO 訓練後 checkpoint),不算 reward。
# 結構參考 repo/train/run_indextts_grpo_full.sh,但只跑推理 → 不需要起 reward services。
#
# 用法:
#   cd /srv/RL_project
#   bash repo/train/eval/run_indextts_grpo_eval.sh [extra args 透傳給 .py]
#
# 預設值可用環境變數覆寫:
#   GRPO_CKPT, INPUT_TEXT, OUTPUT_ROOT, MODEL_NAME, EMOTION_SEED
#
# 範例:
#   GRPO_CKPT=repo/train/runs/try1_arc/step_2000.pth \
#   OUTPUT_ROOT=repo/train/runs/eval/step2000_test/wav \
#   MODEL_NAME=indextts_step2000 \
#   bash repo/train/eval/run_indextts_grpo_eval.sh
#
#   # smoke test 3 筆
#   bash repo/train/eval/run_indextts_grpo_eval.sh --limit 3

set -euo pipefail

# nvcc 12.0 沒支援 sm_120,讓 driver JIT
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT_DIR}"

# ── Python venv ───────────────────────────────────────────────────────────────
VENV_INDEXTTS="${ROOT_DIR}/repo/index-tts/.venv/bin/python"

# ── 預設參數(可被環境變數覆寫)─────────────────────────────────────────────
GRPO_CKPT="${GRPO_CKPT:-repo/train/runs/try1_arc/step_1169.pth}"
INPUT_TEXT="${INPUT_TEXT:-TAT-Vol1/TAT-Vol1-test_manifest_enhanced/text}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/srv/RL_project/repo/outputs/step_1169.pth}"
MODEL_NAME="${MODEL_NAME:-indextts_step1169}"
EMOTION_SEED="${EMOTION_SEED:-42}"

# ── 環境檢查 ─────────────────────────────────────────────────────────────────
if [[ ! -x "${VENV_INDEXTTS}" ]]; then
    echo "[error] 找不到 Python:${VENV_INDEXTTS}"
    exit 1
fi
if [[ ! -f "${INPUT_TEXT}" ]]; then
    echo "[error] 找不到 input text:${INPUT_TEXT}"
    exit 1
fi
if [[ ! -f "${GRPO_CKPT}" ]]; then
    echo "[error] 找不到 GRPO ckpt:${GRPO_CKPT}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[run] ROOT_DIR    = ${ROOT_DIR}"
echo "[run] INPUT_TEXT  = ${INPUT_TEXT}"
echo "[run] GRPO_CKPT   = ${GRPO_CKPT}"
echo "[run] OUTPUT_ROOT = ${OUTPUT_ROOT}"
echo "[run] MODEL_NAME  = ${MODEL_NAME}"
echo "[run] EMOTION_SEED= ${EMOTION_SEED}"
echo ""
echo "[run] 開始 IndexTTS 推理(無 reward services)..."
echo ""

"${VENV_INDEXTTS}" -m repo.train.eval.indextts_grpo_eval \
    --input "${INPUT_TEXT}" \
    --grpo-ckpt "${GRPO_CKPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --model-name "${MODEL_NAME}" \
    --index-tts-dir "${ROOT_DIR}/repo/index-tts" \
    --emotion-seed "${EMOTION_SEED}" \
    "$@"

echo ""
echo "[run] 推理完成。manifest: ${OUTPUT_ROOT}/manifest.json"
