#!/usr/bin/env bash
# 啟動所有 reward service，再跑 IndexTTS GRPO 訓練。
# 用法：
#   cd /srv/RL_project
#   bash repo/train/run_indextts_grpo_full.sh [config.json]
#
# 預設 config：repo/train/config.indextts_grpo.real.json
# 結束時（Ctrl-C 或 exit）自動關閉所有背景服務。

set -euo pipefail

# nvcc 12.0 does not support sm_120 (Blackwell). Compile for sm_89 + PTX so
# the GPU driver can JIT-compile to sm_120 at runtime.
export TORCH_CUDA_ARCH_LIST="8.9+PTX"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-repo/train/config.indextts_grpo.real.json}"

VENV="${ROOT_DIR}/.venv/bin/python"
VENV_W2V2="${ROOT_DIR}/repo/w2v2-how-to/venv_w2v2/bin/python"
VENV_NEMO="${ROOT_DIR}/repo/NeMo/.venv/bin/python"
VENV_INDEXTTS="${ROOT_DIR}/repo/index-tts/.venv/bin/python"
VENV_ASR="${ROOT_DIR}/repo/ASR_sys/.venv/bin/python"

LOG_DIR="${ROOT_DIR}/repo/train/logs"
mkdir -p "${LOG_DIR}"

# ── PID 追蹤 ────────────────────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "[run] 關閉所有背景服務..."
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null && echo "[run] killed PID ${pid}" || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "[run] 清理完成。"
}
trap cleanup EXIT INT TERM

# ── 工具函式 ─────────────────────────────────────────────────────────────────
start_service() {
    local name="$1"; shift
    local logfile="${LOG_DIR}/${name}.log"
    echo "[run] 啟動 ${name} → log: ${logfile}"
    "$@" > "${logfile}" 2>&1 &
    PIDS+=($!)
}

wait_for_service() {
    local name="$1"
    local port="$2"
    local max_wait=60
    local elapsed=0
    echo -n "[run] 等待 ${name} (port ${port})..."
    until curl -sf -X POST "http://127.0.0.1:${port}/health" -d '{}' \
            -H 'Content-Type: application/json' > /dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            echo " TIMEOUT（${max_wait}s）"
            echo "[error] ${name} 啟動逾時，請查看 ${LOG_DIR}/${name}.log"
            exit 1
        fi
        echo -n "."
    done
    echo " ready (${elapsed}s)"
}

# ── 環境檢查 ─────────────────────────────────────────────────────────────────
for bin in "${VENV}" "${VENV_W2V2}" "${VENV_INDEXTTS}" "${VENV_ASR}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "[error] 找不到 Python：${bin}"
        exit 1
    fi
done

if [[ ! -f "${CONFIG}" ]]; then
    echo "[error] 找不到 config：${CONFIG}"
    exit 1
fi

echo "[run] ROOT_DIR  = ${ROOT_DIR}"
echo "[run] CONFIG    = ${CONFIG}"
echo ""

# ── 啟動 Reward Services ─────────────────────────────────────────────────────

# emotion (port 8104) — funasr + emotion2vec
start_service "emotion" \
    "${VENV}" -m repo.train.services.emotion_reward_service \
    --host 127.0.0.1 --port 8104 \
    --model-dir "${ROOT_DIR}/models/emotion2vec_plus_large"

# local emphasis (port 8106) — torchaudio
start_service "local_emphasis" \
    "${VENV}" -m repo.train.services.local_emphasis_service \
    --host 127.0.0.1 --port 8106

# VAD / VA (port 8107) — audonnx，需要 venv_w2v2
start_service "vad" \
    "${VENV_W2V2}" -m repo.train.services.vad_service \
    --host 127.0.0.1 --port 8107

# faster-whisper Taigi alignment (port 8108) — 用主 venv
start_service "alignment" \
    "${VENV}" -m repo.train.services.alignment_service \
    --host 127.0.0.1 --port 8108

# ── 等服務全部就緒 ────────────────────────────────────────────────────────────
wait_for_service "emotion"       8104
wait_for_service "local_emphasis" 8106
wait_for_service "vad"           8107
wait_for_service "alignment"     8108

echo ""
echo "[run] 所有服務就緒，開始 GRPO 訓練..."
echo ""

# ── IndexTTS GRPO 訓練（前景執行，Ctrl-C 可中斷）────────────────────────────
"${VENV_INDEXTTS}" -m repo.train.indextts_grpo_train "${CONFIG}"

echo ""
echo "[run] 訓練完成。"
