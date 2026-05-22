#!/usr/bin/env bash
set -euo pipefail

cd /srv/RL_project/repo/RL

# The base uv environment already contains the dependencies needed for this
# local TTS mock. Reusing it avoids building CUDA extensions that require a
# CUDA toolkit version matching torch cu130.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# NeMo RL manages its own environment through uv. This may install Ray,
# Hydra, torch/RL dependencies, and model backend packages on first run.
uv run examples/run_grpo.py \
  --config examples/configs/tts_grpo_mock.yaml \
  "$@"
