# Train API Orchestration

First version of the training-side pipeline matching `Midterm_Report/asset/architect.png`.

The important design choice is that model components are not imported into one Python
environment. Each heavy component runs as its own HTTP process, and the trainer calls
them through JSON APIs:

- `asqp`: text -> aspect/opinion/VA/emphasis controls
- `tts`: text + control -> wav path
- `asr`: wav -> transcription / WER reward
- `emotion`: wav + target emotion -> emotion classification reward
- `speaker`: wav + prompt wav -> speaker similarity reward
- `local_emphasis`: wav + word boundaries -> local emphasis reward
- `vad`: wav -> VAD logits, with VA exported as `{valence, arousal}`

## Quick dry run

This uses the dataset gold labels and mock services, so it checks the orchestration
without loading CUDA models.

```bash
cd /srv/RL_project
python -m repo.train.pipeline \
  --data data/va_train/zho_restaurant_train_alltasks_tagged.jsonl \
  --config repo/train/config.mock.json \
  --limit 3
```

## Real process layout

Start each service in the environment that owns its dependencies. For example:

```bash
# env: ASR_sys / transformers
python -m repo.train.services.asr_service \
  --host 127.0.0.1 \
  --port 8103 \
  --model repo/ASR_sys/ASR/whisper-large-v3-turbo-finetuned-pinyin \
  --processor repo/ASR_sys/ASR/whisper-large-v3-turbo-finetuned-pinyin \
  --language zh

# env: local emphasis / torch audio
python -m repo.train.services.local_emphasis_service --host 127.0.0.1 --port 8106

# env: w2v2-how-to / audonnx
python -m repo.train.services.vad_service --host 127.0.0.1 --port 8107
```

Then point `config.json` URLs at those services and run the pipeline.

The TTS service can either proxy an existing Gradio server or be replaced by an
IndexTTS service later. See `config.example.json`.

## Reward Rules

Total reward is still a weighted sum over numeric fields in `reward_weights`;
non-scalar detail fields are kept in the rollout JSON but ignored by aggregation.
Current config sets every reward weight to `1.0`. ASR and speaker are optional:
add or remove them in `reward_options.enabled_rewards` without changing the
aggregation interface.

- `asr`: optional. Uses the ASR_sys Whisper path from
  `repo/ASR_sys/src/eval_indextts.py`: `AutoModelForSpeechSeq2Seq`,
  `AutoProcessor`, `transcribe_batch()`, tone stripping, and WER. The reward is
  `max(0, 1 - WER)`, so lower WER is better.
- `emotion`: `+5` when predicted emotion matches the target, `-1` when it does
  not. This is emotion correctness, not intensity.
- `speaker`: optional. Cosine similarity between prompt speaker audio and
  generated audio.
- `local_emphasis`: local pitch/energy emphasis reward from the emphasis service.
  With the default `rl_robust` preset, the theoretical range is approximately
  `[-2.0, 2.6]`: pitch/energy soft terms are clipped to `[-1, 1]`, hard terms
  contribute up to `0.2`, and rank terms up to `0.4`. If there is no emphasis
  target, the pipeline returns `0.0`.
- `vad`: dataset VA is mapped from `1~9` to `0~1` with `((value - 1) / 8)`.
  Generated audio VA comes from `w2v2-how-to`. Reward is
  `max(0, 1 - mean(abs(pred_v - target_v), abs(pred_a - target_a)))`.
- `global_intensity`: measures only VAD distance from neutral, not emotion
  direction. It computes `d = ||v_y_hat - mu_neutral||_2`, gives an interval hit
  reward when `d` is inside `target_range`, and adds a Gaussian reward around
  the interval midpoint. The scalar range is theoretically about `[0, 2]`.

Example: `4.00#5.00` means dataset VA `(4, 5)`, normalized to
`((4 - 1) / 8, (5 - 1) / 8) = (0.375, 0.5)`.

## Future Work

The current `global_intensity` baseline uses a fixed neutral center:

```json
"mu_neutral": [0.5, 0.5, 0.5],
"target_range": [0.4, 0.6]
```

This is only a first-pass assumption for the w2v2 VAD space. A better version
should estimate `mu_neutral` from real neutral samples:

```text
mu_neutral = mean(VAD(audio) for audio labeled/predicted as neutral)
```

The weak/medium/strong distance thresholds should also be calibrated from data
rather than hand-picked. Two practical options:

- Use generated or real neutral/weak/medium/strong speech, compute
  `d = ||VAD(audio) - mu_neutral||_2`, then set thresholds from percentiles.
- Use dataset VA labels mapped to `0~1`, compute distance from neutral VA, and
  initialize ranges such as weak, medium, strong before refining with real audio.

This matters because `global_intensity` intentionally measures only emotional
strength. It should not decide whether the emotion direction is correct; that is
handled by `vad` and `emotion` correctness rewards.

## NeMo RL Integration

`repo/RL` is now wired through a custom NeMo RL environment named
`tts_reward`. The environment lives in:

```text
repo/RL/nemo_rl/environments/tts_reward_environment.py
```

It calls the existing `repo/train` `ComponentHub`, so all heavy components still
use the same process/API boundaries. The companion data processor
`tts_reward_data_processor` preserves `Text`, `TaggedText`, and `Quadruplet` as
environment metadata. The exported NeMo RL JSONL is:

```text
repo/train/nemo_rl_data/tts_va_train.jsonl
```

The starter NeMo RL config is:

```text
repo/RL/examples/configs/tts_grpo_mock.yaml
```

Run it with:

```bash
repo/train/run_nemo_rl_tts_grpo.sh
```

or directly:

```bash
cd /srv/RL_project/repo/RL
uv run examples/run_grpo.py --config examples/configs/tts_grpo_mock.yaml
```

The current config uses `repo/train/config.mock.json`, so it is meant as a first
connection test. For real model updates, switch `train_config_path` to a config
that points at live TTS/VAD/emotion/local-emphasis services and set an actual
policy model/checkpoint in the NeMo RL config.

## W&B Logging

`train_loop.py` supports W&B logging through the `wandb` config block. The
default configs use offline mode so runs do not block on login:

```json
"wandb": {
  "enabled": true,
  "project": "rl-tts-grpo",
  "name": "mock-grpo-smoke",
  "mode": "offline"
}
```

Each GRPO group logs reward mean/std/min/max, best candidate, best reward, and
mean scalar values for each enabled reward component. Offline runs can be synced
later with the `wandb sync <offline-run-dir>` command printed in the run log.

## GRPO

Collect grouped rollouts with:

```bash
python -m repo.train.train_loop \
  --data data/va_train/zho_restaurant_train_alltasks_tagged.jsonl \
  --config repo/train/config.mock.json \
  --limit 3
```

For each input, the trainer samples `grpo.group_size` candidates, scores them,
computes group-normalized advantages `(reward - group_mean) / group_std`, and
writes `grpo_rollouts.jsonl`. If a `policy` service is configured, it sends the
group to `/grpo/update`; otherwise it just produces rollouts for the policy env
to consume. `repo.train.grpo.grpo_loss()` contains the clipped GRPO surrogate
loss for the policy-side PyTorch implementation.

