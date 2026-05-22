"""UnifiedVoice GRPO training loop.

Usage (from /srv/RL_project):
    /srv/RL_project/repo/index-tts/.venv/bin/python \\
        -m repo.train.indextts_grpo_train \\
        repo/train/config.indextts_grpo.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

# Make both `repo.train.*` and bare `train.*` importable from this entry point
_ROOT = Path(__file__).parent.parent.parent  # /srv/RL_project
_REPO = Path(__file__).parent.parent          # /srv/RL_project/repo
for _p in [str(_ROOT), str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    with open(config_path) as f:
        cfg = json.load(f)

    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ── Model loading ──────────────────────────────────────────────────
    log.info("Loading IndexTTS policy…")
    from train.indextts_policy import IndexTTSPolicy
    from train.indextts_decode import IndexTTSDecoder

    policy = IndexTTSPolicy(
        cfg_path=cfg["indextts_cfg"],
        model_dir=cfg["indextts_model_dir"],
        gpt_checkpoint_path=cfg["gpt_checkpoint"],
        bpe_model_path=cfg["bpe_model"],
        device=device,
    )
    decoder = IndexTTSDecoder(policy)
    log.info("Policy loaded.")

    # ── Reward / dataset setup ─────────────────────────────────────────
    from train.schemas import ServiceSpec
    from train.components import ComponentHub
    from train.dataset import load_jsonl, build_gold_control
    from train.grpo import compute_group_advantages, grpo_loss

    service_specs = {
        name: ServiceSpec(name=name, **scfg)
        for name, scfg in cfg.get("services", {}).items()
    }
    hub = ComponentHub(
        specs=service_specs,
        output_dir=output_dir,
        prompt_audio=cfg.get("prompt_audio"),
        reward_options=cfg.get("reward_options", {}),
    )

    data_path = cfg["data_path"]
    limit = cfg.get("limit", 4)
    examples = load_jsonl(data_path, limit=limit)
    log.info("Loaded %d training examples.", len(examples))

    # ── Hyper-params ───────────────────────────────────────────────────
    group_size = cfg.get("group_size", 2)
    max_mel_tokens = cfg.get("max_mel_tokens", 256)
    top_k = cfg.get("top_k", 30)
    top_p = cfg.get("top_p", 0.8)
    temperature = cfg.get("temperature", 0.8)
    clip_eps = cfg.get("grpo", {}).get("clip_eps", 0.2)
    beta = cfg.get("grpo", {}).get("beta", 0.0)
    reward_weights: dict = cfg.get("reward_weights", {})
    enabled_rewards: set = set(
        cfg.get("reward_options", {}).get("enabled_rewards", ["emotion", "vad", "global_intensity"])
    )
    lr = cfg.get("learning_rate", 1e-5)
    n_updates = cfg.get("updates", 1)

    optimizer = torch.optim.AdamW(policy.gpt.parameters(), lr=lr)
    prompt_audio = cfg["prompt_audio"]

    step = 0
    log.info("Starting GRPO training: %d updates × %d examples = %d steps.",
             n_updates, len(examples), n_updates * len(examples))

    for _update in range(n_updates):
        for example in examples:
            control = build_gold_control(example)
            log.info("── Step %d | uid=%s | emotion=%s | text=%r",
                     step, example.uid, control.target_emotion, control.text[:60])

            # 1. Conditioning
            cond = policy.prepare_conditioning(
                prompt_audio, emotion_vector=control.emotion_vector
            )

            # 2. Rollout
            rollout = policy.rollout(
                text=control.text,
                cond=cond,
                group_size=group_size,
                max_mel_tokens=max_mel_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            for k in range(group_size):
                log.info("   rollout k=%d | codes_len=%d", k, len(rollout.codes[k]))

            # 3. Decode → wav
            wav_paths: list[str | None] = []
            for k in range(group_size):
                if len(rollout.codes[k]) == 0:
                    log.warning("   k=%d empty codes; skipping decode.", k)
                    wav_paths.append(None)
                    continue
                wav_path = decoder.decode(rollout, k, cond, output_dir, step, k)
                wav_paths.append(wav_path)
                log.info("   k=%d | wav=%s", k, wav_path)

            # 4. Rewards
            rewards: list[float] = []
            for k, wav_path in enumerate(wav_paths):
                r_dict = hub.score(example, control, wav_path)
                total = sum(
                    float(r_dict.get(name, 0.0)) * reward_weights.get(name, 1.0)
                    for name in enabled_rewards
                )
                rewards.append(total)
                log.info("   k=%d | reward=%.4f | %s",
                         k, total,
                         {n: round(float(r_dict.get(n, 0.0)), 4) for n in enabled_rewards})

            advantages = compute_group_advantages(rewards)
            adv_tensor = torch.tensor(advantages, device=device, dtype=torch.float32)
            log.info("   rewards=%s | advantages=%s", rewards, advantages)

            # 5. GRPO loss + backward
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            valid_k = 0

            for k in range(group_size):
                if len(rollout.codes[k]) == 0:
                    continue
                new_lp = policy.compute_logprobs(rollout, k)        # [T]
                old_lp = rollout.old_logprobs[k].to(device)         # [T]
                T = min(new_lp.shape[0], old_lp.shape[0])
                if T == 0:
                    continue
                loss_k = grpo_loss(
                    new_lp[:T].unsqueeze(0),
                    old_lp[:T].unsqueeze(0).detach(),
                    adv_tensor[k].unsqueeze(0),
                    clip_eps=clip_eps,
                    beta=beta,
                )
                total_loss = total_loss + loss_k
                valid_k += 1

            if valid_k > 0:
                (total_loss / valid_k).backward()
                optimizer.step()
                log.info("   loss=%.6f", total_loss.item())
            else:
                log.warning("   No valid rollouts; skipping optimizer step.")

            # 6. Checkpoint
            ckpt_path = output_dir / "checkpoints" / f"step_{step:04d}.pth"
            torch.save(
                {
                    "step": step,
                    "model": policy.gpt.gpt.state_dict(),
                },
                str(ckpt_path),
            )
            log.info("   checkpoint → %s", ckpt_path)
            step += 1

    log.info("Done. %d steps completed.", step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS UnifiedVoice GRPO trainer")
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()
    main(args.config)
