"""UnifiedVoice GRPO training loop.

Usage (from /srv/RL_project):
    /srv/RL_project/repo/index-tts/.venv/bin/python \\
        -m repo.train.indextts_grpo_train \\
        repo/train/config.indextts_grpo.json
"""
from __future__ import annotations

import os
# protobuf >=4 is incompatible with tensorboard's generated pb2 files (via audiotools).
# Setting this before any import fixes the crash without downgrading packages.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import logging
import re
import sys
from pathlib import Path

import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Make both `repo.train.*` and bare `train.*` importable from this entry point
_ROOT = Path(__file__).parent.parent.parent  # /srv/RL_project
_REPO = Path(__file__).parent.parent          # /srv/RL_project/repo
for _p in [str(_ROOT), str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _remove_silence_batch(wav_paths: list[str | None]) -> None:
    """In-place silence removal for a batch of wav files."""
    for path in wav_paths:
        if not path or not Path(path).exists():
            continue
        try:
            audio = AudioSegment.from_file(path)
            chunks = split_on_silence(
                audio,
                min_silence_len=200,
                silence_thresh=-40,
                keep_silence=50,
            )
            if chunks:
                processed = chunks[0]
                for chunk in chunks[1:]:
                    processed += chunk
                processed.export(path, format="wav")
        except Exception:
            continue


def _rotate_checkpoints(ckpt_dir: Path, keep: int) -> None:
    """Remove oldest checkpoints, keeping only the latest `keep` files."""
    ckpts = sorted(ckpt_dir.glob("step_*.pth"))
    for old in ckpts[:-keep]:
        old.unlink()


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
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    keep_last = cfg.get("keep_last_checkpoints", 5)

    # ── W&B ───────────────────────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "rl-tts-grpo"),
            name=wandb_cfg.get("name", cfg.get("run_name")),
            config=cfg,
            mode=wandb_cfg.get("mode", "online"),
        )
        log.info("W&B initialised: project=%s run=%s", wandb_cfg.get("project"), wandb_cfg.get("name"))

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
    policy.snapshot_reference()
    log.info("Reference model snapshot taken (frozen for KL).")
    policy.gpt.gpt.train()   # GPT transformer in train mode; other components stay eval
    log.info("Policy loaded.")

    # ── Reward / dataset setup ─────────────────────────────────────────
    from train.schemas import ServiceSpec
    from train.components import ComponentHub
    from train.dataset import load_jsonl, build_random_control
    from train.grpo import compute_group_advantages, grpo_loss
    from train.machine_translation import translate_text_with_requests

    translation_cfg = cfg.get("translation", {})
    translation_mode = translation_cfg.get("mode", "taigi_zh_tw_py")
    translation_api_url = translation_cfg.get("api_url", "http://tts001.bronci.com.tw:8802/run/predict")

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

    # Exclude layers that never receive gradients in compute_logprobs
    # (emo_layer, emovec_layer, ConformerEncoder, PerceiverResampler are bypassed)
    _no_grad_names = (
        "emo_layer.", "emovec_layer.",
        "conditioning_encoder.", "perceiver_encoder.",
        "emo_conditioning_encoder.", "emo_perceiver_encoder.",
    )
    gpt_trainable = [
        p for name, p in policy.gpt.named_parameters()
        if not any(name.startswith(s) for s in _no_grad_names)
    ]
    optimizer = torch.optim.AdamW(
        gpt_trainable + list(policy.va_alpha_mlp.parameters()),
        lr=lr,
    )
    prompt_audio = cfg["prompt_audio"]

    step = 0
    log.info("Starting GRPO training: %d updates × %d examples = %d steps.",
             n_updates, len(examples), n_updates * len(examples))

    for _update in range(n_updates):
        for example in examples:
            control = build_random_control(example)
            translated_text = translate_text_with_requests(
                control.tagged_text,
                mode=translation_mode,
                api_url=translation_api_url,
            )
            translated_text = re.sub(r'(?<=")\s+|\s+(?=")', '', translated_text).strip()
            log.info("── Step %d | uid=%s | emotion=%s | va_01=%s | tagged=%r | translated=%r",
                     step, example.uid, control.target_emotion, control.va_01,
                     control.tagged_text[:60], translated_text[:60])

            # 1. Conditioning
            cond = policy.prepare_conditioning(
                prompt_audio,
                emotion_vector=control.emotion_vector,
                va_01=control.va_01,
            )

            # Compute current alpha for logging
            alpha_val: float | None = None
            if control.va_01 is not None:
                with torch.no_grad():
                    va_t = torch.tensor(list(control.va_01), device=device)
                    alpha_val = float(policy.va_alpha_mlp(va_t).item())

            # 2. Rollout
            rollout = policy.rollout(
                text=translated_text,
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

            # 3b. Remove silence from generated wavs
            _remove_silence_batch(wav_paths)

            # 4. Rewards
            rewards: list[float] = []
            all_r_dicts: list[dict] = []
            for k, wav_path in enumerate(wav_paths):
                r_dict = hub.score(example, control, wav_path, translated_text)
                total = sum(
                    float(r_dict.get(name, 0.0)) * reward_weights.get(name, 1.0)
                    for name in enabled_rewards
                )
                rewards.append(total)
                all_r_dicts.append(r_dict)
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
                ref_lp = policy.compute_ref_logprobs(rollout, k)[:T] if beta else None
                loss_k = grpo_loss(
                    new_lp[:T].unsqueeze(0),
                    old_lp[:T].unsqueeze(0).detach(),
                    adv_tensor[k].unsqueeze(0),
                    ref_logprobs=ref_lp.unsqueeze(0) if ref_lp is not None else None,
                    clip_eps=clip_eps,
                    beta=beta,
                )
                total_loss = total_loss + loss_k
                valid_k += 1

            loss_val = float(total_loss.item())
            if valid_k > 0:
                (total_loss / valid_k).backward()
                optimizer.step()
                log.info("   loss=%.6f", loss_val)
            else:
                log.warning("   No valid rollouts; skipping optimizer step.")

            # 6. W&B logging
            if use_wandb:
                log_dict: dict = {
                    "step": step,
                    "loss": loss_val,
                    "reward/mean": sum(rewards) / len(rewards) if rewards else 0.0,
                    "reward/max": max(rewards) if rewards else 0.0,
                    "reward/min": min(rewards) if rewards else 0.0,
                    "advantage/mean": sum(advantages) / len(advantages) if advantages else 0.0,
                    "emotion/target": control.target_emotion,
                }
                if alpha_val is not None:
                    log_dict["mlp/alpha"] = alpha_val
                if control.va_01 is not None:
                    log_dict["va/valence"] = control.va_01[0]
                    log_dict["va/arousal"] = control.va_01[1]
                # Per-reward breakdown (mean across rollouts)
                for name in enabled_rewards:
                    vals = [float(r.get(name, 0.0)) for r in all_r_dicts]
                    log_dict[f"reward/{name}"] = sum(vals) / len(vals)
                wandb.log(log_dict, step=step)

            # 7. Checkpoint (keep latest N)
            ckpt_path = ckpt_dir / f"step_{step:04d}.pth"
            torch.save(
                {
                    "step": step,
                    "model": policy.gpt.state_dict(),
                    "va_alpha_mlp": policy.va_alpha_mlp.state_dict(),
                },
                str(ckpt_path),
            )
            _rotate_checkpoints(ckpt_dir, keep=keep_last)
            log.info("   checkpoint → %s", ckpt_path)
            step += 1

    if use_wandb:
        wandb.finish()
    log.info("Done. %d steps completed.", step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS UnifiedVoice GRPO trainer")
    parser.add_argument("config", help="Path to JSON config file")
    args = parser.parse_args()
    main(args.config)
