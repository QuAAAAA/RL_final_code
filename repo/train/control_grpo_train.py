from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .components import control_to_json
from .control_policy import ControlAction, HashTextEncoder, TinyControlPolicy
from .dataset import build_gold_control, load_jsonl
from .global_intensity_reward import global_emotion_intensity_reward_json
from .grpo import compute_group_advantages, grpo_loss
from .gym_style import (
    attach_rollout_key,
    materialize_training_rows,
    materialized_inputs_path,
    sort_by_rollout_key,
    write_aggregate_metrics,
    write_jsonl,
)
from .pipeline import load_config


def _init_wandb(config: dict, output_dir: Path):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; continuing without W&B logging")
        return None

    mode = wandb_cfg.get("mode", "offline")
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", mode)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    return wandb.init(
        project=wandb_cfg.get("project", "rl-tts-grpo"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or config.get("run_name"),
        mode=mode,
        config=config,
        dir=str(wandb_dir),
        reinit="finish_previous",
    )


def score_control_action(action: ControlAction, gold_control, reward_cfg: dict[str, Any]) -> dict[str, float]:
    target_va = gold_control.va_01 or (0.5, 0.5)
    target_valence, target_arousal = target_va
    valence_abs_error = abs(action.valence - target_valence)
    arousal_abs_error = abs(action.arousal - target_arousal)
    mean_abs_error = (valence_abs_error + arousal_abs_error) / 2.0
    vad_reward = max(0.0, 1.0 - mean_abs_error)
    emotion_reward = 5.0 if action.emotion == gold_control.target_emotion else -1.0

    intensity_cfg = reward_cfg.get("global_intensity", {})
    intensity = global_emotion_intensity_reward_json(
        [action.arousal, float(intensity_cfg.get("mock_dominance", 0.5)), action.valence],
        intensity_cfg.get("mu_neutral", [0.5, 0.5, 0.5]),
        intensity_cfg.get("target_range", [0.4, 0.6]),
        sigma=float(intensity_cfg.get("sigma", 0.1)),
    )
    return {
        "emotion": emotion_reward,
        "vad": vad_reward,
        "global_intensity": float(intensity["reward"]),
        "valence_abs_error": valence_abs_error,
        "arousal_abs_error": arousal_abs_error,
    }


def total_reward(rewards: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(rewards.get(name, 0.0)) * float(weight) for name, weight in weights.items())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_policy(policy: TinyControlPolicy, encoder: HashTextEncoder, examples, device: str) -> dict[str, float]:
    import torch

    policy.eval()
    emotion_hits = 0
    va_errors = []
    with torch.no_grad():
        for example in examples:
            gold = build_gold_control(example)
            feature = encoder.encode([example.text]).to(device)[0]
            action = policy.greedy_action(feature)
            emotion_hits += int(action.emotion == gold.target_emotion)
            if gold.va_01 is not None:
                va_errors.append((abs(action.valence - gold.va_01[0]) + abs(action.arousal - gold.va_01[1])) / 2.0)
    policy.train()
    return {
        "eval/emotion_acc": emotion_hits / max(1, len(examples)),
        "eval/va_mae": _mean(va_errors),
    }


def run_control_grpo(data_path: str | Path, config: dict, limit: int | None = None) -> list[dict]:
    import torch

    train_cfg = config.get("control_grpo", {})
    output_dir = Path(config.get("output_dir", "repo/train/runs/control-grpo")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", config.get("grpo", {}).get("seed", 42)))
    random.seed(seed)
    torch.manual_seed(seed)

    device = str(train_cfg.get("device", "cpu"))
    examples = load_jsonl(data_path, limit=limit)
    random.shuffle(examples)
    if not examples:
        raise ValueError("no training examples loaded")

    eval_size = min(int(train_cfg.get("eval_size", 64)), max(1, len(examples) // 5))
    eval_examples = examples[:eval_size]
    train_examples = examples[eval_size:] or examples
    examples_by_uid = {example.uid: example for example in train_examples}

    feature_dim = int(train_cfg.get("feature_dim", 4096))
    encoder = HashTextEncoder(vocab_size=feature_dim)
    policy = TinyControlPolicy(
        feature_dim=feature_dim,
        hidden_dim=int(train_cfg.get("hidden_dim", 128)),
        va_bins=int(train_cfg.get("va_bins", 9)),
    )
    policy.module.to(device)
    policy.emotion_head.to(device)
    policy.valence_head.to(device)
    policy.arousal_head.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    group_size = int(train_cfg.get("group_size", config.get("grpo", {}).get("group_size", 4)))
    epochs = int(train_cfg.get("epochs", 1))
    max_steps = train_cfg.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    clip_eps = float(train_cfg.get("clip_eps", config.get("grpo", {}).get("clip_eps", 0.2)))
    beta = float(train_cfg.get("beta", config.get("grpo", {}).get("beta", 0.0)))
    advantage_eps = float(train_cfg.get("advantage_eps", config.get("grpo", {}).get("advantage_eps", 1e-6)))
    weights = config.get("reward_weights", {"emotion": 1.0, "vad": 1.0, "global_intensity": 1.0})
    reward_cfg = config.get("reward_options", {})
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 50))
    output_path = output_dir / "control_grpo_rollouts.jsonl"
    materialized_rows = materialize_training_rows(
        train_examples,
        repeats=int(train_cfg.get("repeats", 1)),
        base_seed=seed,
    )
    write_jsonl(materialized_inputs_path(output_path), materialized_rows)

    rows: list[dict] = []
    wandb_run = _init_wandb(config, output_dir)
    step = 0
    try:
        for epoch in range(1, epochs + 1):
            random.shuffle(materialized_rows)
            for materialized_row in materialized_rows:
                step += 1
                torch.manual_seed(int(materialized_row["seed"]) + epoch * 1_000_000)
                example = examples_by_uid[materialized_row["uid"]]
                gold = build_gold_control(example)
                feature = encoder.encode([example.text]).to(device)[0]
                actions, indices, old_logprobs = policy.sample_group(feature, group_size)
                rewards = [score_control_action(action, gold, reward_cfg) for action in actions]
                totals = [total_reward(item, weights) for item in rewards]
                advantages = torch.tensor(
                    compute_group_advantages(totals, eps=advantage_eps),
                    dtype=torch.float32,
                    device=device,
                )

                features = feature.unsqueeze(0).expand(group_size, -1)
                indices = {name: value.to(device) for name, value in indices.items()}
                policy_logprobs = policy.action_logprobs(features, indices)
                loss = grpo_loss(
                    policy_logprobs,
                    old_logprobs.to(device),
                    advantages,
                    beta=beta,
                    clip_eps=clip_eps,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()), float(train_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()

                best_idx = max(range(len(totals)), key=lambda idx: totals[idx])
                row = attach_rollout_key(
                    {
                        "step": step,
                        "epoch": epoch,
                        "uid": example.uid,
                        "gold_control": control_to_json(gold),
                        "best_action": asdict(actions[best_idx]),
                        "best_reward": totals[best_idx],
                        "reward_mean": _mean(totals),
                        "loss": float(loss.detach().cpu().item()),
                        "candidates": [
                        {
                            "candidate_index": idx,
                            "action": asdict(action),
                            "rewards": rewards[idx],
                            "total_reward": totals[idx],
                            "advantage": float(advantages[idx].detach().cpu().item()),
                        }
                        for idx, action in enumerate(actions)
                    ],
                    },
                    materialized_row,
                )
                rows.append(row)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": row["loss"],
                            "train/reward_mean": row["reward_mean"],
                            "train/best_reward": row["best_reward"],
                        },
                        step=step,
                    )

                if step == 1 or step % log_every == 0:
                    print(
                        f"[step {step}] epoch={epoch} uid={example.uid} "
                        f"loss={row['loss']:.4f} mean={row['reward_mean']:.4f} best={row['best_reward']:.4f}"
                    )
                if step % eval_every == 0:
                    metrics = evaluate_policy(policy, encoder, eval_examples, device)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=step)
                    print(
                        f"[eval {step}] emotion_acc={metrics['eval/emotion_acc']:.4f} "
                        f"va_mae={metrics['eval/va_mae']:.4f}"
                    )
                if max_steps is not None and step >= max_steps:
                    break
            if max_steps is not None and step >= max_steps:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    rows = sort_by_rollout_key(rows)
    write_jsonl(output_path, rows)
    metrics_path = write_aggregate_metrics(output_path, rows)
    print(f"[control-grpo] wrote aggregate metrics to {metrics_path}")
    torch.save(
        {
            "policy": policy.state_dict(),
            "config": config,
            "step": step,
        },
        output_dir / "control_policy.pt",
    )
    print(f"[control-grpo] wrote checkpoint to {output_dir / 'control_policy.pt'}")
    print(f"[control-grpo] wrote {len(rows)} steps to {output_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal trainable GRPO control-policy experiment.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="repo/train/config.control_grpo.mock.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_control_grpo(args.data, load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
