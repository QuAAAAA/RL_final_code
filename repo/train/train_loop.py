from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .components import ComponentHub, control_to_json
from .dataset import load_jsonl
from .grpo import compute_group_advantages
from .http_client import JsonServiceClient
from .pipeline import load_config, total_reward
from .service_runner import ServiceManager, spec_from_config


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _wandb_log_group(run, ex_idx: int, group_rows: list[dict], rewards_for_group: list[float]) -> None:
    if run is None:
        return

    best = max(group_rows, key=lambda item: item["total_reward"])
    payload = {
        "train/group_index": ex_idx,
        "train/reward_mean": _mean(rewards_for_group),
        "train/reward_std": _std(rewards_for_group),
        "train/reward_min": min(rewards_for_group),
        "train/reward_max": max(rewards_for_group),
        "train/best_candidate": best["candidate_index"],
        "train/best_reward": best["total_reward"],
    }

    scalar_names = set()
    for row in group_rows:
        scalar_names.update(
            name
            for name, value in row.get("rewards", {}).items()
            if isinstance(value, (int, float))
        )
    for name in sorted(scalar_names):
        values = [
            float(row["rewards"][name])
            for row in group_rows
            if isinstance(row.get("rewards", {}).get(name), (int, float))
        ]
        payload[f"reward/{name}_mean"] = _mean(values)

    run.log(payload, step=ex_idx)


def run_grpo(data_path: str | Path, config: dict, limit: int | None = None) -> list[dict]:
    output_dir = Path(config.get("output_dir", "repo/train/runs/grpo")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {name: spec_from_config(name, item) for name, item in config.get("services", {}).items()}
    manager_specs = [spec for spec in specs.values() if spec.command]
    examples = load_jsonl(data_path, limit=limit)
    weights = config.get("reward_weights", {})
    grpo_cfg = config.get("grpo", {})
    group_size = int(grpo_cfg.get("group_size", 4))
    advantage_eps = float(grpo_cfg.get("advantage_eps", 1e-6))
    base_seed = int(grpo_cfg.get("seed", 42))

    rows: list[dict] = []
    wandb_run = _init_wandb(config, output_dir)
    try:
        with ServiceManager(manager_specs):
            hub = ComponentHub(
                specs,
                output_dir=output_dir,
                prompt_audio=config.get("prompt_audio"),
                reward_options=config.get("reward_options"),
            )
            policy = specs.get("policy")
            policy_client = None if policy is None or policy.mock else JsonServiceClient(policy.url or "", timeout=600)

            for ex_idx, example in enumerate(examples, start=1):
                control = hub.predict_control(example)
                group_rows = []
                rewards_for_group = []

                for cand_idx in range(group_size):
                    sample_id = f"g{ex_idx:06d}_c{cand_idx:02d}"
                    seed = base_seed + ex_idx * 1000 + cand_idx
                    wav_path = hub.synthesize(example, control, sample_id=sample_id, seed=seed)
                    rewards = hub.score(example, control, wav_path)
                    score = total_reward(rewards, weights)
                    row = {
                        "group_id": example.uid,
                        "candidate_index": cand_idx,
                        "uid": example.uid,
                        "seed": seed,
                        "wav_path": wav_path,
                        "control": control_to_json(control),
                        "rewards": rewards,
                        "total_reward": score,
                    }
                    group_rows.append(row)
                    rewards_for_group.append(score)

                advantages = compute_group_advantages(rewards_for_group, eps=advantage_eps)
                for row, advantage in zip(group_rows, advantages):
                    row["advantage"] = advantage
                    rows.append(row)

                if policy_client is not None:
                    policy_client.post(
                        "/grpo/update",
                        {
                            "group_id": example.uid,
                            "candidates": group_rows,
                            "clip_eps": float(grpo_cfg.get("clip_eps", 0.2)),
                            "beta": float(grpo_cfg.get("beta", 0.0)),
                        },
                    )

                _wandb_log_group(wandb_run, ex_idx, group_rows, rewards_for_group)
                best = max(group_rows, key=lambda item: item["total_reward"])
                print(
                    f"[{ex_idx}/{len(examples)}] {example.uid} "
                    f"mean={_mean(rewards_for_group):.4f} "
                    f"best={best['total_reward']:.4f} best_c={best['candidate_index']}"
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    write_jsonl(output_dir / "grpo_rollouts.jsonl", rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[grpo] wrote {len(rows)} candidates to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API-based GRPO rollouts for RL TTS training.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="repo/train/config.mock.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_grpo(args.data, load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
