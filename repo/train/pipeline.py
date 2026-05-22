from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .components import ComponentHub, control_to_json
from .dataset import load_jsonl
from .schemas import GenerationResult
from .service_runner import ServiceManager, spec_from_config


def load_config(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def total_reward(rewards: dict, weights: dict) -> float:
    total = 0.0
    for name, weight in weights.items():
        value = rewards.get(name, 0.0)
        if isinstance(value, (int, float)):
            total += float(value) * float(weight)
    return total


def run_pipeline(data_path: str | Path, config: dict, limit: int | None = None) -> list[GenerationResult]:
    output_dir = Path(config.get("output_dir", "repo/train/runs/default")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {name: spec_from_config(name, item) for name, item in config.get("services", {}).items()}
    manager_specs = [spec for spec in specs.values() if spec.command]
    examples = load_jsonl(data_path, limit=limit)
    weights = config.get("reward_weights", {})

    results: list[GenerationResult] = []
    with ServiceManager(manager_specs):
        hub = ComponentHub(
            specs,
            output_dir=output_dir,
            prompt_audio=config.get("prompt_audio"),
            reward_options=config.get("reward_options"),
        )
        for idx, example in enumerate(examples, start=1):
            control = hub.predict_control(example)
            wav_path = hub.synthesize(example, control)
            rewards = hub.score(example, control, wav_path)
            score = total_reward(rewards, weights)
            result = GenerationResult(
                uid=example.uid,
                wav_path=wav_path,
                control=control,
                rewards=rewards,
                total_reward=score,
            )
            results.append(result)
            print(f"[{idx}/{len(examples)}] {example.uid} reward={score:.4f} wav={wav_path}")

    write_results(output_dir / "results.jsonl", results)
    return results


def write_results(path: Path, results: list[GenerationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in results:
            payload = asdict(item)
            payload["control"] = control_to_json(item.control)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[train] wrote {len(results)} rows to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-pass RL TTS API pipeline.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="repo/train/config.mock.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.data, load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()

