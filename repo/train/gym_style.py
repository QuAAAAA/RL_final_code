from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable


TASK_INDEX_KEY = "_ng_task_index"
ROLLOUT_INDEX_KEY = "_ng_rollout_index"


def materialized_inputs_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_materialized_inputs.jsonl")


def aggregate_metrics_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_aggregate_metrics.json")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_training_rows(examples, repeats: int = 1, base_seed: int = 42) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rows = []
    for task_idx, example in enumerate(examples):
        for rollout_idx in range(repeats):
            rows.append(
                {
                    TASK_INDEX_KEY: task_idx,
                    ROLLOUT_INDEX_KEY: rollout_idx,
                    "uid": example.uid,
                    "seed": base_seed + task_idx * 1000 + rollout_idx,
                    "text": example.text,
                }
            )
    return rows


def rollout_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row[TASK_INDEX_KEY]), int(row[ROLLOUT_INDEX_KEY])


def sort_by_rollout_key(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=rollout_key)


def attach_rollout_key(result: dict[str, Any], materialized_row: dict[str, Any]) -> dict[str, Any]:
    return {
        TASK_INDEX_KEY: materialized_row[TASK_INDEX_KEY],
        ROLLOUT_INDEX_KEY: materialized_row[ROLLOUT_INDEX_KEY],
        **result,
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": float(mean(values)),
        "max": float(max(values)),
        "min": float(min(values)),
        "median": float(median(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row.get(TASK_INDEX_KEY, 0))].append(row)

    scalar_names = set()
    for row in rows:
        for name, value in row.items():
            if name in {TASK_INDEX_KEY, ROLLOUT_INDEX_KEY, "epoch"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scalar_names.add(name)

    agent_metrics = {}
    for name in sorted(scalar_names):
        stats = numeric_summary([float(row[name]) for row in rows if isinstance(row.get(name), (int, float))])
        for stat_name, value in stats.items():
            agent_metrics[f"{stat_name}/{name}"] = value

    group_level_metrics = []
    for task_idx in sorted(by_task):
        task_rows = by_task[task_idx]
        group = {TASK_INDEX_KEY: task_idx, "num_rollouts": len(task_rows)}
        for name in sorted(scalar_names):
            values = [float(row[name]) for row in task_rows if isinstance(row.get(name), (int, float))]
            if values:
                stats = numeric_summary(values)
                group[f"mean/{name}"] = stats["mean"]
                group[f"max/{name}"] = stats["max"]
                group[f"min/{name}"] = stats["min"]
        group_level_metrics.append(group)

    key_metrics = {name: value for name, value in agent_metrics.items() if name.startswith("mean/")}
    return {
        "agent_metrics": agent_metrics,
        "key_metrics": key_metrics,
        "group_level_metrics": group_level_metrics,
    }


def write_aggregate_metrics(output_path: str | Path, rows: list[dict[str, Any]]) -> Path:
    metrics_path = aggregate_metrics_path(output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(aggregate_rollouts(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics_path
