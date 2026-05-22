from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from repo.train.dataset import load_jsonl
from repo.train.gym_style import (
    ROLLOUT_INDEX_KEY,
    TASK_INDEX_KEY,
    aggregate_metrics_path,
    aggregate_rollouts,
    materialize_training_rows,
    materialized_inputs_path,
    read_jsonl,
    write_aggregate_metrics,
    write_jsonl,
)


class GymStyleTest(unittest.TestCase):
    def test_materialize_training_rows_repeats_with_stable_keys(self) -> None:
        examples = load_jsonl("data/va_train/zho_restaurant_train_alltasks_tagged.jsonl", limit=2)
        rows = materialize_training_rows(examples, repeats=2, base_seed=7)

        self.assertEqual(
            [(row[TASK_INDEX_KEY], row[ROLLOUT_INDEX_KEY], row["seed"]) for row in rows],
            [(0, 0, 7), (0, 1, 8), (1, 0, 1007), (1, 1, 1008)],
        )
        self.assertEqual(rows[0]["uid"], examples[0].uid)

    def test_jsonl_and_aggregate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "rollouts.jsonl"
            rows = [
                {TASK_INDEX_KEY: 0, ROLLOUT_INDEX_KEY: 0, "reward_mean": 1.0, "best_reward": 2.0},
                {TASK_INDEX_KEY: 0, ROLLOUT_INDEX_KEY: 1, "reward_mean": 3.0, "best_reward": 4.0},
            ]
            write_jsonl(output_path, rows)
            self.assertEqual(read_jsonl(output_path), rows)

            self.assertEqual(materialized_inputs_path(output_path).name, "rollouts_materialized_inputs.jsonl")
            self.assertEqual(aggregate_metrics_path(output_path).name, "rollouts_aggregate_metrics.json")

            metrics_path = write_aggregate_metrics(output_path, rows)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["agent_metrics"]["mean/reward_mean"], 2.0)
            self.assertEqual(metrics["agent_metrics"]["max/best_reward"], 4.0)
            self.assertEqual(metrics["group_level_metrics"][0]["num_rollouts"], 2)

    def test_aggregate_empty_rows(self) -> None:
        metrics = aggregate_rollouts([])
        self.assertEqual(metrics["agent_metrics"], {})
        self.assertEqual(metrics["group_level_metrics"], [])


if __name__ == "__main__":
    unittest.main()
