from __future__ import annotations

import unittest

from repo.train.control_grpo_train import score_control_action
from repo.train.control_policy import ControlAction, HashTextEncoder, TinyControlPolicy
from repo.train.dataset import build_gold_control, load_jsonl


class ControlGrpoTest(unittest.TestCase):
    def test_policy_samples_group_with_logprob_tokens(self) -> None:
        encoder = HashTextEncoder(vocab_size=32)
        policy = TinyControlPolicy(feature_dim=32, hidden_dim=8, va_bins=5)
        feature = encoder.encode(["餐點很好吃"])[0]

        actions, indices, old_logprobs = policy.sample_group(feature, group_size=3)
        new_logprobs = policy.action_logprobs(feature.unsqueeze(0).expand(3, -1), indices)

        self.assertEqual(len(actions), 3)
        self.assertEqual(tuple(old_logprobs.shape), (3, 3))
        self.assertEqual(tuple(new_logprobs.shape), (3, 3))

    def test_gold_like_action_scores_higher_than_bad_emotion_far_va(self) -> None:
        example = load_jsonl("data/va_train/zho_restaurant_train_alltasks_tagged.jsonl", limit=1)[0]
        gold = build_gold_control(example)
        target_valence, target_arousal = gold.va_01 or (0.5, 0.5)
        good = ControlAction(
            emotion_index=0,
            valence_index=0,
            arousal_index=0,
            emotion=gold.target_emotion,
            valence=target_valence,
            arousal=target_arousal,
        )
        bad = ControlAction(
            emotion_index=0,
            valence_index=0,
            arousal_index=0,
            emotion="neutral" if gold.target_emotion != "neutral" else "anger",
            valence=1.0 - target_valence,
            arousal=1.0 - target_arousal,
        )

        self.assertGreater(
            score_control_action(good, gold, {})["emotion"] + score_control_action(good, gold, {})["vad"],
            score_control_action(bad, gold, {})["emotion"] + score_control_action(bad, gold, {})["vad"],
        )


if __name__ == "__main__":
    unittest.main()
