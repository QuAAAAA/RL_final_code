from __future__ import annotations

import math
import unittest

from repo.train.global_intensity_reward import global_emotion_intensity_reward_json


class GlobalEmotionIntensityRewardTest(unittest.TestCase):
    def test_center_reward_is_close_to_two(self):
        result = global_emotion_intensity_reward_json([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.6], sigma=0.1)
        self.assertTrue(math.isclose(result["d_y_hat"], 0.5, abs_tol=1e-6))
        self.assertTrue(math.isclose(result["reward"], 2.0, rel_tol=1e-6))

    def test_inside_interval_off_center_is_greater_than_one(self):
        result = global_emotion_intensity_reward_json([0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.6], sigma=0.1)
        self.assertEqual(result["R_interval"], 1.0)
        self.assertGreater(result["reward"], 1.0)
        self.assertLess(result["reward"], 2.0)

    def test_outside_interval_only_has_gaussian_reward(self):
        result = global_emotion_intensity_reward_json([0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.6], sigma=0.1)
        self.assertEqual(result["R_interval"], 0.0)
        self.assertTrue(math.isclose(result["reward"], result["R_gaussian"], rel_tol=1e-6))

    def test_non_positive_sigma_raises(self):
        with self.assertRaises(ValueError):
            global_emotion_intensity_reward_json([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.6], sigma=0.0)


if __name__ == "__main__":
    unittest.main()
