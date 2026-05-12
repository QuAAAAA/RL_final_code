from __future__ import annotations

from math import log
from typing import List

import torch

from repo.Local_Emphasis import LocalEmphasisReward


class FixedFeatureReward(LocalEmphasisReward):
    def __init__(
        self,
        frame_pitch: List[float],
        frame_energy: List[float],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.frame_pitch = frame_pitch
        self.frame_energy = frame_energy

    def _extract_f0(self, wav: torch.Tensor, sr: int) -> List[float]:
        return self.frame_pitch

    def _extract_energy(self, wav: torch.Tensor) -> List[float]:
        if self.config.use_log_energy:
            return [float(log(value + self.config.eps)) for value in self.frame_energy]
        return self.frame_energy


def compute_with_features(
    frame_pitch: List[float],
    frame_energy: List[float],
    target_idx: int,
    **kwargs,
):
    sr = 1
    hop_length = 1
    wav = torch.zeros(len(frame_pitch))
    boundaries = [
        (f"w{i}", float(i), float(i + 1))
        for i in range(len(frame_pitch))
    ]
    reward = FixedFeatureReward(
        frame_pitch,
        frame_energy,
        hop_length=hop_length,
        n_fft=2,
        **kwargs,
    )
    return reward(wav, sr, boundaries, target_idx)


def compute_with_boundaries(
    frame_pitch: List[float],
    frame_energy: List[float],
    boundaries,
    target_idx: int,
    **kwargs,
):
    sr = 1
    wav = torch.zeros(max(len(frame_pitch), 1))
    reward = FixedFeatureReward(
        frame_pitch,
        frame_energy,
        hop_length=1,
        n_fft=2,
        **kwargs,
    )
    return reward(wav, sr, boundaries, target_idx)


def test_target_highest_pitch_and_energy_gets_hard_rewards() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 300.0, 200.0],
        frame_energy=[1.0, 10.0, 2.0],
        target_idx=1,
        preset="rl_robust",
    )

    assert result["pitch_hard"] == 1.0
    assert result["energy_hard"] == 1.0


def test_target_with_no_voiced_frames_masks_pitch_rewards() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 0.0, 200.0],
        frame_energy=[1.0, 10.0, 2.0],
        target_idx=1,
        preset="rl_robust",
        min_voiced_ratio=0.2,
    )

    assert result["target_voiced_ratio"] == 0.0
    assert result["pitch_valid"] is False
    assert result["pitch_soft"] == 0.0
    assert result["pitch_hard"] == 0.0
    assert result["pitch_rank_soft"] == 0.0
    assert result["pitch_confidence"] == 0.0


def test_zero_voiced_target_downweights_energy_soft_and_rank() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 200.0, 0.0],
        frame_energy=[10.0, 10.0, 1.0],
        target_idx=2,
        preset="rl_robust",
        min_voiced_ratio=0.2,
        use_log_energy=False,
    )

    assert result["energy_confidence"] == 0.5
    assert result["energy_soft"] == -0.5
    assert 0.0 < result["energy_rank_soft"] < 0.25


def test_pitch_rewards_are_weighted_by_voiced_ratio_confidence() -> None:
    result = compute_with_boundaries(
        frame_pitch=[100.0, 0.0, 300.0, 0.0, 120.0, 120.0],
        frame_energy=[1.0, 1.0, 10.0, 10.0, 2.0, 2.0],
        boundaries=[
            ("w0", 0.0, 2.0),
            ("w1", 2.0, 4.0),
            ("w2", 4.0, 6.0),
        ],
        target_idx=1,
        preset="rl_robust",
        min_voiced_ratio=0.2,
    )

    assert result["target_voiced_ratio"] == 0.5
    assert result["pitch_confidence"] == 0.5
    assert result["pitch_hard"] == 0.5
    assert 0.0 < result["pitch_rank_soft"] < 0.5


def test_zscore_mode_is_safe_when_variance_is_tiny() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 100.0, 100.0],
        frame_energy=[1e-12, 0.0, -1e-12],
        target_idx=1,
        preset="rl_robust",
        use_log_energy=False,
        soft_mode="zscore",
    )

    assert result["energy_soft"] == 0.0
    assert result["energy_valid"] is False
    assert result["normalized_word_energy"] == [0.0, 0.0, 0.0]


def test_log_energy_compresses_raw_stft_scale() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 200.0, 300.0],
        frame_energy=[1.0, 1_000_000.0, 2_000_000.0],
        target_idx=2,
        preset="rl_robust",
        use_log_energy=True,
    )

    assert result["energy_mode"] == "log"
    assert max(result["word_energy"]) < 20.0


def test_ranking_uses_normalized_features_not_raw_energy_difference() -> None:
    result = compute_with_features(
        frame_pitch=[100.0, 200.0, 300.0],
        frame_energy=[1.0, 1_000_000.0, 2_000_000.0],
        target_idx=2,
        preset="rl_robust",
        use_log_energy=False,
    )

    assert result["energy_rank_soft"] < 0.9
    assert max(result["normalized_word_energy"]) < 2.0
