from __future__ import annotations

from typing import Sequence


def _as_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def global_emotion_intensity_reward(
    v_y_hat,
    mu_neutral,
    target_range: Sequence[float],
    sigma: float = 0.1,
) -> dict:
    """Compute global emotion intensity reward.

    This reward measures only distance from neutral VAD space, not emotion
    direction or emotion-label correctness. Inputs may be torch tensors, numpy
    arrays, or Python lists. Batch shape is supported as [..., vad_dim].
    """
    import torch

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if len(target_range) != 2:
        raise ValueError("target_range must contain [low, high]")
    low = float(target_range[0])
    high = float(target_range[1])
    if high < low:
        raise ValueError("target_range high must be >= low")

    v = _as_tensor(v_y_hat)
    mu = _as_tensor(mu_neutral).to(device=v.device, dtype=v.dtype)
    if v.shape[-1] != mu.shape[-1]:
        raise ValueError("v_y_hat and mu_neutral must have the same last dimension")

    d_y_hat = torch.linalg.vector_norm(v - mu, ord=2, dim=-1)
    interval = ((d_y_hat >= low) & (d_y_hat <= high)).to(dtype=v.dtype)
    midpoint = (low + high) / 2.0
    gaussian = torch.exp(-((d_y_hat - midpoint) ** 2) / (2.0 * sigma * sigma))
    reward = interval + gaussian
    return {
        "reward": reward,
        "d_y_hat": d_y_hat,
        "R_interval": interval,
        "R_gaussian": gaussian,
    }


def global_emotion_intensity_reward_json(
    v_y_hat,
    mu_neutral,
    target_range: Sequence[float],
    sigma: float = 0.1,
) -> dict:
    result = global_emotion_intensity_reward(v_y_hat, mu_neutral, target_range, sigma)

    def item_or_list(tensor):
        values = tensor.detach().cpu()
        if values.ndim == 0:
            return float(values.item())
        return [float(x) for x in values.reshape(-1).tolist()]

    return {key: item_or_list(value) for key, value in result.items()}
