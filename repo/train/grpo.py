from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GrpoCandidate:
    group_id: str
    candidate_index: int
    reward: float
    advantage: float
    payload: dict


def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance ** 0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (std + eps) for reward in rewards]


def grpo_loss(
    policy_logprobs,
    old_logprobs,
    advantages,
    response_mask=None,
    ref_logprobs=None,
    beta: float = 0.0,
    clip_eps: float = 0.2,
):
    """Torch GRPO clipped surrogate loss.

    Shapes are expected to be [batch, tokens]. The caller owns the policy env,
    so torch is imported lazily here. `advantages` can be [batch] or
    [batch, 1].
    """
    import torch

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(policy_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss_terms = -torch.minimum(unclipped, clipped)

    if beta and ref_logprobs is not None:
        kl = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1.0
        loss_terms = loss_terms + beta * kl

    if response_mask is None:
        return loss_terms.mean()
    denom = response_mask.sum().clamp_min(1)
    return (loss_terms * response_mask).sum() / denom
