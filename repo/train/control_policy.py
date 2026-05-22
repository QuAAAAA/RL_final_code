from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .dataset import EMOTION_LABELS


@dataclass(frozen=True)
class ControlAction:
    emotion_index: int
    valence_index: int
    arousal_index: int
    emotion: str
    valence: float
    arousal: float


class HashTextEncoder:
    def __init__(self, vocab_size: int = 4096) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size

    def encode(self, texts: Iterable[str]):
        import torch

        vectors = []
        for text in texts:
            vector = torch.zeros(self.vocab_size, dtype=torch.float32)
            tokens = self._tokens(text)
            if not tokens:
                tokens = ["<empty>"]
            scale = 1.0 / max(1, len(tokens))
            for token in tokens:
                vector[int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.vocab_size] += scale
            vectors.append(vector)
        return torch.stack(vectors, dim=0)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.strip().lower()
        if not normalized:
            return []
        pieces = normalized.split()
        if len(pieces) > 1:
            return pieces
        return list(normalized)


class TinyControlPolicy:
    def __init__(
        self,
        feature_dim: int = 4096,
        hidden_dim: int = 128,
        va_bins: int = 9,
    ) -> None:
        import torch

        if va_bins < 2:
            raise ValueError("va_bins must be at least 2")
        self.va_bins = va_bins
        self.emotion_labels = list(EMOTION_LABELS)
        self.module = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.emotion_head = torch.nn.Linear(hidden_dim, len(self.emotion_labels))
        self.valence_head = torch.nn.Linear(hidden_dim, va_bins)
        self.arousal_head = torch.nn.Linear(hidden_dim, va_bins)

    def parameters(self):
        yield from self.module.parameters()
        yield from self.emotion_head.parameters()
        yield from self.valence_head.parameters()
        yield from self.arousal_head.parameters()

    def train(self) -> None:
        self.module.train()
        self.emotion_head.train()
        self.valence_head.train()
        self.arousal_head.train()

    def eval(self) -> None:
        self.module.eval()
        self.emotion_head.eval()
        self.valence_head.eval()
        self.arousal_head.eval()

    def state_dict(self) -> dict:
        return {
            "module": self.module.state_dict(),
            "emotion_head": self.emotion_head.state_dict(),
            "valence_head": self.valence_head.state_dict(),
            "arousal_head": self.arousal_head.state_dict(),
            "va_bins": self.va_bins,
            "emotion_labels": self.emotion_labels,
        }

    def logits(self, features):
        hidden = self.module(features)
        return {
            "emotion": self.emotion_head(hidden),
            "valence": self.valence_head(hidden),
            "arousal": self.arousal_head(hidden),
        }

    def sample_group(self, feature, group_size: int):
        import torch

        if group_size <= 0:
            raise ValueError("group_size must be positive")
        features = feature.unsqueeze(0).expand(group_size, -1)
        logits = self.logits(features)
        emotion_dist = torch.distributions.Categorical(logits=logits["emotion"])
        valence_dist = torch.distributions.Categorical(logits=logits["valence"])
        arousal_dist = torch.distributions.Categorical(logits=logits["arousal"])
        emotion_idx = emotion_dist.sample()
        valence_idx = valence_dist.sample()
        arousal_idx = arousal_dist.sample()

        logprobs = torch.stack(
            [
                emotion_dist.log_prob(emotion_idx),
                valence_dist.log_prob(valence_idx),
                arousal_dist.log_prob(arousal_idx),
            ],
            dim=1,
        )
        actions = [
            self.action_from_indices(int(e), int(v), int(a))
            for e, v, a in zip(emotion_idx, valence_idx, arousal_idx)
        ]
        indices = {
            "emotion": emotion_idx,
            "valence": valence_idx,
            "arousal": arousal_idx,
        }
        return actions, indices, logprobs.detach()

    def action_logprobs(self, features, indices: dict):
        import torch

        logits = self.logits(features)
        emotion_dist = torch.distributions.Categorical(logits=logits["emotion"])
        valence_dist = torch.distributions.Categorical(logits=logits["valence"])
        arousal_dist = torch.distributions.Categorical(logits=logits["arousal"])
        return torch.stack(
            [
                emotion_dist.log_prob(indices["emotion"]),
                valence_dist.log_prob(indices["valence"]),
                arousal_dist.log_prob(indices["arousal"]),
            ],
            dim=1,
        )

    def action_from_indices(self, emotion_index: int, valence_index: int, arousal_index: int) -> ControlAction:
        return ControlAction(
            emotion_index=emotion_index,
            valence_index=valence_index,
            arousal_index=arousal_index,
            emotion=self.emotion_labels[emotion_index],
            valence=self._bin_to_unit(valence_index),
            arousal=self._bin_to_unit(arousal_index),
        )

    def greedy_action(self, feature) -> ControlAction:
        import torch

        with torch.no_grad():
            logits = self.logits(feature.unsqueeze(0))
            return self.action_from_indices(
                int(torch.argmax(logits["emotion"], dim=-1).item()),
                int(torch.argmax(logits["valence"], dim=-1).item()),
                int(torch.argmax(logits["arousal"], dim=-1).item()),
            )

    def _bin_to_unit(self, index: int) -> float:
        return float(index) / float(self.va_bins - 1)
