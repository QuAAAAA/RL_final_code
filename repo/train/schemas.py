from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    url: str | None = None
    mock: bool = False
    command: list[str] | None = None
    cwd: Path | None = None
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class TrainExample:
    uid: str
    text: str
    tagged_text: str
    quadruplets: list[JsonDict]


@dataclass(frozen=True)
class ControlPlan:
    text: str
    tagged_text: str
    target_emotion: str
    emotion_vector: list[float]
    va: tuple[float, float] | None
    va_01: tuple[float, float] | None
    emphasis_targets: list[JsonDict]
    quadruplets: list[JsonDict]


@dataclass(frozen=True)
class GenerationResult:
    uid: str
    wav_path: str | None
    control: ControlPlan
    rewards: JsonDict
    total_reward: float

