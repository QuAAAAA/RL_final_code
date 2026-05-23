from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

from .schemas import ControlPlan, JsonDict, TrainExample


EMOTION_LABELS = ["joy", "anger", "sadness", "fear", "disgust", "low_mood", "surprise", "neutral"]


def load_jsonl(path: str | Path, limit: int | None = None) -> list[TrainExample]:
    rows: list[TrainExample] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                TrainExample(
                    uid=str(item.get("ID") or f"line-{line_no}"),
                    text=str(item.get("Text") or ""),
                    tagged_text=str(item.get("TaggedText") or item.get("Text") or ""),
                    quadruplets=list(item.get("Quadruplet") or []),
                )
            )
    return rows


def parse_va(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    text = str(value)
    match = re.match(r"\s*([0-9.]+)\s*#\s*([0-9.]+)\s*$", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def mean_va(quadruplets: Iterable[JsonDict]) -> tuple[float, float] | None:
    vals = [va for va in (parse_va(q.get("VA")) for q in quadruplets) if va is not None]
    if not vals:
        return None
    return sum(v for v, _ in vals) / len(vals), sum(a for _, a in vals) / len(vals)


def va_to_unit(va: tuple[float, float] | None) -> tuple[float, float] | None:
    """Map dataset VA from 1~9 scale to model VA on 0~1 scale."""
    if va is None:
        return None
    valence, arousal = va
    return ((valence - 1.0) / 8.0, (arousal - 1.0) / 8.0)


def unit_to_dataset_va(va_01: tuple[float, float] | None) -> tuple[float, float] | None:
    if va_01 is None:
        return None
    valence, arousal = va_01
    return (valence * 8.0 + 1.0, arousal * 8.0 + 1.0)


def va_to_emotion(va: tuple[float, float] | None) -> str:
    if va is None:
        return "neutral"
    valence, arousal = va
    if valence >= 5.8 and arousal >= 5.8:
        return "joy"
    if valence <= 4.0 and arousal >= 5.4:
        return "anger"
    if valence <= 4.2 and arousal <= 4.8:
        return "sadness"
    if valence <= 4.8 and arousal >= 6.2:
        return "fear"
    if valence <= 4.5:
        return "disgust"
    if arousal >= 6.5:
        return "surprise"
    return "neutral"


def emotion_vector(label: str, strength: float = 1.4) -> list[float]:
    vector = [0.0] * len(EMOTION_LABELS)
    if label in EMOTION_LABELS and label != "neutral":
        vector[EMOTION_LABELS.index(label)] = strength
    return vector


def emphasis_targets(quadruplets: Iterable[JsonDict]) -> list[JsonDict]:
    targets: list[JsonDict] = []
    for item in quadruplets:
        opinion = str(item.get("Opinion") or "").strip()
        aspect = str(item.get("Aspect") or "").strip()
        if not opinion:
            continue
        va = parse_va(item.get("VA"))
        strength = "medium"
        if va is not None:
            valence, arousal = va
            if arousal >= 6.2 or abs(valence - 4.0) >= 2.0:
                strength = "strong"
            elif arousal <= 4.8:
                strength = "weak"
        targets.append({"word": opinion, "aspect": aspect, "strength": strength, "va": item.get("VA")})
    return targets


def build_random_control(example: TrainExample) -> ControlPlan:
    """Sample target emotion uniformly at random; VA values still provided for MLP."""
    va = mean_va(example.quadruplets)
    sampled_emotion = random.choice(EMOTION_LABELS)
    return ControlPlan(
        text=example.text,
        tagged_text=example.tagged_text,
        target_emotion=sampled_emotion,
        emotion_vector=emotion_vector(sampled_emotion),
        va=va,
        va_01=va_to_unit(va),
        emphasis_targets=emphasis_targets(example.quadruplets),
        quadruplets=example.quadruplets,
    )


def build_gold_control(example: TrainExample) -> ControlPlan:
    va = mean_va(example.quadruplets)
    emotion = va_to_emotion(va)
    return ControlPlan(
        text=example.text,
        tagged_text=example.tagged_text,
        target_emotion=emotion,
        emotion_vector=emotion_vector(emotion),
        va=va,
        va_01=va_to_unit(va),
        emphasis_targets=emphasis_targets(example.quadruplets),
        quadruplets=example.quadruplets,
    )

