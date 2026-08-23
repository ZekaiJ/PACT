"""Reference fusion rules used in the controlled comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import CONTRACT_CLASSES, normalize_distribution, top_label


@dataclass(frozen=True)
class BaselinePrediction:
    predicted_contract: str
    probabilities: dict[str, float]
    confidence: float


def _available_sources(
    record: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    sources = record.get("sources", {})
    if not isinstance(sources, Mapping):
        return []
    return [
        (name, source)
        for name, source in sources.items()
        if isinstance(source, Mapping) and not bool(source.get("missing", False))
    ]


def _prediction(probabilities: Mapping[str, float]) -> BaselinePrediction:
    distribution = normalize_distribution(probabilities)
    label, confidence, _ = top_label(distribution)
    return BaselinePrediction(label, distribution, confidence)


def quality_weighted_prediction(
    record: Mapping[str, Any],
) -> BaselinePrediction:
    """Average source distributions in proportion to source quality."""

    sources = _available_sources(record)
    scores = {label: 0.0 for label in CONTRACT_CLASSES}
    total = 0.0
    for _, source in sources:
        weight = min(max(float(source.get("quality", 0.0)), 0.0), 1.0)
        distribution = normalize_distribution(source.get("probabilities", {}))
        total += weight
        for label in CONTRACT_CLASSES:
            scores[label] += weight * distribution[label]
    if total <= 0.0:
        return _prediction({})
    return _prediction({label: value / total for label, value in scores.items()})


def product_evidence_prediction(
    record: Mapping[str, Any],
) -> BaselinePrediction:
    """Multiply reliability-discounted source distributions."""

    scores = {label: 1.0 for label in CONTRACT_CLASSES}
    uniform = 1.0 / len(CONTRACT_CLASSES)
    for _, source in _available_sources(record):
        distribution = normalize_distribution(source.get("probabilities", {}))
        quality = min(max(float(source.get("quality", 0.0)), 0.0), 1.0)
        conflict = min(max(float(source.get("conflict", 0.0)), 0.0), 1.0)
        reliability = quality * (1.0 - conflict)
        for label in CONTRACT_CLASSES:
            scores[label] *= (
                reliability * distribution[label]
                + (1.0 - reliability) * uniform
            )
    return _prediction(scores)


_CLASS_COUNT = len(CONTRACT_CLASSES)
_FRAME = (1 << _CLASS_COUNT) - 1
_EPSILON = 1e-12
_IGNORANCE_FLOOR = 0.01


def _supersets(mask: int):
    return (
        other
        for other in range(1 << _CLASS_COUNT)
        if other & mask == mask
    )


def _commonality(mass: Mapping[int, float]) -> dict[int, float]:
    return {
        mask: sum(mass.get(other, 0.0) for other in _supersets(mask))
        for mask in range(1 << _CLASS_COUNT)
    }


def _conjunctive_log_weights(mass: Mapping[int, float]) -> dict[int, float]:
    commonality = _commonality(mass)
    weights: dict[int, float] = {}
    for mask in range(_FRAME):
        value = 0.0
        for other in _supersets(mask):
            odd = (other.bit_count() - mask.bit_count() + 1) % 2
            exponent = -1.0 if odd else 1.0
            value += exponent * math.log(max(commonality[other], _EPSILON))
        weights[mask] = value
    return weights


def _mass_from_log_weights(log_weights: Mapping[int, float]) -> dict[int, float]:
    commonality = {
        mask: math.exp(
            sum(
                value
                for focal, value in log_weights.items()
                if mask & focal != mask
            )
        )
        for mask in range(1 << _CLASS_COUNT)
    }
    mass: dict[int, float] = {}
    for mask in range(1 << _CLASS_COUNT):
        value = 0.0
        for other in _supersets(mask):
            odd = (other.bit_count() - mask.bit_count()) % 2
            value += (-1.0 if odd else 1.0) * commonality[other]
        if -1e-8 < value < 0.0:
            value = 0.0
        if value < -1e-8:
            raise ValueError("Cautious combination produced a negative mass")
        mass[mask] = value
    if not math.isclose(sum(mass.values()), 1.0, abs_tol=1e-7):
        raise ValueError("Cautious masses do not sum to one")
    return mass


def _discounted_mass(
    probabilities: Mapping[str, float],
    quality: float,
    conflict: float,
) -> dict[int, float]:
    distribution = normalize_distribution(probabilities)
    reliability = max(
        _EPSILON,
        min(1.0 - _IGNORANCE_FLOOR, quality * (1.0 - conflict)),
    )
    mass = {mask: 0.0 for mask in range(1 << _CLASS_COUNT)}
    for index, label in enumerate(CONTRACT_CLASSES):
        mass[1 << index] = reliability * distribution[label]
    mass[_FRAME] = 1.0 - reliability
    return mass


def _cautious_combine(
    masses: list[Mapping[int, float]],
) -> dict[int, float]:
    if not masses:
        return {
            mask: 1.0 if mask == _FRAME else 0.0
            for mask in range(1 << _CLASS_COUNT)
        }
    log_weights = [_conjunctive_log_weights(mass) for mass in masses]
    combined = {
        mask: min(weights[mask] for weights in log_weights)
        for mask in range(_FRAME)
    }
    unnormalized = _mass_from_log_weights(combined)
    conflict = unnormalized[0]
    denominator = 1.0 - conflict
    if denominator <= _EPSILON:
        raise ValueError("Cautious combination has total conflict")
    return {
        mask: 0.0 if mask == 0 else value / denominator
        for mask, value in unnormalized.items()
    }


def _pignistic(mass: Mapping[int, float]) -> dict[str, float]:
    values = [0.0] * _CLASS_COUNT
    for mask, value in mass.items():
        if mask == 0 or value == 0.0:
            continue
        size = mask.bit_count()
        for index in range(_CLASS_COUNT):
            if mask & (1 << index):
                values[index] += value / size
    return normalize_distribution(
        {
            label: values[index]
            for index, label in enumerate(CONTRACT_CLASSES)
        }
    )


def cautious_evidence_prediction(
    record: Mapping[str, Any],
) -> BaselinePrediction:
    """Apply the normalized cautious conjunctive rule."""

    masses = []
    for _, source in _available_sources(record):
        masses.append(
            _discounted_mass(
                source.get("probabilities", {}),
                float(source.get("quality", 0.0)),
                float(source.get("conflict", 0.0)),
            )
        )
    return _prediction(_pignistic(_cautious_combine(masses)))
