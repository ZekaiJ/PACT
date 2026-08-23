"""Shared action-contract vocabulary and probability utilities."""

from __future__ import annotations

from typing import Mapping, Sequence


CONTRACT_CLASSES = (
    "normal",
    "slow_clearance",
    "hold_confirm",
    "retreat_fallback",
    "bounded_urgent",
)

PREDICTIVE_SOURCES = ("language", "vision", "geometry", "risk")
OBSERVABLE_SOURCES = (
    "language",
    "vision",
    "geometry",
    "risk",
    "semantic",
    "depth",
)
POST_EXECUTION_FIELDS = (
    "trace",
    "validator",
    "telemetry",
    "execution_log",
    "post_execution_telemetry",
)


def normalize_distribution(probabilities: Mapping[str, float]) -> dict[str, float]:
    values = {
        label: max(float(probabilities.get(label, 0.0)), 0.0)
        for label in CONTRACT_CLASSES
    }
    total = sum(values.values())
    if total <= 0.0:
        return {label: 1.0 / len(CONTRACT_CLASSES) for label in CONTRACT_CLASSES}
    return {label: value / total for label, value in values.items()}


def top_label(probabilities: Mapping[str, float]) -> tuple[str, float, float]:
    ordered = sorted(
        normalize_distribution(probabilities).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return ordered[0][0], ordered[0][1], ordered[0][1] - ordered[1][1]


def assert_pre_action_sources(source_names: Sequence[str]) -> None:
    blocked = sorted(set(source_names) & set(POST_EXECUTION_FIELDS))
    if blocked:
        raise ValueError(
            "Post-execution evidence cannot enter pre-action inference: "
            + ", ".join(blocked)
        )
    unknown = sorted(set(source_names) - set(OBSERVABLE_SOURCES))
    if unknown:
        raise ValueError("Unknown pre-action source(s): " + ", ".join(unknown))

