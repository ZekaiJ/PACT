"""Controlled five-class hierarchy-matched cautious--cumulative fusion.

Each source is mapped to its reliability-discounted singleton/frame mass and
then to canonical conjunctive log weights. We take the coordinatewise minimum
within each registered parent component, add weights across distinct
components, reconstruct the mass, and normalize once. The posterior is the
pignistic projection of that final mass. The native selective score is
``1 - m(frame)``; it is intentionally separate from posterior confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .baselines import (
    _EPSILON,
    _FRAME,
    _conjunctive_log_weights,
    _discounted_mass,
    _mass_from_log_weights,
    _pignistic,
)
from .contracts import CONTRACT_CLASSES
from .pcecf import SourceEvidence, registered_components


@dataclass(frozen=True)
class HierarchyMatchedCautiousOutput:
    """Posterior and native score of the controlled cautious comparator."""

    predicted_contract: str
    probabilities: dict[str, float]
    selection_score: float
    frame_mass: float
    pre_normalization_conflict: float
    normalization_denominator: float
    registered_components: tuple[tuple[str, ...], ...]


def _vacuous_mass() -> dict[int, float]:
    return {
        mask: 1.0 if mask == _FRAME else 0.0
        for mask in range(_FRAME + 1)
    }


def _source_mass(source: SourceEvidence) -> dict[int, float]:
    if source.missing:
        return _vacuous_mass()
    probabilities = {
        label: float(source.probabilities[index])
        for index, label in enumerate(CONTRACT_CLASSES)
    }
    return _discounted_mass(
        probabilities,
        float(source.quality),
        float(source.conflict),
    )


def hierarchy_matched_cautious(
    sources: Sequence[SourceEvidence],
) -> HierarchyMatchedCautiousOutput:
    """Fuse a fixed five-class source catalog using registered parent components."""

    components = registered_components(sources)
    log_weights = [
        _conjunctive_log_weights(_source_mass(source))
        for source in sources
    ]
    component_weights = [
        {
            focal: min(log_weights[index][focal] for index in component)
            for focal in range(_FRAME)
        }
        for component in components
    ]
    combined_weights = {
        focal: sum(weights[focal] for weights in component_weights)
        for focal in range(_FRAME)
    }
    unnormalized = _mass_from_log_weights(combined_weights)
    conflict = float(unnormalized[0])
    denominator = 1.0 - conflict
    if denominator <= _EPSILON:
        raise ValueError("hierarchy-matched cautious rule has total conflict")
    mass = {
        focal: 0.0 if focal == 0 else float(value) / denominator
        for focal, value in unnormalized.items()
    }
    probabilities = _pignistic(mass)
    values = np.asarray(
        [float(probabilities[label]) for label in CONTRACT_CLASSES],
        dtype=np.float64,
    )
    predicted_index = int(np.argmax(values))
    return HierarchyMatchedCautiousOutput(
        predicted_contract=CONTRACT_CLASSES[predicted_index],
        probabilities={
            label: float(values[index])
            for index, label in enumerate(CONTRACT_CLASSES)
        },
        selection_score=1.0 - float(mass[_FRAME]),
        frame_mass=float(mass[_FRAME]),
        pre_normalization_conflict=conflict,
        normalization_denominator=denominator,
        registered_components=tuple(
            tuple(sources[index].source_id for index in component)
            for component in components
        ),
    )
