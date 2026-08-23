"""Provenance-conditioned evidence-conserving fusion (PC-ECF).

The operator maps decision-level source opinions to registered lineage
components, conserves evidence within each component with a coordinatewise
lower envelope, and accumulates evidence across distinct components. Registered
parent relations are observable inputs; the operator does not authenticate them.
A parent denotes a directly reused source event or intermediate evidence object;
common physical origin alone is not sufficient to merge opinions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SourceEvidence:
    """One source opinion and the fields consumed by PC-ECF."""

    source_id: str
    probabilities: FloatArray
    quality: float
    conflict: float
    missing: bool
    parents: tuple[str, ...]
    valid: bool = True
    evidence: FloatArray | None = None


@dataclass(frozen=True)
class PCECFOutput:
    """Posterior and component-level outputs of PC-ECF."""

    posterior: FloatArray
    predicted_index: int
    confidence: float
    vacuity: float
    selection_score: float
    group_ids: tuple[tuple[str, ...], ...]
    group_evidence: FloatArray


def _probability_vector(
    values: Sequence[float], *, classes: int | None = None, allow_zero: bool = False
) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError("probabilities must be a one-dimensional vector with K >= 2")
    if classes is not None and vector.size != classes:
        raise ValueError(f"expected {classes} classes, found {vector.size}")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(vector.sum())
    if total <= 0.0:
        if allow_zero:
            return vector
        raise ValueError("available probabilities must have positive mass")
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("available probabilities must sum to one")
    return vector


def registered_components(
    sources: Sequence[SourceEvidence],
) -> tuple[tuple[int, ...], ...]:
    """Return connected components induced by overlap of registered parents."""

    if not sources:
        raise ValueError("at least one source is required")
    names = [source.source_id for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("source identifiers must be unique")
    parent_sets = [set(source.parents) for source in sources]
    empty = [names[index] for index, parents in enumerate(parent_sets) if not parents]
    if empty:
        raise ValueError(
            "registered parents are required for every source; "
            f"missing for {', '.join(empty)}"
        )

    remaining = set(range(len(sources)))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining, key=lambda index: names[index])
        remaining.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            linked = {
                other
                for other in remaining
                if parent_sets[current] & parent_sets[other]
            }
            remaining.difference_update(linked)
            stack.extend(sorted(linked, reverse=True))
        components.append(tuple(sorted(component, key=lambda index: names[index])))
    return tuple(sorted(components, key=lambda component: names[component[0]]))


def discounted_evidence(source: SourceEvidence, concentration: float) -> FloatArray:
    """Map a source distribution and reliability metadata to nonnegative evidence."""

    if concentration <= 0.0:
        raise ValueError("concentration must be positive")
    shaped = _probability_vector(
        source.probabilities,
        allow_zero=source.missing or not source.valid or source.evidence is not None,
    )
    if source.missing or not source.valid:
        return np.zeros_like(shaped)
    quality = float(np.clip(source.quality, 0.0, 1.0))
    conflict = float(np.clip(source.conflict, 0.0, 1.0))
    if source.evidence is not None:
        direct = np.asarray(source.evidence, dtype=np.float64)
        if direct.shape != shaped.shape:
            raise ValueError("direct evidence and probabilities must have the same shape")
        if not np.all(np.isfinite(direct)) or np.any(direct < 0.0):
            raise ValueError("direct evidence must be finite and nonnegative")
        return quality * (1.0 - conflict) * direct
    distribution = _probability_vector(source.probabilities)
    return concentration * quality * (1.0 - conflict) * distribution


def _group_lower_envelope(
    evidence: FloatArray, component: Sequence[int]
) -> FloatArray:
    """Return the multiplicity-invariant evidence budget of one component."""

    return np.min(evidence[np.asarray(component, dtype=int), :], axis=0)


def forward(
    sources: Sequence[SourceEvidence],
    *,
    concentration: float,
    prior_per_class: float = 1.0,
    expected_source_ids: Sequence[str] | None = None,
) -> PCECFOutput:
    """Apply PC-ECF to a fixed source catalog."""

    if not sources:
        raise ValueError("at least one instantiated source is required")
    if prior_per_class <= 0.0:
        raise ValueError("prior_per_class must be positive")
    source_ids = tuple(source.source_id for source in sources)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source identifiers must be unique")
    if expected_source_ids is not None:
        expected = tuple(expected_source_ids)
        if len(set(expected)) != len(expected):
            raise ValueError("expected source identifiers must be unique")
        if source_ids != expected:
            raise ValueError(
                f"instantiated catalog {source_ids!r} does not match expected {expected!r}"
            )

    class_count = _probability_vector(
        sources[0].probabilities,
        allow_zero=(
            sources[0].missing
            or not sources[0].valid
            or sources[0].evidence is not None
        ),
    ).size
    for source in sources:
        _probability_vector(
            source.probabilities,
            classes=class_count,
            allow_zero=source.missing or not source.valid or source.evidence is not None,
        )
    evidence = np.vstack(
        [discounted_evidence(source, concentration) for source in sources]
    )
    components = registered_components(sources)
    group_evidence = np.vstack(
        [_group_lower_envelope(evidence, component) for component in components]
    )
    fused = group_evidence.sum(axis=0)
    prior = np.full(class_count, prior_per_class, dtype=np.float64)
    posterior = (fused + prior) / float(fused.sum() + prior.sum())
    vacuity = float(prior.sum() / (fused.sum() + prior.sum()))
    predicted = int(np.argmax(posterior))
    group_ids = tuple(
        tuple(sources[index].source_id for index in component)
        for component in components
    )
    return PCECFOutput(
        posterior=posterior,
        predicted_index=predicted,
        confidence=float(posterior[predicted]),
        vacuity=vacuity,
        selection_score=1.0 - vacuity,
        group_ids=group_ids,
        group_evidence=group_evidence,
    )
