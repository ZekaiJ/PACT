"""TMC-style Dirichlet composition on a decision-level opinion interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import CONTRACT_CLASSES, PREDICTIVE_SOURCES, normalize_distribution
from .lineage import connected_components


@dataclass(frozen=True)
class DirichletSourceInput:
    """Fusion-only fields exposed by one opinion source."""

    probabilities: Mapping[str, float] | None
    quality: float
    conflict: float
    missing: bool


@dataclass(frozen=True)
class DirichletInput:
    """Restricted predictive input; excludes lineage and verifier-only fields."""

    sources: Mapping[str, DirichletSourceInput]


@dataclass(frozen=True)
class EvidentialPrediction:
    predicted_contract: str
    probabilities: dict[str, float]
    confidence: float
    selection_score: float
    uncertainty: float
    eligible: bool
    opinion_count: int
    lineage_group_count: int | None = None


Opinion = tuple[list[float], float]


def _source_payload(record: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    sources = record.get("sources", {})
    if not isinstance(sources, Mapping):
        return {}
    payload = sources.get(source, {})
    return payload if isinstance(payload, Mapping) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def restrict_input(
    record: Mapping[str, Any],
    *,
    sources: Sequence[str] = PREDICTIVE_SOURCES,
) -> DirichletInput:
    """Project a complete source record onto the fusion-only Dirichlet input."""

    restricted: dict[str, DirichletSourceInput] = {}
    for source in sources:
        payload = _source_payload(record, source)
        probabilities = payload.get("probabilities")
        restricted[source] = DirichletSourceInput(
            probabilities=(
                dict(probabilities) if isinstance(probabilities, Mapping) else None
            ),
            quality=_to_float(payload.get("quality")),
            conflict=_to_float(payload.get("conflict")),
            missing=not payload or bool(payload.get("missing", False)),
        )
    return DirichletInput(sources=restricted)


def _evidence(
    record: DirichletInput,
    source: str,
    concentration: float,
) -> list[float] | None:
    payload = record.sources.get(source)
    if payload is None or payload.missing or payload.probabilities is None:
        return None
    reliability = min(max(payload.quality, 0.0), 1.0)
    reliability *= 1.0 - min(max(payload.conflict, 0.0), 1.0)
    distribution = normalize_distribution(payload.probabilities)
    return [
        concentration * reliability * distribution[label]
        for label in CONTRACT_CLASSES
    ]


def _opinion_from_evidence(evidence: Sequence[float]) -> Opinion:
    strength = sum(evidence) + len(CONTRACT_CLASSES)
    return (
        [value / strength for value in evidence],
        len(CONTRACT_CLASSES) / strength,
    )


def _combine(left: Opinion, right: Opinion) -> Opinion:
    belief_left, uncertainty_left = left
    belief_right, uncertainty_right = right
    conflict = max(
        sum(belief_left) * sum(belief_right)
        - sum(x * y for x, y in zip(belief_left, belief_right)),
        0.0,
    )
    denominator = max(1.0 - conflict, 1e-12)
    belief = [
        (
            belief_left[index] * belief_right[index]
            + belief_left[index] * uncertainty_right
            + belief_right[index] * uncertainty_left
        )
        / denominator
        for index in range(len(CONTRACT_CLASSES))
    ]
    return belief, (uncertainty_left * uncertainty_right) / denominator


def _prediction(
    opinions: Sequence[Opinion],
    *,
    lineage_group_count: int | None = None,
) -> EvidentialPrediction:
    if len(opinions) < 2:
        uniform = 1.0 / len(CONTRACT_CLASSES)
        return EvidentialPrediction(
            predicted_contract=CONTRACT_CLASSES[0],
            probabilities={label: uniform for label in CONTRACT_CLASSES},
            confidence=uniform,
            selection_score=0.0,
            uncertainty=1.0,
            eligible=False,
            opinion_count=len(opinions),
            lineage_group_count=lineage_group_count,
        )

    combined = opinions[0]
    for opinion in opinions[1:]:
        combined = _combine(combined, opinion)
    belief, uncertainty = combined
    probabilities = {
        label: belief[index] + uncertainty / len(CONTRACT_CLASSES)
        for index, label in enumerate(CONTRACT_CLASSES)
    }
    probabilities = normalize_distribution(probabilities)
    predicted_contract = max(probabilities, key=probabilities.get)
    return EvidentialPrediction(
        predicted_contract=predicted_contract,
        probabilities=probabilities,
        confidence=probabilities[predicted_contract],
        selection_score=1.0 - uncertainty,
        uncertainty=uncertainty,
        eligible=True,
        opinion_count=len(opinions),
        lineage_group_count=lineage_group_count,
    )


def predict(
    record: DirichletInput,
    *,
    concentration: float = 24.0,
    sources: Sequence[str] = PREDICTIVE_SOURCES,
) -> EvidentialPrediction:
    """Compose available opinions from the restricted fusion-only input."""

    if not isinstance(record, DirichletInput):
        raise TypeError("predict() requires restrict_input(record)")
    opinions = []
    for source in sources:
        evidence = _evidence(record, source, concentration)
        if evidence is not None:
            opinions.append(_opinion_from_evidence(evidence))
    return _prediction(opinions)


def predict_grouped(
    record: DirichletInput,
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    concentration: float = 24.0,
    lineage_exponent: float = 1.0,
    sources: Sequence[str] = PREDICTIVE_SOURCES,
) -> EvidentialPrediction:
    """Compose restricted opinions after capping registered-group evidence mass."""

    if not isinstance(record, DirichletInput):
        raise TypeError("predict_grouped() requires restrict_input(record)")
    source_evidence = {
        source: evidence
        for source in sources
        if (evidence := _evidence(record, source, concentration)) is not None
    }
    groups = connected_components(tuple(source_evidence), lineage_graph)
    opinions = []
    for group in groups:
        divisor = len(group) ** lineage_exponent
        evidence = [
            sum(source_evidence[source][index] for source in group) / divisor
            for index in range(len(CONTRACT_CLASSES))
        ]
        opinions.append(_opinion_from_evidence(evidence))
    return _prediction(opinions, lineage_group_count=len(groups))

