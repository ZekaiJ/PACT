"""Registered-lineage relations and the log-linear exact-copy analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    CONTRACT_CLASSES,
    assert_pre_action_sources,
    normalize_distribution,
    top_label,
)


@dataclass(frozen=True)
class SourceOpinion:
    source: str
    probabilities: Mapping[str, float]
    quality: float
    conflict: float = 0.0
    missing: bool = False


def graph_from_parent_sets(
    source_parents: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str], float]:
    """Connect opinions that share at least one registered parent."""

    names = sorted(source_parents)
    parents = {name: set(map(str, source_parents[name])) for name in names}
    return {
        (left, right): 1.0
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if parents[left] & parents[right]
    }


def connected_components(
    nodes: Sequence[str],
    lineage_graph: Mapping[tuple[str, str], float],
) -> list[list[str]]:
    remaining = set(nodes)
    components: list[list[str]] = []
    while remaining:
        stack = [remaining.pop()]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            linked = {
                other
                for other in remaining
                if float(
                    lineage_graph.get(
                        (current, other),
                        lineage_graph.get((other, current), 0.0),
                    )
                )
                > 0.0
            }
            remaining.difference_update(linked)
            stack.extend(linked)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def registered_component_count(
    supporters: Sequence[str],
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    partition_nodes: Sequence[str] | None = None,
) -> int:
    supporting = set(supporters)
    nodes = tuple(partition_nodes) if partition_nodes is not None else tuple(supporters)
    return sum(
        bool(supporting.intersection(component))
        for component in connected_components(nodes, lineage_graph)
    )


def registered_lineage_discounts(
    opinions: Sequence[SourceOpinion],
    lineage_graph: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    """Discount connected same-class opinions by their registered group size."""

    names = [opinion.source for opinion in opinions]
    if len(names) != len(set(names)):
        raise ValueError("Source names must be unique.")
    assert_pre_action_sources(names)
    labels = {
        opinion.source: top_label(opinion.probabilities)[0]
        for opinion in opinions
        if not opinion.missing
    }
    discounts: dict[str, float] = {}
    for opinion in opinions:
        if opinion.missing:
            discounts[opinion.source] = 1.0
            continue
        shared = 0.0
        for other in opinions:
            if other.source == opinion.source or other.missing:
                continue
            edge = lineage_graph.get(
                (opinion.source, other.source),
                lineage_graph.get((other.source, opinion.source), 0.0),
            )
            if labels[opinion.source] == labels[other.source]:
                shared += min(max(float(edge), 0.0), 1.0)
        discounts[opinion.source] = 1.0 / (1.0 + shared)
    return discounts


def source_weight_logits(
    opinions: Sequence[SourceOpinion],
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    quality_gain: float = 2.0,
    conflict_penalty: float = 2.0,
    missing_penalty: float = 4.0,
    lineage_exponent: float = 1.0,
) -> dict[str, float]:
    discounts = registered_lineage_discounts(opinions, lineage_graph)
    logits: dict[str, float] = {}
    for opinion in opinions:
        quality = min(max(float(opinion.quality), 0.0), 1.0)
        conflict = min(max(float(opinion.conflict), 0.0), 1.0)
        missing = 1.0 if opinion.missing else 0.0
        logits[opinion.source] = (
            quality_gain * quality
            - conflict_penalty * conflict
            - missing_penalty * missing
            + lineage_exponent * math.log(discounts[opinion.source])
        )
    return logits


def unnormalized_group_mass(
    opinions: Sequence[SourceOpinion],
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    lineage_exponent: float = 1.0,
) -> float:
    logits = source_weight_logits(
        opinions,
        lineage_graph,
        lineage_exponent=lineage_exponent,
    )
    return sum(math.exp(value) for value in logits.values())


def log_linear_posterior(
    opinions: Sequence[SourceOpinion],
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    lineage_exponent: float = 1.0,
) -> dict[str, float]:
    """Compute the registered-lineage log-linear analytical posterior."""

    if not opinions:
        raise ValueError("At least one opinion is required.")
    logits = source_weight_logits(
        opinions,
        lineage_graph,
        lineage_exponent=lineage_exponent,
    )
    maximum = max(logits.values())
    weights = {source: math.exp(value - maximum) for source, value in logits.items()}
    weight_sum = sum(weights.values())
    weights = {source: value / weight_sum for source, value in weights.items()}
    class_scores = {label: 0.0 for label in CONTRACT_CLASSES}
    for opinion in opinions:
        distribution = normalize_distribution(opinion.probabilities)
        for label in CONTRACT_CLASSES:
            class_scores[label] += weights[opinion.source] * math.log(
                max(distribution[label], 1e-12)
            )
    score_maximum = max(class_scores.values())
    exponentials = {
        label: math.exp(value - score_maximum)
        for label, value in class_scores.items()
    }
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}

