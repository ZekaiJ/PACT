#!/usr/bin/env python3
"""Exhaustive PACT coarsening surface over six frozen HandWritten views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.topology_interventions import pact


EXPECTED_INPUT_SHA256 = (
    "2245CD2B272542F0AA1BD0E4DAFF94DC962FCBE8B6B620247D5828E83D8EAB98"
)
PAIR_SPECS = (
    ("HandWritten-Mfeat__TMC", "TMC", 200),
    ("HandWritten-Mfeat__RCML", "RCML", 400),
)
SUPPORT = (0.10, 0.90)
SCORES = ("native_nonvacuity", "posterior_confidence")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_rgs(labels: Iterable[int]) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    result = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        result.append(mapping[label])
    return tuple(result)


def set_partitions(size: int) -> list[tuple[int, ...]]:
    """Enumerate restricted-growth strings in deterministic order."""

    if size < 1:
        raise ValueError("size must be positive")
    output: list[tuple[int, ...]] = []

    def visit(prefix: list[int], current_max: int) -> None:
        if len(prefix) == size:
            output.append(tuple(prefix))
            return
        for label in range(current_max + 2):
            prefix.append(label)
            visit(prefix, max(current_max, label))
            prefix.pop()

    visit([0], 0)
    return output


def blocks(rgs: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(index for index, observed in enumerate(rgs) if observed == label)
        for label in range(max(rgs) + 1)
    )


def merge_blocks(rgs: tuple[int, ...], left: int, right: int) -> tuple[int, ...]:
    merged = [left if label == right else label for label in rgs]
    return canonical_rgs(merged)


def cover_edges(
    partitions: list[tuple[int, ...]],
) -> list[tuple[int, int, int, int]]:
    index = {partition: offset for offset, partition in enumerate(partitions)}
    edges = set()
    for fine_index, partition in enumerate(partitions):
        count = max(partition) + 1
        for left in range(count):
            for right in range(left + 1, count):
                coarse = merge_blocks(partition, left, right)
                edges.add((fine_index, index[coarse], left, right))
    return sorted(edges)


def load_pair(path: Path, backbone: str, expected_records: int) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["backbone"] == backbone and row["split"] == "test":
                rows.append(row)
    seeds = np.asarray(sorted({int(row["seed"]) for row in rows}), dtype=np.int64)
    record_ids = np.asarray(
        sorted({int(row["record_id"]) for row in rows}), dtype=np.int64
    )
    if len(seeds) != 5 or len(record_ids) != expected_records:
        raise AssertionError(
            f"{backbone}: expected 5 x {expected_records}, found "
            f"{len(seeds)} x {len(record_ids)}"
        )
    by_key = {(int(row["seed"]), int(row["record_id"])): row for row in rows}
    if len(by_key) != len(seeds) * len(record_ids):
        raise AssertionError("every frozen seed must cover every record identifier")
    labels = np.asarray(
        [int(by_key[(int(seeds[0]), int(record_id))]["y"]) for record_id in record_ids],
        dtype=np.int64,
    )
    evidence = np.asarray(
        [
            [by_key[(int(seed), int(record_id))]["evidences"] for record_id in record_ids]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    if evidence.ndim != 4 or evidence.shape[2] != 6:
        raise AssertionError(f"expected [seed, record, 6, class], found {evidence.shape}")
    for seed in seeds:
        observed = np.asarray(
            [int(by_key[(int(seed), int(record_id))]["y"]) for record_id in record_ids]
        )
        if not np.array_equal(observed, labels):
            raise AssertionError("labels differ across frozen seeds")
    return {
        "seeds": seeds,
        "record_ids": record_ids,
        "labels": labels,
        "evidence": evidence,
    }


def fuse_all(
    evidence: np.ndarray, partitions: list[tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """Apply PACT to every declared coarsening without changing source values."""

    seed_count, record_count, _, class_count = evidence.shape
    shape = (len(partitions), seed_count, record_count)
    posterior = np.empty(shape + (class_count,), dtype=np.float64)
    budget = np.empty(shape, dtype=np.float64)
    for partition_index, partition in enumerate(partitions):
        fused = np.zeros((seed_count, record_count, class_count), dtype=np.float64)
        for component in blocks(partition):
            fused += np.min(evidence[:, :, component, :], axis=2)
        total = np.sum(fused, axis=-1)
        budget[partition_index] = total
        posterior[partition_index] = (fused + 1.0) / (total[..., None] + class_count)
    confidence = np.max(posterior, axis=-1)
    nonvacuity = budget / (budget + class_count)
    return {
        "posterior": posterior,
        "budget": budget,
        "native_nonvacuity": nonvacuity,
        "posterior_confidence": confidence,
    }


def tie_groups(scores: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores) != 0.0) + 1
    return [part for part in np.split(order, boundaries) if len(part)]


def continuous_fractional_ncs(
    scores: np.ndarray,
    wrong: np.ndarray,
    weights: np.ndarray | None = None,
    low: float = SUPPORT[0],
    high: float = SUPPORT[1],
) -> np.ndarray:
    """Tie-aware risk integral with analytic fractional inclusion per score group."""

    if not 0.0 < low < high <= 1.0:
        raise ValueError("invalid support")
    groups = tie_groups(scores)
    if weights is None:
        weights = np.ones((1, len(scores)), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[1] != len(scores):
            raise ValueError("weights must have shape [draw, record]")
    total_records = np.sum(weights, axis=1)
    if not np.allclose(total_records, total_records[0]):
        raise AssertionError("every bootstrap draw must preserve the record count")
    lower = low * total_records
    upper = high * total_records
    accepted = np.zeros(len(weights), dtype=np.float64)
    errors = np.zeros(len(weights), dtype=np.float64)
    area = np.zeros(len(weights), dtype=np.float64)
    wrong_float = wrong.astype(np.float64)
    for group in groups:
        group_total = np.sum(weights[:, group], axis=1)
        group_errors = np.sum(weights[:, group] * wrong_float[group], axis=1)
        end = accepted + group_total
        start_clip = np.maximum(accepted, lower)
        end_clip = np.minimum(end, upper)
        active = end_clip > start_clip
        if np.any(active):
            rate = np.divide(
                group_errors,
                group_total,
                out=np.zeros_like(group_errors),
                where=group_total > 0.0,
            )
            coefficient = errors - rate * accepted
            area[active] += (
                rate[active] * (end_clip[active] - start_clip[active])
                + coefficient[active]
                * np.log(end_clip[active] / start_clip[active])
            )
        accepted = end
        errors += group_errors
    denominator = (high - low) * total_records
    return area / denominator


def bootstrap_weights(
    labels: np.ndarray, draws: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(labels)), dtype=np.int16)
    for draw in range(draws):
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            selected = rng.choice(members, size=len(members), replace=True)
            weights[draw] += np.bincount(selected, minlength=len(labels)).astype(np.int16)
    if not np.all(np.sum(weights, axis=1) == len(labels)):
        raise AssertionError("class-stratified bootstrap changed record count")
    return weights


def mad(values: np.ndarray) -> float:
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def point_and_bootstrap(
    pair_id: str,
    data: dict[str, Any],
    fused: dict[str, np.ndarray],
    partitions: list[tuple[int, ...]],
    edges: list[tuple[int, int, int, int]],
    draws: int,
    bootstrap_seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    labels = data["labels"]
    weights = bootstrap_weights(labels, draws, bootstrap_seed)
    prediction = np.argmax(fused["posterior"], axis=-1)
    wrong = prediction != labels[None, None, :]
    finest = partitions.index(tuple(range(6)))
    posterior_shift = np.mean(
        np.sum(
            np.abs(fused["posterior"] - fused["posterior"][finest : finest + 1]),
            axis=-1,
        ),
        axis=(1, 2),
    )
    accuracy = np.mean(prediction == labels[None, None, :], axis=(1, 2))
    mean_budget = np.mean(fused["budget"], axis=(1, 2))

    partition_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dispersion_rows: list[dict[str, Any]] = []
    for score_name in SCORES:
        point = np.empty((len(partitions), len(data["seeds"])), dtype=np.float64)
        boot = np.empty((draws, len(partitions), len(data["seeds"])), dtype=np.float64)
        for partition_index in range(len(partitions)):
            for seed_index in range(len(data["seeds"])):
                score = fused[score_name][partition_index, seed_index]
                error = wrong[partition_index, seed_index]
                point[partition_index, seed_index] = continuous_fractional_ncs(
                    score, error
                )[0]
                boot[:, partition_index, seed_index] = continuous_fractional_ncs(
                    score, error, weights
                )
        point_macro = np.mean(point, axis=1)
        boot_macro = np.mean(boot, axis=2)
        for partition_index, partition in enumerate(partitions):
            tie_values = []
            for seed_values in fused[score_name][partition_index]:
                counts = Counter(seed_values.tolist())
                tie_values.append(
                    (
                        len(counts) / len(seed_values),
                        max(counts.values()) / len(seed_values),
                        sum(count for count in counts.values() if count > 1)
                        / len(seed_values),
                    )
                )
            tie_values = np.asarray(tie_values, dtype=np.float64)
            partition_rows.append(
                {
                    "pair_id": pair_id,
                    "score": score_name,
                    "partition_id": f"P{partition_index:03d}",
                    "rgs": "".join(map(str, partition)),
                    "component_count": max(partition) + 1,
                    "ncsAURC_0p10_0p90": float(point_macro[partition_index]),
                    "ncsAURC_seed_sd": float(np.std(point[partition_index], ddof=1)),
                    "accuracy": float(accuracy[partition_index]),
                    "mean_evidence_budget": float(mean_budget[partition_index]),
                    "posterior_l1_from_finest": float(posterior_shift[partition_index]),
                    "unique_score_fraction": float(np.mean(tie_values[:, 0])),
                    "largest_tie_fraction": float(np.mean(tie_values[:, 1])),
                    "records_in_nonsingleton_ties_fraction": float(
                        np.mean(tie_values[:, 2])
                    ),
                }
            )

        edge_delta = np.empty((draws, len(edges)), dtype=np.float64)
        point_delta = np.empty(len(edges), dtype=np.float64)
        for edge_index, (fine, coarse, left, right) in enumerate(edges):
            point_delta[edge_index] = point_macro[coarse] - point_macro[fine]
            edge_delta[:, edge_index] = boot_macro[:, coarse] - boot_macro[:, fine]
            edge_rows.append(
                {
                    "pair_id": pair_id,
                    "score": score_name,
                    "edge_id": f"E{edge_index:03d}",
                    "fine_partition": f"P{fine:03d}",
                    "coarse_partition": f"P{coarse:03d}",
                    "fine_components": max(partitions[fine]) + 1,
                    "coarse_components": max(partitions[coarse]) + 1,
                    "merged_blocks": f"{left}+{right}",
                    "mean_budget_delta": float(mean_budget[coarse] - mean_budget[fine]),
                    "ncsAURC_delta": float(point_delta[edge_index]),
                    "direction": (
                        "coarsening_improves"
                        if point_delta[edge_index] < -TOLERANCE
                        else "coarsening_worsens"
                        if point_delta[edge_index] > TOLERANCE
                        else "tie"
                    ),
                }
            )
        point_fractions = {
            "coarsening_improves": float(np.mean(point_delta < -TOLERANCE)),
            "coarsening_worsens": float(np.mean(point_delta > TOLERANCE)),
            "tie": float(np.mean(np.abs(point_delta) <= TOLERANCE)),
        }
        boot_fractions = {
            "coarsening_improves": np.mean(edge_delta < -TOLERANCE, axis=1),
            "coarsening_worsens": np.mean(edge_delta > TOLERANCE, axis=1),
            "tie": np.mean(np.abs(edge_delta) <= TOLERANCE, axis=1),
        }
        for direction, values in boot_fractions.items():
            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "score": score_name,
                    "estimand": f"cover_edge_fraction_{direction}",
                    "point": point_fractions[direction],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "reference_edges": len(edges),
                    "bootstrap_draws": draws,
                }
            )
        for component_count in range(1, 7):
            selected = np.asarray(
                [
                    index
                    for index, partition in enumerate(partitions)
                    if max(partition) + 1 == component_count
                ],
                dtype=int,
            )
            point_values = point_macro[selected]
            boot_values = boot_macro[:, selected]
            statistics = {
                "iqr": (
                    float(np.quantile(point_values, 0.75) - np.quantile(point_values, 0.25)),
                    np.quantile(boot_values, 0.75, axis=1)
                    - np.quantile(boot_values, 0.25, axis=1),
                ),
                "range": (
                    float(np.max(point_values) - np.min(point_values)),
                    np.max(boot_values, axis=1) - np.min(boot_values, axis=1),
                ),
                "mad": (
                    mad(point_values),
                    np.median(
                        np.abs(boot_values - np.median(boot_values, axis=1)[:, None]),
                        axis=1,
                    ),
                ),
            }
            for statistic, (point_value, bootstrap_values) in statistics.items():
                dispersion_rows.append(
                    {
                        "pair_id": pair_id,
                        "score": score_name,
                        "component_count": component_count,
                        "partitions": len(selected),
                        "statistic": statistic,
                        "point": point_value,
                        "ci_low": float(np.quantile(bootstrap_values, 0.025)),
                        "ci_high": float(np.quantile(bootstrap_values, 0.975)),
                        "bootstrap_draws": draws,
                    }
                )
    return partition_rows, edge_rows, summary_rows, dispersion_rows


def structural_gate(
    pair_id: str,
    budgets: np.ndarray,
    partitions: list[tuple[int, ...]],
    edges: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    residuals = np.asarray(
        [np.max(budgets[coarse] - budgets[fine]) for fine, coarse, _, _ in edges],
        dtype=np.float64,
    )
    maximum_residual = float(np.max(residuals))
    return {
        "pair_id": pair_id,
        "partitions": len(partitions),
        "cover_edges": len(edges),
        "max_coarse_minus_fine_budget_residual": maximum_residual,
        "max_budget_violation": max(0.0, maximum_residual),
        "budget_nonincrease_on_every_record_seed_edge": bool(
            maximum_residual <= TOLERANCE
        ),
        "tolerance": TOLERANCE,
        "interpretation": (
            "Implementation-consistency check for the registered-coarsening budget order; "
            "not an empirical proof of the analytic corollary."
        ),
    }


def implementation_equivalence_gate(
    pair_id: str,
    data: dict[str, Any],
    fused: dict[str, np.ndarray],
    partitions: list[tuple[int, ...]],
) -> dict[str, Any]:
    """Check the vectorized sweep against the released PACT implementation."""

    seed_indices = sorted({0, len(data["seeds"]) // 2, len(data["seeds"]) - 1})
    record_indices = sorted(
        {0, len(data["record_ids"]) // 2, len(data["record_ids"]) - 1}
    )
    partition_indices = sorted({0, 17, len(partitions) // 2, len(partitions) - 1})
    posterior_residual = 0.0
    score_residual = 0.0
    budget_residual = 0.0
    component_equal = True
    cases = 0
    for seed_index in seed_indices:
        for record_index in record_indices:
            evidence = [
                np.asarray(values, dtype=np.float64)
                for values in data["evidence"][seed_index, record_index]
            ]
            for partition_index in partition_indices:
                partition = partitions[partition_index]
                parents = [f"component:{partition[view]}" for view in range(6)]
                observed = pact(evidence, parents)
                posterior_residual = max(
                    posterior_residual,
                    float(
                        np.max(
                            np.abs(
                                observed["posterior"]
                                - fused["posterior"][
                                    partition_index, seed_index, record_index
                                ]
                            )
                        )
                    ),
                )
                score_residual = max(
                    score_residual,
                    abs(
                        float(observed["score"])
                        - float(
                            fused["native_nonvacuity"][
                                partition_index, seed_index, record_index
                            ]
                        )
                    ),
                )
                budget_residual = max(
                    budget_residual,
                    abs(
                        float(observed["evidence_budget"])
                        - float(fused["budget"][partition_index, seed_index, record_index])
                    ),
                )
                component_equal &= int(observed["component_count"]) == max(partition) + 1
                cases += 1
    checks = {
        "posterior": posterior_residual <= TOLERANCE,
        "native_score": score_residual <= TOLERANCE,
        "evidence_budget": budget_residual <= TOLERANCE,
        "component_count": bool(component_equal),
    }
    return {
        "pair_id": pair_id,
        "pass": all(checks.values()),
        "cases": cases,
        "checks": checks,
        "max_posterior_residual": posterior_residual,
        "max_score_residual": score_residual,
        "max_budget_residual": budget_residual,
        "tolerance": TOLERANCE,
    }


def self_test() -> None:
    partitions = set_partitions(6)
    assert len(partitions) == 203
    assert Counter(max(partition) + 1 for partition in partitions) == {
        1: 1,
        2: 31,
        3: 90,
        4: 65,
        5: 15,
        6: 1,
    }
    edges = cover_edges(partitions)
    assert len(edges) == 856
    scores = np.asarray([0.9, 0.9, 0.2, 0.1])
    wrong = np.asarray([False, True, False, True])
    observed = continuous_fractional_ncs(scores, wrong, low=0.25, high=1.0)[0]
    dense = np.linspace(0.25, 1.0, 20001)
    # Analytic fractional risk for this toy: first tie has risk 0.5; later groups
    # add one correct and one incorrect item in descending-score order.
    expected_risk = np.where(
        dense <= 0.5,
        0.5,
        np.where(dense <= 0.75, 0.25 / dense, (dense - 0.5) / dense),
    )
    expected = float(np.trapezoid(expected_risk, dense) / 0.75)
    assert abs(observed - expected) < 1e-7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.protocol is None or args.output is None:
        parser.error("--input, --protocol, and --output are required")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    draws = int(args.draws if args.draws is not None else protocol["bootstrap"]["draws"])
    if sha256(args.input) != protocol["frozen_inputs"]["per_view_evidence_sha256"]:
        raise AssertionError("frozen evidence hash mismatch")
    if sha256(Path(__file__)) != protocol["frozen_inputs"]["analysis_script_sha256"]:
        raise AssertionError("analysis script hash mismatch")
    if sha256(ROOT / "src" / "action_admission" / "pcecf.py") != protocol[
        "frozen_inputs"
    ]["pcecf_sha256"]:
        raise AssertionError("PACT implementation hash mismatch")
    if sha256(ROOT / "experiments" / "topology_interventions.py") != protocol[
        "frozen_inputs"
    ]["topology_interventions_sha256"]:
        raise AssertionError("topology helper hash mismatch")
    if draws != int(protocol["bootstrap"]["draws"]):
        raise AssertionError("draw count differs from preregistered protocol")

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    partitions = set_partitions(6)
    edges = cover_edges(partitions)
    registry = [
        {
            "partition_id": f"P{index:03d}",
            "rgs": "".join(map(str, partition)),
            "component_count": max(partition) + 1,
            "blocks": "|".join("".join(map(str, block)) for block in blocks(partition)),
            "declared_policy": partition == tuple(range(6)),
            "all_merge_boundary": len(set(partition)) == 1,
        }
        for index, partition in enumerate(partitions)
    ]
    edge_registry = [
        {
            "edge_id": f"E{index:03d}",
            "fine_partition": f"P{fine:03d}",
            "coarse_partition": f"P{coarse:03d}",
            "fine_components": max(partitions[fine]) + 1,
            "coarse_components": max(partitions[coarse]) + 1,
            "merged_blocks": f"{left}+{right}",
        }
        for index, (fine, coarse, left, right) in enumerate(edges)
    ]
    write_csv(args.output / "PARTITION_REGISTRY.csv", registry)
    write_csv(args.output / "COVER_EDGE_REGISTRY.csv", edge_registry)

    partition_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    dispersion_rows: list[dict[str, Any]] = []
    gates = []
    implementation_gates = []
    for pair_offset, (pair_id, backbone, records) in enumerate(PAIR_SPECS):
        data = load_pair(args.input, backbone, records)
        fused = fuse_all(data["evidence"], partitions)
        implementation_gate = implementation_equivalence_gate(
            pair_id, data, fused, partitions
        )
        implementation_gates.append(implementation_gate)
        if not implementation_gate["pass"]:
            write_json(
                args.output / "IMPLEMENTATION_EQUIVALENCE_GATE.json",
                implementation_gates,
            )
            raise AssertionError(f"implementation-equivalence gate failed for {pair_id}")
        gate = structural_gate(pair_id, fused["budget"], partitions, edges)
        gates.append(gate)
        if not gate["budget_nonincrease_on_every_record_seed_edge"]:
            write_json(args.output / "STRUCTURAL_GATE.json", gates)
            raise AssertionError(f"budget-order gate failed for {pair_id}")
        rows = point_and_bootstrap(
            pair_id,
            data,
            fused,
            partitions,
            edges,
            draws,
            int(protocol["bootstrap"]["seed"]) + pair_offset,
        )
        partition_rows.extend(rows[0])
        edge_rows.extend(rows[1])
        summary_rows.extend(rows[2])
        dispersion_rows.extend(rows[3])

    write_csv(args.output / "PARTITION_METRICS.csv", partition_rows)
    write_csv(args.output / "COVER_EDGE_OUTCOMES.csv", edge_rows)
    write_csv(args.output / "COVER_EDGE_SUMMARY.csv", summary_rows)
    write_csv(args.output / "WITHIN_K_DISPERSION.csv", dispersion_rows)
    write_json(args.output / "STRUCTURAL_GATE.json", gates)
    write_json(
        args.output / "IMPLEMENTATION_EQUIVALENCE_GATE.json", implementation_gates
    )
    write_json(
        args.output / "RUN_MANIFEST.json",
        {
            "status": "PASS",
            "analysis": "exhaustive six-view registered-coarsening surface",
            "partitions": len(partitions),
            "cover_edges": len(edges),
            "pairs": [pair_id for pair_id, _, _ in PAIR_SPECS],
            "scores": list(SCORES),
            "support": list(SUPPORT),
            "bootstrap_draws": draws,
            "bootstrap_unit": "class-stratified record identifier",
            "frozen_emitter_realizations": 5,
            "protocol_sha256": sha256(args.protocol),
            "analysis_script_sha256": sha256(Path(__file__)),
            "input_sha256": sha256(args.input),
            "elapsed_seconds": time.time() - started,
            "scope": (
                "PACT-specific coarsening analysis; the 203 partitions do not test "
                "false splitting and do not generalize to registration-blind comparators."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
