"""Reproduce the controlled PC-ECF fusion and shared-verifier comparisons."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "experiments"))

from action_admission import (  # noqa: E402
    CONTRACT_CLASSES,
    SourceOpinion,
    dirichlet_predict,
    graph_from_parent_sets,
    hierarchy_matched_cautious,
    log_linear_posterior,
    restrict_dirichlet_input,
    verify_source_state,
)
from action_admission.baselines import (  # noqa: E402
    cautious_evidence_prediction,
    product_evidence_prediction,
    quality_weighted_prediction,
)
from controlled_study import (  # noqa: E402
    cluster_id,
    read_records,
    scene_fold_map,
    source_opinions,
    source_parents,
    uncompressed_sha256,
)
from action_admission.pcecf import SourceEvidence, forward  # noqa: E402


DATA = REPOSITORY / "data" / "controlled" / "source_records.jsonl.gz"
LABELS = REPOSITORY / "data" / "controlled" / "evaluation_labels.jsonl.gz"
CONFIG = REPOSITORY / "configs" / "controlled_study.json"
PROTOCOL = REPOSITORY / "configs" / "pcecf_study.json"
OUTPUT = REPOSITORY / "outputs" / "pcecf_study"
OPERATOR = REPOSITORY / "src" / "action_admission" / "pcecf.py"
HIERARCHICAL_CAUTIOUS_OPERATOR = (
    REPOSITORY / "src" / "action_admission" / "hierarchical_cautious.py"
)
METHODS = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
    "pcecf",
    "hierarchical_cautious_cumulative",
)
SOURCE_NAMES = ("language", "geometry", "risk")
CLASS_INDEX = {label: index for index, label in enumerate(CONTRACT_CLASSES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_id(record: Mapping[str, Any]) -> str:
    return str(record["metadata"]["scene_id"])


def vector(payload: Mapping[str, Any]) -> np.ndarray:
    probabilities = payload.get("probabilities", {})
    values = np.asarray(
        [float(probabilities.get(label, 0.0)) for label in CONTRACT_CLASSES],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("source probabilities must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros(len(CONTRACT_CLASSES), dtype=np.float64)
    return values / total


def structurally_valid(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema_valid", True) is not False
        and float(vector(payload).sum()) > 0.0
    )


def pcecf_sources(
    record: Mapping[str, Any],
    *,
    drop_missing: bool = False,
) -> list[SourceEvidence]:
    parents = source_parents(record)
    result = []
    for name in SOURCE_NAMES:
        payload = record["sources"].get(name, {})
        missing = not payload or bool(payload.get("missing", False))
        probabilities = vector(payload)
        valid = structurally_valid(payload)
        if drop_missing and missing:
            continue
        result.append(
            SourceEvidence(
                source_id=name,
                probabilities=probabilities,
                quality=float(payload.get("quality", 0.0)),
                conflict=float(payload.get("conflict", 0.0)),
                missing=missing,
                parents=tuple(parents[name]),
                valid=valid,
            )
        )
    return result


def distribution_dict(values: Sequence[float]) -> dict[str, float]:
    return {
        label: float(values[index])
        for index, label in enumerate(CONTRACT_CLASSES)
    }


def observed_source_count(record: Mapping[str, Any]) -> int:
    """Count available source slots that can contribute nonzero evidence."""

    count = 0
    for name in SOURCE_NAMES:
        payload = record["sources"].get(name, {})
        if (
            payload
            and not bool(payload.get("missing", False))
            and structurally_valid(payload)
        ):
            count += 1
    return count


def common_eligibility(record: Mapping[str, Any]) -> bool:
    """Method-independent support used by both headline isolation tables."""

    return observed_source_count(record) >= 2


def predict_record(
    record: Mapping[str, Any],
    method: str,
    concentration: float,
    *,
    pcecf_drop_missing: bool = False,
) -> tuple[str, dict[str, float], float, bool]:
    registered_graph = graph_from_parent_sets(source_parents(record))
    if method == "nested_evidential_composition":
        prediction = dirichlet_predict(
            restrict_dirichlet_input(record), concentration=concentration
        )
        return (
            prediction.predicted_contract,
            dict(prediction.probabilities),
            float(prediction.selection_score),
            bool(prediction.eligible),
        )
    if method == "quality_weighted_fusion":
        prediction = quality_weighted_prediction(record)
        return prediction.predicted_contract, dict(prediction.probabilities), float(prediction.confidence), True
    if method == "product_evidence_fusion":
        prediction = product_evidence_prediction(record)
        return prediction.predicted_contract, dict(prediction.probabilities), float(prediction.confidence), True
    if method == "cautious_evidence_fusion":
        prediction = cautious_evidence_prediction(record)
        return prediction.predicted_contract, dict(prediction.probabilities), float(prediction.confidence), True
    if method in {"lineage_unaware_pooling", "registered_lineage_pooling"}:
        graph = {} if method == "lineage_unaware_pooling" else registered_graph
        posterior = log_linear_posterior(source_opinions(record), graph)
        predicted = max(CONTRACT_CLASSES, key=posterior.__getitem__)
        eligible = max(
            float(record["sources"].get(name, {}).get("conflict", 0.0))
            for name in SOURCE_NAMES
        ) <= 0.65
        return predicted, dict(posterior), float(posterior[predicted]), eligible
    if method == "hierarchical_cautious_cumulative":
        output = hierarchy_matched_cautious(pcecf_sources(record))
        return (
            output.predicted_contract,
            dict(output.probabilities),
            float(output.selection_score),
            True,
        )
    if method == "pcecf":
        sources = pcecf_sources(record, drop_missing=pcecf_drop_missing)
        observed = observed_source_count(record)
        if not sources:
            posterior_values = np.full(
                len(CONTRACT_CLASSES), 1.0 / len(CONTRACT_CLASSES)
            )
            return (
                CONTRACT_CLASSES[0],
                distribution_dict(posterior_values),
                0.0,
                False,
            )
        output = forward(
            sources,
            concentration=concentration,
            expected_source_ids=(None if pcecf_drop_missing else SOURCE_NAMES),
        )
        posterior = distribution_dict(output.posterior)
        predicted = CONTRACT_CLASSES[output.predicted_index]
        return predicted, posterior, float(output.selection_score), observed >= 2
    raise ValueError(method)


def make_rows(
    records: list[dict[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    folds: Mapping[str, int],
    concentrations: Mapping[int, float],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    concentration_methods = {
        "nested_evidential_composition",
        "pcecf",
    }
    fixed_methods = tuple(method for method in METHODS if method not in concentration_methods)
    fixed_rows = {method: [] for method in fixed_methods}
    variant_rows = {
        (method, concentration): []
        for method in concentration_methods
        for concentration in sorted(set(concentrations.values()))
    }

    def append_row(
        target: list[dict[str, Any]],
        record: Mapping[str, Any],
        record_id: str,
        fold: int,
        label: str,
        method: str,
        concentration: float,
        registered_graph: Mapping[tuple[str, str], float],
    ) -> None:
        predicted, probabilities, score, native_eligible = predict_record(
            record, method, concentration
        )
        shared_eligible = common_eligibility(record)
        verification = verify_source_state(record, predicted, registered_graph)
        target.append(
            {
                "record_id": record_id,
                "scene_id": scene_id(record),
                "fold": fold,
                "score": score,
                "eligible": shared_eligible,
                "common_eligible": shared_eligible,
                "native_eligible": native_eligible,
                "verifier_pass": bool(verification.admissible),
                "verifier_route": str(verification.route),
                "verifier_reason": str(verification.reason),
                "fold_local_concentration": (
                    concentration if method in concentration_methods else None
                ),
                "predicted_contract": predicted,
                "preferred_contract": label,
                "probabilities": probabilities,
            }
        )

    for index, record in enumerate(records):
        record_id = str(record["record_id"])
        fold = int(folds[cluster_id(record_id)])
        registered_graph = graph_from_parent_sets(source_parents(record))
        label = str(labels_by_id[record_id]["preferred_contract"])
        for method in fixed_methods:
            append_row(
                fixed_rows[method],
                record,
                record_id,
                fold,
                label,
                method,
                float(concentrations[fold]),
                registered_graph,
            )
        for method in concentration_methods:
            for concentration in sorted(set(concentrations.values())):
                append_row(
                    variant_rows[(method, concentration)],
                    record,
                    record_id,
                    fold,
                    label,
                    method,
                    concentration,
                    registered_graph,
                )
        if (index + 1) % 4000 == 0:
            print(f"predictions: {index + 1}/{len(records)}", flush=True)
    rows_by_outer: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for method in fixed_methods:
        rows_by_outer[method] = {
            outer_fold: fixed_rows[method]
            for outer_fold in sorted(concentrations)
        }
    for method in concentration_methods:
        rows_by_outer[method] = {
            outer_fold: variant_rows[(method, float(concentrations[outer_fold]))]
            for outer_fold in sorted(concentrations)
        }
    return rows_by_outer


def threshold(rows: list[dict[str, Any]], target: float, *, verifier: bool) -> float:
    available = [
        row
        for row in rows
        if row["eligible"] and (row["verifier_pass"] or not verifier)
    ]
    available.sort(key=lambda row: (-float(row["score"]), str(row["record_id"])))
    count = int(round(target * len(rows)))
    if not available or count <= 0:
        return float("inf")
    return float(available[min(count, len(available)) - 1]["score"])


def evaluate_curve(
    rows_by_outer_fold: Mapping[int, list[dict[str, Any]]],
    targets: Sequence[float],
    folds: int,
    *,
    method: str,
    table: str,
    verifier: bool,
) -> tuple[
    list[dict[str, float]],
    dict[tuple[float, str], dict[str, int]],
    list[dict[str, Any]],
]:
    curve: list[dict[str, float]] = []
    scene_counts: dict[tuple[float, str], dict[str, int]] = {}
    threshold_rows: list[dict[str, Any]] = []
    for target in targets:
        total = defaultdict(int)
        by_scene: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        for fold in range(folds):
            rows = rows_by_outer_fold[fold]
            train = [row for row in rows if int(row["fold"]) != fold]
            test = [row for row in rows if int(row["fold"]) == fold]
            cutoff = threshold(train, target, verifier=verifier)
            threshold_rows.append(
                {
                    "table": table,
                    "method": method,
                    "outer_fold": fold,
                    "target_coverage": float(target),
                    "threshold_fit_scope": "outer_train_only",
                    "threshold": cutoff,
                    "train_records": len(train),
                    "train_common_eligible": sum(bool(row["eligible"]) for row in train),
                    "train_native_eligible": sum(bool(row["native_eligible"]) for row in train),
                    "eligibility_protocol": "at_least_two_observed_sources_for_every_method",
                    "train_verifier_admissible": sum(
                        bool(row["verifier_pass"]) for row in train
                    ),
                    "test_records": len(test),
                    "fold_local_concentration": (
                        test[0]["fold_local_concentration"] if test else None
                    ),
                    "verifier_required": verifier,
                }
            )
            for row in test:
                accepted = bool(
                    row["eligible"]
                    and (row["verifier_pass"] or not verifier)
                    and float(row["score"]) >= cutoff
                )
                correct = bool(accepted and row["predicted_contract"] == row["preferred_contract"])
                wrong = bool(accepted and not correct)
                for bucket in (total, by_scene[str(row["scene_id"])]):
                    bucket["n"] += 1
                    bucket["admitted"] += int(accepted)
                    bucket["wrong"] += int(wrong)
                    bucket["correct"] += int(correct)
        n = total["n"]
        admitted = total["admitted"]
        curve.append(
            {
                "target": float(target),
                "n": int(n),
                "admitted": int(admitted),
                "wrong": int(total["wrong"]),
                "correct": int(total["correct"]),
                "coverage": admitted / n,
                "wrong_all": total["wrong"] / n,
                "correct_all": total["correct"] / n,
                "wrong_admitted": total["wrong"] / admitted if admitted else 0.0,
            }
        )
        for scene, counts in by_scene.items():
            scene_counts[(float(target), scene)] = dict(counts)
    return curve, scene_counts, threshold_rows

def interpolate_naurc(curve: Sequence[Mapping[str, float]], grid: np.ndarray) -> float:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in curve:
        grouped[float(row["coverage"])].append(float(row["wrong_admitted"]))
    coverage = np.asarray(sorted(grouped), dtype=np.float64)
    risk = np.asarray([np.mean(grouped[value]) for value in coverage], dtype=np.float64)
    if coverage[0] > grid[0] or coverage[-1] < grid[-1]:
        raise ValueError(f"curve support [{coverage[0]}, {coverage[-1]}] misses {grid[[0,-1]]}")
    values = np.interp(grid, coverage, risk)
    return float(np.trapezoid(values, grid) / (grid[-1] - grid[0]))


def bootstrap_naurc(
    counts: Mapping[str, Mapping[tuple[float, str], Mapping[str, int]]],
    targets: Sequence[float],
    scenes: Sequence[str],
    grid: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], tuple[str, ...], np.ndarray]:
    methods = tuple(counts)
    fields = ("n", "admitted", "wrong", "correct")
    values = np.zeros((len(methods), len(targets), len(scenes), len(fields)))
    for method_index, method in enumerate(methods):
        for target_index, target in enumerate(targets):
            for scene_index, scene in enumerate(scenes):
                cell = counts[method][(float(target), scene)]
                values[method_index, target_index, scene_index, :] = [cell[field] for field in fields]
    rng = np.random.default_rng(seed)
    draws = np.zeros((replicates, len(methods)), dtype=np.float64)
    for replicate in range(replicates):
        weights = rng.multinomial(len(scenes), np.full(len(scenes), 1.0 / len(scenes)))
        totals = np.tensordot(values, weights, axes=(2, 0))
        for method_index in range(len(methods)):
            curve = []
            for target_index in range(len(targets)):
                n, admitted, wrong, _ = totals[method_index, target_index, :]
                curve.append(
                    {
                        "coverage": admitted / n if n else 0.0,
                        "wrong_admitted": wrong / admitted if admitted else 0.0,
                    }
                )
            draws[replicate, method_index] = interpolate_naurc(curve, grid)
    summary = {
        method: {
            "mean": float(np.mean(draws[:, index])),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
            "replicates": replicates,
            "seed": seed,
        }
        for index, method in enumerate(methods)
    }
    return summary, methods, draws


def paired_delta(
    methods: Sequence[str],
    draws: np.ndarray,
    left: str,
    right: str,
) -> dict[str, float | int | str]:
    left_index = methods.index(left)
    right_index = methods.index(right)
    delta = draws[:, left_index] - draws[:, right_index]
    return {
        "estimand": f"nAURC({left}) - nAURC({right})",
        "left_method": left,
        "right_method": right,
        "mean": float(np.mean(delta)),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "fraction_below_zero": float(np.mean(delta < 0.0)),
        "replicates": int(delta.size),
    }

def calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    correct = np.asarray(
        [row["predicted_contract"] == row["preferred_contract"] for row in rows],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [float(row["probabilities"][row["predicted_contract"]]) for row in rows],
        dtype=np.float64,
    )
    nll = []
    brier = []
    for row in rows:
        preferred = str(row["preferred_contract"])
        probabilities = row["probabilities"]
        nll.append(-math.log(max(float(probabilities[preferred]), 1e-12)))
        brier.append(
            sum(
                (float(probabilities[label]) - float(label == preferred)) ** 2
                for label in CONTRACT_CLASSES
            )
        )
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return {
        "accuracy": float(np.mean(correct)),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece10": ece,
    }


def scene_macro_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(
            float(row["predicted_contract"] == row["preferred_contract"])
        )
    return float(np.mean([np.mean(values) for values in by_scene.values()]))


def duplicate_record(record: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(record))
    copied["sources"][f"{source_name}_copy"] = json.loads(
        json.dumps(copied["sources"][source_name])
    )
    return copied


def generic_nested(record: Mapping[str, Any], concentration: float) -> np.ndarray:
    opinions: list[tuple[np.ndarray, float]] = []
    for payload in record["sources"].values():
        if not isinstance(payload, Mapping) or bool(payload.get("missing", False)):
            continue
        evidence = concentration * float(np.clip(payload.get("quality", 0.0), 0.0, 1.0))
        evidence *= 1.0 - float(np.clip(payload.get("conflict", 0.0), 0.0, 1.0))
        values = evidence * vector(payload)
        strength = float(values.sum() + len(CONTRACT_CLASSES))
        opinions.append((values / strength, len(CONTRACT_CLASSES) / strength))
    if len(opinions) < 2:
        return np.full(len(CONTRACT_CLASSES), 1.0 / len(CONTRACT_CLASSES))
    belief, uncertainty = opinions[0]
    for right_belief, right_uncertainty in opinions[1:]:
        conflict = max(float(belief.sum() * right_belief.sum() - np.dot(belief, right_belief)), 0.0)
        denominator = max(1.0 - conflict, 1e-12)
        belief = (
            belief * right_belief
            + belief * right_uncertainty
            + right_belief * uncertainty
        ) / denominator
        uncertainty = uncertainty * right_uncertainty / denominator
    posterior = belief + uncertainty / len(CONTRACT_CLASSES)
    return posterior / posterior.sum()


def pcecf_duplicate_output(
    record: Mapping[str, Any],
    source_name: str,
    concentration: float,
    *,
    epsilon: float = 0.0,
) -> np.ndarray:
    sources = pcecf_sources(record)
    original = next(source for source in sources if source.source_id == source_name)
    probabilities = original.probabilities.copy()
    if epsilon > 0.0:
        order = np.argsort(-probabilities)
        move = min(epsilon, float(probabilities[order[0]]) / 2.0)
        probabilities[order[0]] -= move
        probabilities[order[1]] += move
    sources.append(
        SourceEvidence(
            source_id=f"{source_name}_copy",
            probabilities=probabilities,
            quality=original.quality,
            conflict=original.conflict,
            missing=original.missing,
            parents=original.parents,
        )
    )
    return forward(
        sources,
        concentration=concentration,
    ).posterior


def hierarchical_cautious_duplicate_output(
    record: Mapping[str, Any],
    source_name: str,
) -> tuple[np.ndarray, float]:
    sources = pcecf_sources(record)
    original = next(source for source in sources if source.source_id == source_name)
    sources.append(
        SourceEvidence(
            source_id=f"{source_name}_copy",
            probabilities=original.probabilities.copy(),
            quality=original.quality,
            conflict=original.conflict,
            missing=original.missing,
            parents=original.parents,
        )
    )
    output = hierarchy_matched_cautious(sources)
    return (
        np.asarray(
            [float(output.probabilities[label]) for label in CONTRACT_CLASSES],
            dtype=np.float64,
        ),
        float(output.selection_score),
    )


def duplication_metrics(
    clean_records: Sequence[Mapping[str, Any]],
    folds: Mapping[str, int],
    concentrations: Mapping[int, float],
) -> dict[str, Any]:
    exact_drifts = []
    exact_flips = 0
    hierarchical_cautious_exact_drifts = []
    hierarchical_cautious_exact_score_drifts = []
    hierarchical_cautious_exact_flips = 0
    near_rows = []
    for record in clean_records:
        record_id = str(record["record_id"])
        concentration = float(concentrations[int(folds[cluster_id(record_id)])])
        base = forward(
            pcecf_sources(record),
            concentration=concentration,
        ).posterior
        hierarchical_cautious_base_output = hierarchy_matched_cautious(
            pcecf_sources(record)
        )
        hierarchical_cautious_base = np.asarray(
            [
                float(hierarchical_cautious_base_output.probabilities[label])
                for label in CONTRACT_CLASSES
            ],
            dtype=np.float64,
        )
        for source_name in SOURCE_NAMES:
            copied = pcecf_duplicate_output(record, source_name, concentration)
            drift = float(np.linalg.norm(copied - base, ord=1))
            exact_drifts.append(drift)
            exact_flips += int(np.argmax(copied) != np.argmax(base))
            (
                hierarchical_cautious_copied,
                hierarchical_cautious_copied_score,
            ) = hierarchical_cautious_duplicate_output(record, source_name)
            hierarchical_cautious_drift = float(
                np.linalg.norm(
                    hierarchical_cautious_copied - hierarchical_cautious_base,
                    ord=1,
                )
            )
            hierarchical_cautious_exact_drifts.append(hierarchical_cautious_drift)
            hierarchical_cautious_exact_score_drifts.append(
                abs(
                    hierarchical_cautious_copied_score
                    - float(hierarchical_cautious_base_output.selection_score)
                )
            )
            hierarchical_cautious_exact_flips += int(
                np.argmax(hierarchical_cautious_copied)
                != np.argmax(hierarchical_cautious_base)
            )
        if record_id.endswith("__seed00"):
            for epsilon in (0.001, 0.005, 0.01, 0.02):
                copied = pcecf_duplicate_output(
                    record, "geometry", concentration, epsilon=epsilon
                )
                near_rows.append(
                    {
                        "epsilon": epsilon,
                        "l1_drift": float(np.linalg.norm(copied - base, ord=1)),
                    }
                )
    x = np.asarray([row["epsilon"] for row in near_rows])
    y = np.asarray([row["l1_drift"] for row in near_rows])
    slope = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
    return {
        "exact_duplicate_comparisons": len(exact_drifts),
        "exact_duplicate_max_l1": max(exact_drifts, default=0.0),
        "exact_duplicate_mean_l1": float(np.mean(exact_drifts)) if exact_drifts else 0.0,
        "exact_duplicate_flip_rate": exact_flips / len(exact_drifts) if exact_drifts else 0.0,
        "hierarchical_cautious_exact_duplicate_comparisons": len(
            hierarchical_cautious_exact_drifts
        ),
        "hierarchical_cautious_exact_duplicate_max_l1": max(
            hierarchical_cautious_exact_drifts, default=0.0
        ),
        "hierarchical_cautious_exact_duplicate_mean_l1": (
            float(np.mean(hierarchical_cautious_exact_drifts))
            if hierarchical_cautious_exact_drifts
            else 0.0
        ),
        "hierarchical_cautious_exact_duplicate_max_score_delta": max(
            hierarchical_cautious_exact_score_drifts, default=0.0
        ),
        "hierarchical_cautious_exact_duplicate_mean_score_delta": (
            float(np.mean(hierarchical_cautious_exact_score_drifts))
            if hierarchical_cautious_exact_score_drifts
            else 0.0
        ),
        "hierarchical_cautious_exact_duplicate_flip_rate": (
            hierarchical_cautious_exact_flips
            / len(hierarchical_cautious_exact_drifts)
            if hierarchical_cautious_exact_drifts
            else 0.0
        ),
        "near_copy_comparisons": len(near_rows),
        "near_copy_origin_slope": slope,
        "near_copy_max_l1": max((row["l1_drift"] for row in near_rows), default=0.0),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
            for row in rows:
                payload = (
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.write(payload.encode("utf-8"))


def held_out_rows(
    rows_by_outer_fold: Mapping[int, list[dict[str, Any]]],
    fold_count: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for outer_fold in range(fold_count)
        for row in rows_by_outer_fold[outer_fold]
        if int(row["fold"]) == outer_fold
    ]
    record_ids = {str(row["record_id"]) for row in rows}
    if len(rows) != 31_200 or len(record_ids) != len(rows):
        raise AssertionError("OOF rows must contain each controlled record exactly once")
    return rows


def make_source_absent_rows(
    records: Sequence[Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    folds: Mapping[str, int],
    concentrations: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    by_concentration: dict[float, list[dict[str, Any]]] = {
        value: [] for value in sorted(set(concentrations.values()))
    }
    for record in records:
        record_id = str(record["record_id"])
        fold = int(folds[cluster_id(record_id)])
        registered_graph = graph_from_parent_sets(source_parents(record))
        label = str(labels_by_id[record_id]["preferred_contract"])
        for concentration, target in by_concentration.items():
            predicted, probabilities, score, native_eligible = predict_record(
                record,
                "pcecf",
                concentration,
                pcecf_drop_missing=True,
            )
            shared_eligible = common_eligibility(record)
            verification = verify_source_state(record, predicted, registered_graph)
            target.append(
                {
                    "record_id": record_id,
                    "scene_id": scene_id(record),
                    "fold": fold,
                    "score": score,
                    "eligible": shared_eligible,
                    "common_eligible": shared_eligible,
                    "native_eligible": native_eligible,
                    "verifier_pass": bool(verification.admissible),
                    "verifier_route": str(verification.route),
                    "verifier_reason": str(verification.reason),
                    "fold_local_concentration": concentration,
                    "predicted_contract": predicted,
                    "preferred_contract": label,
                    "probabilities": probabilities,
                }
            )
    return {
        outer_fold: by_concentration[float(concentrations[outer_fold])]
        for outer_fold in sorted(concentrations)
    }


def build_fold_manifest(
    records: Sequence[Mapping[str, Any]],
    folds: Mapping[str, int],
    concentrations: Mapping[int, float],
) -> dict[str, Any]:
    scenes_by_fold: dict[int, set[str]] = defaultdict(set)
    clusters_by_fold: dict[int, set[str]] = defaultdict(set)
    records_by_fold: dict[int, int] = defaultdict(int)
    observed_scene_fold: dict[str, set[int]] = defaultdict(set)
    for record in records:
        record_id = str(record["record_id"])
        cluster = cluster_id(record_id)
        fold = int(folds[cluster])
        scene = scene_id(record)
        scenes_by_fold[fold].add(scene)
        clusters_by_fold[fold].add(cluster)
        records_by_fold[fold] += 1
        observed_scene_fold[scene].add(fold)
    if any(len(values) != 1 for values in observed_scene_fold.values()):
        raise AssertionError("a scene crossed outer folds")
    return {
        "fold_count": len(concentrations),
        "assignment_unit": "scene",
        "folds": [
            {
                "outer_fold": fold,
                "test_scenes": sorted(scenes_by_fold[fold]),
                "test_scene_count": len(scenes_by_fold[fold]),
                "test_cluster_count": len(clusters_by_fold[fold]),
                "test_record_count": records_by_fold[fold],
                "train_scene_count": sum(
                    len(scenes_by_fold[other])
                    for other in concentrations
                    if other != fold
                ),
                "train_record_count": sum(
                    records_by_fold[other]
                    for other in concentrations
                    if other != fold
                ),
                "fold_local_dirichlet_concentration": concentrations[fold],
            }
            for fold in sorted(concentrations)
        ],
    }


def exact_rank_selection(
    rows: Sequence[Mapping[str, Any]],
    count_by_fold: Mapping[int, int],
    *,
    verifier: bool,
) -> tuple[dict[str, Any], set[str]]:
    selected_rows: list[Mapping[str, Any]] = []
    thresholds: dict[str, float | None] = {}
    available_by_fold: dict[str, int] = {}
    for fold in sorted(count_by_fold):
        candidates = [
            row
            for row in rows
            if int(row["fold"]) == fold
            and bool(row["eligible"])
            and (bool(row["verifier_pass"]) or not verifier)
        ]
        candidates.sort(
            key=lambda row: (-float(row["score"]), str(row["record_id"]))
        )
        requested = int(count_by_fold[fold])
        chosen = candidates[:requested]
        selected_rows.extend(chosen)
        thresholds[str(fold)] = float(chosen[-1]["score"]) if chosen else None
        available_by_fold[str(fold)] = len(candidates)
    selected = {str(row["record_id"]) for row in selected_rows}
    n = len(rows)
    requested_total = sum(int(value) for value in count_by_fold.values())
    wrong = sum(
        str(row["predicted_contract"]) != str(row["preferred_contract"])
        for row in selected_rows
    )
    correct = len(selected_rows) - wrong
    return (
        {
            "n": n,
            "requested_admitted": requested_total,
            "requested_admitted_by_outer_fold": json.dumps(
                {str(key): int(value) for key, value in sorted(count_by_fold.items())},
                sort_keys=True,
            ),
            "available_candidates": sum(available_by_fold.values()),
            "available_candidates_by_outer_fold": json.dumps(
                available_by_fold, sort_keys=True
            ),
            "admitted": len(selected_rows),
            "coverage": len(selected_rows) / n,
            "wrong": wrong,
            "wrong_all": wrong / n,
            "wrong_admitted": wrong / len(selected_rows) if selected_rows else 0.0,
            "correct": correct,
            "correct_all": correct / n,
            "score_threshold_by_outer_fold": json.dumps(
                thresholds, sort_keys=True
            ),
            "selection": "retrospective fold-stratified score-only exact-count ranking",
            "tie_break": "record_id ascending within outer fold",
            "uses_evaluation_labels_for_selection": False,
        },
        selected,
    )

def missing_slot_sensitivity(
    records: Sequence[Mapping[str, Any]],
    retained: Sequence[Mapping[str, Any]],
    absent: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records_by_id = {str(record["record_id"]): record for record in records}
    absent_by_id = {str(row["record_id"]): row for row in absent}
    by_missing: dict[bool, list[float]] = {False: [], True: []}
    flips: dict[bool, int] = {False: 0, True: 0}
    missing_slots = 0
    all_sources_absent = 0
    for row in retained:
        record_id = str(row["record_id"])
        record = records_by_id[record_id]
        missing_count = sum(
            not bool(record["sources"].get(name, {}))
            or bool(record["sources"].get(name, {}).get("missing", False))
            for name in SOURCE_NAMES
        )
        missing_slots += missing_count
        all_sources_absent += int(missing_count == len(SOURCE_NAMES))
        other = absent_by_id[record_id]
        left = np.asarray(
            [float(row["probabilities"][label]) for label in CONTRACT_CLASSES]
        )
        right = np.asarray(
            [float(other["probabilities"][label]) for label in CONTRACT_CLASSES]
        )
        key = missing_count > 0
        by_missing[key].append(float(np.linalg.norm(left - right, ord=1)))
        flips[key] += int(row["predicted_contract"] != other["predicted_contract"])

    def summarize(key: bool) -> dict[str, Any]:
        values = by_missing[key]
        return {
            "records": len(values),
            "mean_l1_posterior_drift": float(np.mean(values)) if values else 0.0,
            "max_l1_posterior_drift": max(values, default=0.0),
            "prediction_flips": flips[key],
            "prediction_flip_rate": flips[key] / len(values) if values else 0.0,
        }

    return {
        "primary_protocol": "missing-slot-retained",
        "sensitivity_protocol": "source-absent",
        "primary_protocol_changed": False,
        "missing_slots": missing_slots,
        "all_sources_absent_records": all_sources_absent,
        "records_without_missing_source": summarize(False),
        "records_with_at_least_one_missing_source": summarize(True),
    }


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    OUTPUT = arguments.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    observed_source_hash = uncompressed_sha256(DATA)
    observed_label_hash = uncompressed_sha256(LABELS)
    if observed_source_hash != config["source_records_sha256"]:
        raise AssertionError("source-record content hash changed")
    if observed_label_hash != config["evaluation_labels_sha256"]:
        raise AssertionError("evaluation-label content hash changed")
    artifact_hashes = {
        "operator": {"path": str(OPERATOR), "sha256": sha256(OPERATOR)},
        "hierarchical_cautious_operator": {
            "path": str(HIERARCHICAL_CAUTIOUS_OPERATOR),
            "sha256": sha256(HIERARCHICAL_CAUTIOUS_OPERATOR),
        },
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "source_records": {
            "path": str(DATA),
            "file_sha256": sha256(DATA),
            "uncompressed_sha256": observed_source_hash,
        },
        "evaluation_labels": {
            "path": str(LABELS),
            "file_sha256": sha256(LABELS),
            "uncompressed_sha256": observed_label_hash,
        },
        "controlled_config": {"path": str(CONFIG), "sha256": sha256(CONFIG)},
        "controlled_study_module": {
            "path": str(REPOSITORY / "experiments" / "controlled_study.py"),
            "sha256": sha256(REPOSITORY / "experiments" / "controlled_study.py"),
        },
        "verifier_module": {
            "path": str(REPOSITORY / "src" / "action_admission" / "verifier.py"),
            "sha256": sha256(REPOSITORY / "src" / "action_admission" / "verifier.py"),
        },
    }
    (OUTPUT / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records = read_records(DATA)
    labels = read_records(LABELS)
    if len(records) != 31_200 or len(labels) != len(records):
        raise AssertionError("expected 31,200 one-to-one controlled records and labels")
    labels_by_id = {str(row["record_id"]): row for row in labels}
    if len(labels_by_id) != len(labels):
        raise AssertionError("evaluation-label identifiers are not unique")
    record_ids = {str(record["record_id"]) for record in records}
    if set(labels_by_id) != record_ids:
        raise AssertionError("source-record and evaluation-label identifiers differ")

    fold_count = int(config["fold_count"])
    folds = scene_fold_map(records, fold_count)
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    fold_manifest = build_fold_manifest(records, folds, concentrations)
    (OUTPUT / "fold_manifest.json").write_text(
        json.dumps(fold_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows_by_outer = make_rows(records, labels_by_id, folds, concentrations)
    oof_by_method = {
        method: held_out_rows(rows_by_outer[method], fold_count)
        for method in METHODS
    }
    targets = tuple(round(index / 100.0, 2) for index in range(1, 61))
    common = np.linspace(*protocol["common_coverage"], 36)
    scenes = sorted({scene_id(record) for record in records})
    bootstrap_replicates = int(config["bootstrap_replicates"])
    bootstrap_seed = int(config["bootstrap_seed"])

    calibration_rows = []
    clean_rows = []
    for method in METHODS:
        all_metrics = calibration(oof_by_method[method])
        selected_clean = [
            row
            for row in oof_by_method[method]
            if "__clean_control__" in str(row["record_id"])
        ]
        clean_metrics = calibration(selected_clean)
        calibration_rows.append(
            {
                "method": method,
                "subset": "all",
                **all_metrics,
                "scene_macro_accuracy": scene_macro_accuracy(oof_by_method[method]),
            }
        )
        clean_rows.append(
            {
                "method": method,
                "subset": "clean_control",
                **clean_metrics,
                "scene_macro_accuracy": scene_macro_accuracy(selected_clean),
            }
        )

    table_results: dict[str, Any] = {}
    all_threshold_rows: list[dict[str, Any]] = []
    bootstrap_draws: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    training_target = float(config["fixed_target"])
    for verifier, table_name in (
        (False, "table1_no_verifier"),
        (True, "table2_shared_verifier"),
    ):
        curves = {}
        counts = {}
        summaries = []
        training_points = []
        for method in METHODS:
            curve, method_counts, threshold_rows = evaluate_curve(
                rows_by_outer[method],
                targets,
                fold_count,
                method=method,
                table=table_name,
                verifier=verifier,
            )
            for row in threshold_rows:
                row["protocol_role"] = "primary"
                if row["threshold"] is not None and not math.isfinite(float(row["threshold"])):
                    row["threshold"] = None
            curves[method] = curve
            counts[method] = method_counts
            all_threshold_rows.extend(threshold_rows)
            summaries.append(
                {
                    "method": method,
                    "normalized_aurc_point": interpolate_naurc(curve, common),
                }
            )
            point = next(
                row for row in curve if math.isclose(float(row["target"]), training_target)
            )
            training_points.append(
                {
                    "method": method,
                    "operating_point_type": "outer-train-fitted threshold at prespecified coverage target",
                    "prespecified_training_target": training_target,
                    **point,
                }
            )
        bootstrap, method_order, draws = bootstrap_naurc(
            counts,
            targets,
            scenes,
            common,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        bootstrap_draws[table_name] = (method_order, draws)
        for row in summaries:
            row.update(
                {
                    f"scene_bootstrap_{key}": value
                    for key, value in bootstrap[row["method"]].items()
                }
            )
        flat_curve = [
            {"method": method, **row}
            for method, curve in curves.items()
            for row in curve
        ]
        write_csv(OUTPUT / f"{table_name}_risk_coverage.csv", flat_curve)
        write_csv(OUTPUT / f"{table_name}_summary.csv", summaries)
        write_csv(OUTPUT / f"{table_name}_training_fitted_point.csv", training_points)
        table_results[table_name] = {
            "summary": summaries,
            "training_fitted_operating_point": training_points,
        }

    paired_deltas = {
        "table1_pcecf_minus_product": paired_delta(
            *bootstrap_draws["table1_no_verifier"],
            "pcecf",
            "product_evidence_fusion",
        ),
        "table1_pcecf_minus_registered_lineage": paired_delta(
            *bootstrap_draws["table1_no_verifier"],
            "pcecf",
            "registered_lineage_pooling",
        ),
        "table2_pcecf_minus_nested": paired_delta(
            *bootstrap_draws["table2_shared_verifier"],
            "pcecf",
            "nested_evidential_composition",
        ),
    }
    (OUTPUT / "paired_scene_bootstrap_deltas.json").write_text(
        json.dumps(paired_deltas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    table2_summary = {
        row["method"]: row
        for row in table_results["table2_shared_verifier"]["summary"]
    }
    table2_training = {
        row["method"]: row
        for row in table_results["table2_shared_verifier"]["training_fitted_operating_point"]
    }
    baseline_anchor_checks: dict[str, Any] = {}
    # Only nested composition had the same native and common valid-source eligibility rule.
    # The remaining canonical anchors belong to the earlier method-native
    # eligibility protocol and are not valid equality targets after harmonization.
    for method in ("nested_evidential_composition",):
        expected = config["expected"][method]
        observed = float(table2_summary[method]["scene_bootstrap_mean"])
        expected_naurc = float(expected["normalized_aurc"])
        baseline_anchor_checks[f"{method}_bootstrap_naurc"] = {
            "expected": expected_naurc,
            "observed": observed,
            "absolute_difference": abs(observed - expected_naurc),
            "pass": bool(np.isclose(observed, expected_naurc, atol=1e-12, rtol=0.0)),
            "scope": "unchanged native/common valid-source eligibility",
        }
    nested_expected = config["expected"]["nested_evidential_composition"]
    nested_point = table2_training["nested_evidential_composition"]
    baseline_anchor_checks["nested_training_target_counts"] = {
        "expected_admitted": int(nested_expected["fixed_target_admitted"]),
        "observed_admitted": int(nested_point["admitted"]),
        "expected_wrong": int(nested_expected["fixed_target_wrong"]),
        "observed_wrong": int(nested_point["wrong"]),
        "pass": (
            int(nested_point["admitted"]) == int(nested_expected["fixed_target_admitted"])
            and int(nested_point["wrong"]) == int(nested_expected["fixed_target_wrong"])
        ),
    }
    baseline_anchor_pass = all(
        bool(check["pass"]) for check in baseline_anchor_checks.values()
    )

    primary_threshold_at_training_target = {
        (
            str(row["table"]),
            str(row["method"]),
            int(row["outer_fold"]),
        ): row["threshold"]
        for row in all_threshold_rows
        if row["protocol_role"] == "primary"
        and math.isclose(float(row["target_coverage"]), training_target)
    }
    matched_rows = []
    matched_selected: dict[str, dict[str, set[str]]] = {}
    for table_name, verifier in (
        ("table1_no_verifier", False),
        ("table2_shared_verifier", True),
    ):
        reference_count_by_fold: dict[int, int] = defaultdict(int)
        for row in oof_by_method["pcecf"]:
            outer_fold = int(row["fold"])
            cutoff = primary_threshold_at_training_target[
                (table_name, "pcecf", outer_fold)
            ]
            admitted = bool(
                cutoff is not None
                and row["eligible"]
                and (row["verifier_pass"] or not verifier)
                and float(row["score"]) >= float(cutoff)
            )
            reference_count_by_fold[outer_fold] += int(admitted)
        matched_selected[table_name] = {}
        for method in METHODS:
            metrics, selected = exact_rank_selection(
                oof_by_method[method],
                reference_count_by_fold,
                verifier=verifier,
            )
            matched_selected[table_name][method] = selected
            matched_rows.append(
                {
                    "table": table_name,
                    "method": method,
                    "reference_method": "pcecf",
                    "reference_count_source": (
                        "PC-ECF outer-train-fitted 0.13 target point, matched within outer fold"
                    ),
                    **metrics,
                }
            )
    write_csv(OUTPUT / "retrospective_matched_coverage_diagnostic.csv", matched_rows)
    absent_rows_by_outer = make_source_absent_rows(
        records,
        labels_by_id,
        folds,
        concentrations,
    )
    absent_oof = held_out_rows(absent_rows_by_outer, fold_count)
    sensitivity = missing_slot_sensitivity(
        records,
        oof_by_method["pcecf"],
        absent_oof,
    )
    sensitivity_tables = {}
    for verifier, table_name in (
        (False, "table1_no_verifier"),
        (True, "table2_shared_verifier"),
    ):
        curve, _, threshold_rows = evaluate_curve(
            absent_rows_by_outer,
            targets,
            fold_count,
            method="pcecf_source_absent_sensitivity",
            table=table_name,
            verifier=verifier,
        )
        for row in threshold_rows:
            row["protocol_role"] = "missing-representation sensitivity"
            if row["threshold"] is not None and not math.isfinite(float(row["threshold"])):
                row["threshold"] = None
        all_threshold_rows.extend(threshold_rows)
        absent_naurc = interpolate_naurc(curve, common)
        retained_naurc = next(
            float(row["normalized_aurc_point"])
            for row in table_results[table_name]["summary"]
            if row["method"] == "pcecf"
        )
        point = next(
            row for row in curve if math.isclose(float(row["target"]), training_target)
        )
        sensitivity_tables[table_name] = {
            "missing_slot_retained_normalized_aurc": retained_naurc,
            "source_absent_normalized_aurc": absent_naurc,
            "source_absent_minus_retained": absent_naurc - retained_naurc,
            "source_absent_training_fitted_point": point,
        }
    sensitivity["selective_metrics"] = sensitivity_tables
    no_missing_drift = float(
        sensitivity["records_without_missing_source"]["max_l1_posterior_drift"]
    )
    sensitivity["unchanged_when_no_slot_is_missing"] = no_missing_drift <= 1e-15
    (OUTPUT / "missing_slot_retained_vs_source_absent.json").write_text(
        json.dumps(sensitivity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_jsonl_gz(OUTPUT / "fold_thresholds.jsonl.gz", all_threshold_rows)
    threshold_at_training_target = {
        (
            str(row["table"]),
            str(row["method"]),
            int(row["outer_fold"]),
        ): row["threshold"]
        for row in all_threshold_rows
        if row["protocol_role"] == "primary"
        and math.isclose(float(row["target_coverage"]), training_target)
    }
    matched_by_key = {
        (str(row["table"]), str(row["method"])): row
        for row in matched_rows
    }
    matched_thresholds_by_key = {
        key: json.loads(str(value["score_threshold_by_outer_fold"]))
        for key, value in matched_by_key.items()
    }
    oof_payload = []
    for method in METHODS:
        for row in oof_by_method[method]:
            outer_fold = int(row["fold"])
            table_payload = {}
            for table_name, verifier in (
                ("table1_no_verifier", False),
                ("table2_shared_verifier", True),
            ):
                cutoff = threshold_at_training_target[(table_name, method, outer_fold)]
                training_admitted = bool(
                    cutoff is not None
                    and row["eligible"]
                    and (row["verifier_pass"] or not verifier)
                    and float(row["score"]) >= float(cutoff)
                )
                matched_meta = matched_by_key[(table_name, method)]
                table_payload[table_name] = {
                    "outer_train_fitted_target": training_target,
                    "outer_train_fitted_threshold": cutoff,
                    "outer_train_fitted_admitted": training_admitted,
                    "retrospective_matched_coverage_reference_method": "pcecf",
                    "retrospective_matched_coverage_score_threshold": matched_thresholds_by_key[(table_name, method)][str(outer_fold)],
                    "retrospective_matched_coverage_admitted": str(row["record_id"])
                    in matched_selected[table_name][method],
                }
            oof_payload.append(
                {
                    "record_id": str(row["record_id"]),
                    "scene_id": str(row["scene_id"]),
                    "outer_fold": outer_fold,
                    "method": method,
                    "y": str(row["preferred_contract"]),
                    "prediction": str(row["predicted_contract"]),
                    "score": float(row["score"]),
                    "probabilities": row["probabilities"],
                    "common_eligible": bool(row["eligible"]),
                    "native_eligible": bool(row["native_eligible"]),
                    "verifier_outcome": {
                        "admissible": bool(row["verifier_pass"]),
                        "route": str(row["verifier_route"]),
                        "reason": str(row["verifier_reason"]),
                    },
                    "fold_local_concentration": row["fold_local_concentration"],
                    "operating_points": table_payload,
                }
            )
    expected_oof = len(records) * len(METHODS)
    if len(oof_payload) != expected_oof:
        raise AssertionError(f"expected {expected_oof} OOF rows, found {len(oof_payload)}")
    write_jsonl_gz(OUTPUT / "oof_predictions.jsonl.gz", oof_payload)

    duplication = duplication_metrics(
        [record for record in records if "__clean_control__" in str(record["record_id"])],
        folds,
        concentrations,
    )
    table1_by_method = {
        row["method"]: row
        for row in table_results["table1_no_verifier"]["summary"]
    }
    table2_by_method = {
        row["method"]: row
        for row in table_results["table2_shared_verifier"]["summary"]
    }
    clean_by_method = {row["method"]: row for row in clean_rows}
    hierarchical_cautious_reference = protocol["hierarchical_cautious_reference"]
    reference_tolerance = float(hierarchical_cautious_reference["absolute_tolerance"])
    fusion_only_observed = float(
        table1_by_method["hierarchical_cautious_cumulative"][
            "normalized_aurc_point"
        ]
    )
    fusion_only_expected = float(
        hierarchical_cautious_reference["fusion_only_normalized_aurc"]
    )
    shared_verifier_observed = float(
        table2_by_method["hierarchical_cautious_cumulative"][
            "normalized_aurc_point"
        ]
    )
    shared_verifier_expected = float(
        hierarchical_cautious_reference["shared_verifier_normalized_aurc"]
    )
    hierarchical_cautious_reference_checks = {
        "fusion_only_normalized_aurc": {
            "expected": fusion_only_expected,
            "observed": fusion_only_observed,
            "absolute_difference": abs(fusion_only_observed - fusion_only_expected),
            "pass": bool(
                np.isclose(
                    fusion_only_observed,
                    fusion_only_expected,
                    atol=reference_tolerance,
                    rtol=0.0,
                )
            ),
        },
        "shared_verifier_normalized_aurc": {
            "expected": shared_verifier_expected,
            "observed": shared_verifier_observed,
            "absolute_difference": abs(
                shared_verifier_observed - shared_verifier_expected
            ),
            "pass": bool(
                np.isclose(
                    shared_verifier_observed,
                    shared_verifier_expected,
                    atol=reference_tolerance,
                    rtol=0.0,
                )
            ),
        },
        "exact_duplicate_invariance": {
            "expected_max_l1": float(
                hierarchical_cautious_reference["exact_duplicate_max_l1"]
            ),
            "observed_max_l1": float(
                duplication["hierarchical_cautious_exact_duplicate_max_l1"]
            ),
            "expected_max_score_delta": float(
                hierarchical_cautious_reference[
                    "exact_duplicate_max_score_delta"
                ]
            ),
            "observed_max_score_delta": float(
                duplication[
                    "hierarchical_cautious_exact_duplicate_max_score_delta"
                ]
            ),
            "expected_flip_rate": float(
                hierarchical_cautious_reference["exact_duplicate_flip_rate"]
            ),
            "observed_flip_rate": float(
                duplication["hierarchical_cautious_exact_duplicate_flip_rate"]
            ),
            "pass": (
                float(duplication["hierarchical_cautious_exact_duplicate_max_l1"])
                <= float(
                    hierarchical_cautious_reference["exact_duplicate_max_l1"]
                )
                and float(
                    duplication[
                        "hierarchical_cautious_exact_duplicate_max_score_delta"
                    ]
                )
                <= float(
                    hierarchical_cautious_reference[
                        "exact_duplicate_max_score_delta"
                    ]
                )
                and float(
                    duplication[
                        "hierarchical_cautious_exact_duplicate_flip_rate"
                    ]
                )
                <= float(
                    hierarchical_cautious_reference["exact_duplicate_flip_rate"]
                )
            ),
        },
    }
    hierarchical_cautious_reference_pass = all(
        bool(check["pass"])
        for check in hierarchical_cautious_reference_checks.values()
    )
    table1_validation = {
        "exact_duplicate_invariance": (
            duplication["exact_duplicate_max_l1"]
            <= protocol["validation_tolerances"]["table1"]["exact_duplicate_max_l1"]
            and duplication["exact_duplicate_flip_rate"]
            <= protocol["validation_tolerances"]["table1"]["exact_duplicate_flip_rate"]
        ),
        "clean_accuracy_noninferiority": (
            clean_by_method["pcecf"]["scene_macro_accuracy"]
            + protocol["validation_tolerances"]["table1"]["clean_scene_macro_accuracy_margin"]
            >= clean_by_method["nested_evidential_composition"]["scene_macro_accuracy"]
        ),
        "naurc_noninferiority_vs_registered": (
            table1_by_method["pcecf"]["normalized_aurc_point"]
            <= table1_by_method["registered_lineage_pooling"]["normalized_aurc_point"]
            + protocol["validation_tolerances"]["table1"]["naurc_margin_vs_registered_lineage"]
        ),
    }
    table1_validation["pass"] = all(table1_validation.values())
    table2_validation = {
        "naurc_noninferiority_vs_nested": (
            table2_by_method["pcecf"]["normalized_aurc_point"]
            <= table2_by_method["nested_evidential_composition"]["normalized_aurc_point"]
            + protocol["validation_tolerances"]["table2"]["naurc_margin_vs_nested_evidential"]
        ),
        "training_target_wrong_all_noninferiority": (
            float(table2_training["pcecf"]["wrong_all"])
            <= float(table2_training["nested_evidential_composition"]["wrong_all"])
            + protocol["validation_tolerances"]["table2"]["fixed_target_wrong_all_margin"]
        ),
    }
    table2_validation["pass"] = all(table2_validation.values())

    write_csv(OUTPUT / "calibration_all.csv", calibration_rows)
    write_csv(OUTPUT / "calibration_clean.csv", clean_rows)
    status = "PASS" if (
        table1_validation["pass"]
        and table2_validation["pass"]
        and baseline_anchor_pass
        and hierarchical_cautious_reference_pass
        and sensitivity["unchanged_when_no_slot_is_missing"]
    ) else "FAIL"
    result = {
        "status": status,
        "scope": "controlled PC-ECF comparison with and without a shared verifier",
        "records": len(records),
        "scenes": len(scenes),
        "oof_rows": len(oof_payload),
        "headline_eligibility": {
            "definition": "at least two available and structurally valid source slots are present",
            "method_independent": True,
            "native_method_eligibility_used_in_headline_tables": False,
        },
        "threshold_fit_fields": [
            "score",
            "common_eligibility",
            "verifier_admissible_when_required",
            "record_id_tie_break",
        ],
        "threshold_fit_uses_evaluation_labels": False,
        "operating_point_distinction": {
            "training_fitted": (
                "Each outer-fold threshold is fitted on the other scene folds at the "
                "prespecified 0.13 coverage target."
            ),
            "matched_coverage_diagnostic": (
                "Held-out scores are retrospectively ranked within each outer fold to the exact PC-ECF admitted "
                "count without using y; this diagnostic is not a confirmatory comparison."
            ),
        },
        "artifact_hashes": artifact_hashes,
        "baseline_anchor_checks": baseline_anchor_checks,
        "baseline_anchor_scope": (
            "Nested composition only: its native and common eligibility both require at least two available and structurally valid sources. "
            "Other reference anchors used method-specific eligibility rules and are not treated as equality targets."
        ),
        "baseline_anchor_pass": baseline_anchor_pass,
        "hierarchical_cautious_reference_checks": hierarchical_cautious_reference_checks,
        "hierarchical_cautious_reference_pass": hierarchical_cautious_reference_pass,
        "duplication": duplication,
        "paired_scene_bootstrap_deltas": paired_deltas,
        "tables": table_results,
        "matched_coverage_diagnostic": matched_rows,
        "missing_representation_sensitivity": sensitivity,
        "table1_validation": table1_validation,
        "table2_validation": table2_validation,
        "reference_mismatch": None if baseline_anchor_pass else "canonical baseline anchor mismatch",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "output_manifest_sha256.json"
    }
    (OUTPUT / "output_manifest_sha256.json").write_text(
        json.dumps(output_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "baseline_anchor_pass": baseline_anchor_pass,
                "hierarchical_cautious_reference_pass": hierarchical_cautious_reference_pass,
                "hierarchical_cautious_reference_checks": hierarchical_cautious_reference_checks,
                "table1_validation": table1_validation,
                "table2_validation": table2_validation,
                "paired_scene_bootstrap_deltas": paired_deltas,
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
