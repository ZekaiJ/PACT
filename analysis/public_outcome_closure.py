#!/usr/bin/env python3
"""Donor-complete paired outcome readout for frozen public evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.topology_interventions import nested, pact, product


ARMS = ("native", "registered_copy_m8", "false_split_m8", "all_view_merge")
METHODS = ("PACT", "product", "nested")
METRICS = (
    "ncsAURC",
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece_10bin",
    "risk_at_0p10",
    "posterior_l1_from_native",
    "budget_ratio",
    "top_class_flip_rate",
)
LOSS_METRICS = {
    "ncsAURC",
    "nll",
    "brier",
    "ece_10bin",
    "risk_at_0p10",
    "posterior_l1_from_native",
    "budget_ratio",
    "top_class_flip_rate",
}
CONTRASTS = {
    "conservation": ("native", "registered_copy_m8"),
    "provenance_error": ("registered_copy_m8", "false_split_m8"),
    "coarsening_boundary": ("native", "all_view_merge"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        tp = np.sum((y == label) & (pred == label))
        fp = np.sum((y != label) & (pred == label))
        fn = np.sum((y == label) & (pred != label))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else float(2 * tp / denominator))
    return float(np.mean(values))


def fractional_risk(scores: np.ndarray, wrong: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)[::-1]
    totals = np.asarray([np.sum(scores == value) for value in unique], dtype=np.float64)
    errors = np.asarray(
        [np.sum((scores == value) & wrong) for value in unique], dtype=np.float64
    )
    cumulative_total = np.cumsum(totals)
    cumulative_errors = np.cumsum(errors)
    result = np.empty(len(coverages), dtype=np.float64)
    for index, coverage in enumerate(coverages):
        target = float(coverage * len(scores))
        group = int(np.searchsorted(cumulative_total, target, side="left"))
        before_total = cumulative_total[group - 1] if group else 0.0
        before_errors = cumulative_errors[group - 1] if group else 0.0
        fraction = (target - before_total) / totals[group]
        result[index] = (before_errors + fraction * errors[group]) / target
    return result


def old_full_support_ncs(scores: np.ndarray, wrong: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    grid = np.linspace(max(0.01, coverage[0]), 1.0, 101)
    return float(np.trapezoid(np.interp(grid, coverage, risk), grid) / (grid[-1] - grid[0]))


def metric_bundle(
    posterior: np.ndarray,
    score: np.ndarray,
    y: np.ndarray,
    support: np.ndarray,
    native_posterior: np.ndarray,
    budget: np.ndarray | None,
    native_budget: np.ndarray | None,
) -> dict[str, float]:
    prediction = posterior.argmax(axis=1)
    wrong = prediction != y
    one_hot = np.eye(posterior.shape[1], dtype=np.float64)[y]
    confidence = posterior.max(axis=1)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(prediction[mask] == y[mask])) - float(np.mean(confidence[mask]))
            )
    risk = fractional_risk(score, wrong, np.concatenate((support, np.asarray([0.10]))))
    return {
        "ncsAURC": float(np.trapezoid(risk[:-1], support) / (support[-1] - support[0])),
        "accuracy": float(np.mean(prediction == y)),
        "macro_f1": macro_f1(y, prediction, posterior.shape[1]),
        "nll": float(-np.mean(np.log(np.clip(posterior[np.arange(len(y)), y], 1e-12, 1.0)))),
        "brier": float(np.mean(np.sum(np.square(posterior - one_hot), axis=1))),
        "ece_10bin": float(ece),
        "risk_at_0p10": float(risk[-1]),
        "posterior_l1_from_native": float(np.mean(np.sum(np.abs(posterior - native_posterior), axis=1))),
        "budget_ratio": float("nan") if budget is None else float(np.sum(budget) / np.sum(native_budget)),
        "top_class_flip_rate": float(np.mean(prediction != native_posterior.argmax(axis=1))),
    }


def load_rows(path: Path, backbone: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["backbone"] == backbone and row["split"] == "test":
                rows.append(row)
    if not rows:
        raise ValueError(f"no test rows for {backbone}")
    return rows


def build_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = np.asarray(sorted({int(row["seed"]) for row in rows}), dtype=int)
    record_ids = np.asarray(sorted({int(row["record_id"]) for row in rows}), dtype=int)
    by_key = {(int(row["seed"]), int(row["record_id"])): row for row in rows}
    if len(by_key) != len(seeds) * len(record_ids):
        raise AssertionError("every frozen seed must cover the same record identifiers")
    labels = np.asarray([int(by_key[(int(seeds[0]), int(rid))]["y"]) for rid in record_ids])
    for seed in seeds:
        observed = np.asarray([int(by_key[(int(seed), int(rid))]["y"]) for rid in record_ids])
        if not np.array_equal(observed, labels):
            raise AssertionError("record labels differ across frozen seeds")
    views = len(by_key[(int(seeds[0]), int(record_ids[0]))]["evidences"])
    classes = len(by_key[(int(seeds[0]), int(record_ids[0]))]["evidences"][0])
    shape = (len(seeds), views, len(ARMS), len(record_ids), classes)
    posterior = {method: np.empty(shape, dtype=np.float64) for method in METHODS}
    score = {method: np.empty(shape[:-1], dtype=np.float64) for method in METHODS}
    budget = np.empty(shape[:-1], dtype=np.float64)
    components = np.empty(shape[:-1], dtype=np.int16)
    arm_index = {arm: index for index, arm in enumerate(ARMS)}

    for seed_index, seed in enumerate(seeds):
        for record_index, record_id in enumerate(record_ids):
            row = by_key[(int(seed), int(record_id))]
            native = [np.asarray(values, dtype=np.float64) for values in row["evidences"]]
            distinct = [f"native-view:{index}" for index in range(views)]
            native_outputs = {
                "PACT": pact(native, distinct),
                "product": product(native),
                "nested": nested(native),
            }
            merge_output = pact(native, ["all-view-merge"] * views)
            for donor in range(views):
                for method, output in native_outputs.items():
                    posterior[method][seed_index, donor, arm_index["native"], record_index] = output["posterior"]
                    score[method][seed_index, donor, arm_index["native"], record_index] = output["score"]
                posterior["PACT"][seed_index, donor, arm_index["all_view_merge"], record_index] = merge_output["posterior"]
                score["PACT"][seed_index, donor, arm_index["all_view_merge"], record_index] = merge_output["score"]
                for method in ("product", "nested"):
                    posterior[method][seed_index, donor, arm_index["all_view_merge"], record_index] = native_outputs[method]["posterior"]
                    score[method][seed_index, donor, arm_index["all_view_merge"], record_index] = native_outputs[method]["score"]

                exact = native + [native[donor].copy() for _ in range(7)]
                registered_parents = distinct + [distinct[donor]] * 7
                split_parents = distinct + [f"false-split:{donor}:{copy}" for copy in range(7)]
                registered = pact(exact, registered_parents)
                split = pact(exact, split_parents)
                blind = {"product": product(exact), "nested": nested(exact)}
                for arm, output in (("registered_copy_m8", registered), ("false_split_m8", split)):
                    posterior["PACT"][seed_index, donor, arm_index[arm], record_index] = output["posterior"]
                    score["PACT"][seed_index, donor, arm_index[arm], record_index] = output["score"]
                for method, output in blind.items():
                    for arm in ("registered_copy_m8", "false_split_m8"):
                        posterior[method][seed_index, donor, arm_index[arm], record_index] = output["posterior"]
                        score[method][seed_index, donor, arm_index[arm], record_index] = output["score"]

                pact_outputs = {
                    "native": native_outputs["PACT"],
                    "registered_copy_m8": registered,
                    "false_split_m8": split,
                    "all_view_merge": merge_output,
                }
                for arm, output in pact_outputs.items():
                    budget[seed_index, donor, arm_index[arm], record_index] = output["evidence_budget"]
                    components[seed_index, donor, arm_index[arm], record_index] = output["component_count"]

    return {
        "seeds": seeds,
        "record_ids": record_ids,
        "labels": labels,
        "views": views,
        "classes": classes,
        "posterior": posterior,
        "score": score,
        "budget": budget,
        "components": components,
    }


def tie_signature(scores: np.ndarray, record_ids: np.ndarray) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: dict[str, list[int]] = defaultdict(list)
    for score, record_id in zip(scores, record_ids, strict=True):
        groups[float(score).hex()].append(int(record_id))
    return tuple(sorted((value, tuple(sorted(ids))) for value, ids in groups.items()))


def support_for_pair(data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    a1 = ARMS.index("registered_copy_m8")
    a2 = ARMS.index("false_split_m8")
    minima = []
    for seed in range(len(data["seeds"])):
        for donor in range(data["views"]):
            for arm in (a1, a2):
                scores = data["score"]["PACT"][seed, donor, arm]
                minima.append(float(np.mean(scores == np.max(scores))))
    low = max(0.10, max(minima))
    high = 0.90
    if not low < high:
        raise AssertionError(f"empty common support: [{low}, {high}]")
    return np.linspace(low, high, 36), {
        "gamma_min": low,
        "gamma_max": high,
        "grid_points": 36,
        "largest_top_tie_coverage": max(minima),
        "selection_uses_labels": False,
        "tie_rule": "analytic fractional inclusion within equal-score group",
    }


def invariance_gate(data: dict[str, Any], support: np.ndarray, tolerance: float) -> dict[str, Any]:
    native = ARMS.index("native")
    registered = ARMS.index("registered_copy_m8")
    posterior_residual = float(
        np.max(np.abs(data["posterior"]["PACT"][:, :, registered] - data["posterior"]["PACT"][:, :, native]))
    )
    score_residual = float(
        np.max(np.abs(data["score"]["PACT"][:, :, registered] - data["score"]["PACT"][:, :, native]))
    )
    budget_residual = float(np.max(np.abs(data["budget"][:, :, registered] - data["budget"][:, :, native])))
    prediction_equal = bool(
        np.array_equal(
            data["posterior"]["PACT"][:, :, registered].argmax(axis=-1),
            data["posterior"]["PACT"][:, :, native].argmax(axis=-1),
        )
    )
    tie_equal = True
    metric_residual = 0.0
    labels = data["labels"]
    for seed in range(len(data["seeds"])):
        for donor in range(data["views"]):
            tie_equal &= tie_signature(
                data["score"]["PACT"][seed, donor, native], data["record_ids"]
            ) == tie_signature(data["score"]["PACT"][seed, donor, registered], data["record_ids"])
            bundles = []
            for arm in (native, registered):
                bundles.append(
                    metric_bundle(
                        data["posterior"]["PACT"][seed, donor, arm],
                        data["score"]["PACT"][seed, donor, arm],
                        labels,
                        support,
                        data["posterior"]["PACT"][seed, donor, native],
                        data["budget"][seed, donor, arm],
                        data["budget"][seed, donor, native],
                    )
                )
            metric_residual = max(
                metric_residual,
                max(abs(bundles[0][metric] - bundles[1][metric]) for metric in METRICS),
            )
    blind_equal = True
    for method in ("product", "nested"):
        blind_equal &= bool(
            np.array_equal(
                data["posterior"][method][:, :, registered],
                data["posterior"][method][:, :, ARMS.index("false_split_m8")],
            )
        )
        blind_equal &= bool(
            np.array_equal(
                data["score"][method][:, :, registered],
                data["score"][method][:, :, ARMS.index("false_split_m8")],
            )
        )
    checks = {
        "posterior": posterior_residual <= tolerance,
        "predicted_class": prediction_equal,
        "non_vacuity": score_residual <= tolerance,
        "budget": budget_residual <= tolerance,
        "tie_groups": bool(tie_equal),
        "outcomes": metric_residual <= tolerance,
        "registration_blind_A1_A2": bool(blind_equal),
    }
    return {
        "pass": all(checks.values()),
        "tolerance": tolerance,
        "checks": checks,
        "max_posterior_residual": posterior_residual,
        "max_score_residual": score_residual,
        "max_budget_residual": budget_residual,
        "max_outcome_residual": metric_residual,
    }


def point_metrics(data: dict[str, Any], support: np.ndarray) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float]]:
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    values: dict[tuple[str, str, str], list[list[float]]] = {}
    for method in METHODS:
        for arm in ARMS:
            for metric in METRICS:
                values[(method, arm, metric)] = []
    for seed in range(len(data["seeds"])):
        per_seed = defaultdict(list)
        for donor in range(data["views"]):
            for method in METHODS:
                native_posterior = data["posterior"][method][seed, donor, arm_index["native"]]
                for arm in ARMS:
                    index = arm_index[arm]
                    bundle = metric_bundle(
                        data["posterior"][method][seed, donor, index],
                        data["score"][method][seed, donor, index],
                        data["labels"],
                        support,
                        native_posterior,
                        data["budget"][seed, donor, index] if method == "PACT" else None,
                        data["budget"][seed, donor, arm_index["native"]] if method == "PACT" else None,
                    )
                    for metric, value in bundle.items():
                        per_seed[(method, arm, metric)].append(value)
        for key, donor_values in per_seed.items():
            values[key].append([float(np.mean(donor_values)), *map(float, donor_values)])
    rows = []
    lookup = {}
    for (method, arm, metric), seed_rows in values.items():
        donor_macro_by_seed = np.asarray([row[0] for row in seed_rows], dtype=np.float64)
        point = float(np.mean(donor_macro_by_seed))
        lookup[(method, arm, metric)] = point
        rows.append(
            {
                "method": method,
                "arm": arm,
                "metric": metric,
                "donor_macro": point,
                "seed_sd": float(np.std(donor_macro_by_seed, ddof=1)),
                "seed_min": float(np.min(donor_macro_by_seed)),
                "seed_max": float(np.max(donor_macro_by_seed)),
                "seeds": len(data["seeds"]),
                "donors": data["views"],
                "records": len(data["record_ids"]),
            }
        )
    return rows, lookup


def sampled_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        pieces.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(pieces)


def point_contrasts(data: dict[str, Any], support: np.ndarray) -> dict[tuple[str, str, str], float]:
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    per_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for seed in range(len(data["seeds"])):
        donor_bundles: dict[str, list[dict[str, float]]] = defaultdict(list)
        for donor in range(data["views"]):
            native_posterior = data["posterior"]["PACT"][seed, donor, arm_index["native"]]
            for arm in ARMS:
                index = arm_index[arm]
                donor_bundles[arm].append(
                    metric_bundle(
                        data["posterior"]["PACT"][seed, donor, index],
                        data["score"]["PACT"][seed, donor, index],
                        data["labels"],
                        support,
                        native_posterior,
                        data["budget"][seed, donor, index],
                        data["budget"][seed, donor, arm_index["native"]],
                    )
                )
        for contrast, (left, right) in CONTRASTS.items():
            for metric in METRICS:
                donor_deltas = np.asarray(
                    [
                        donor_bundles[right][donor][metric] - donor_bundles[left][donor][metric]
                        for donor in range(data["views"])
                    ],
                    dtype=np.float64,
                )
                per_seed[(contrast, metric, "donor_macro")].append(float(np.mean(donor_deltas)))
                right_values = np.asarray(
                    [donor_bundles[right][donor][metric] for donor in range(data["views"])],
                    dtype=np.float64,
                )
                if metric == "budget_ratio" and contrast == "coarsening_boundary":
                    worst = int(np.argmin(right_values))
                else:
                    worst = int(np.argmax(right_values) if metric in LOSS_METRICS else np.argmin(right_values))
                per_seed[(contrast, metric, "worst_donor")].append(float(donor_deltas[worst]))
    return {key: float(np.mean(values)) for key, values in per_seed.items()}

def bootstrap(data: dict[str, Any], support: np.ndarray, draws: int, seed_value: int) -> list[dict[str, Any]]:
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    rng = np.random.default_rng(seed_value)
    samples = {
        (contrast, metric, aggregation): np.empty(draws, dtype=np.float64)
        for contrast in CONTRASTS
        for metric in METRICS
        for aggregation in ("donor_macro", "worst_donor")
    }
    for replicate in range(draws):
        selected = sampled_indices(data["labels"], rng)
        y = data["labels"][selected]
        per_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for seed in range(len(data["seeds"])):
            donor_bundles: dict[str, list[dict[str, float]]] = defaultdict(list)
            for donor in range(data["views"]):
                native_posterior = data["posterior"]["PACT"][seed, donor, arm_index["native"], selected]
                for arm in ARMS:
                    index = arm_index[arm]
                    donor_bundles[arm].append(
                        metric_bundle(
                            data["posterior"]["PACT"][seed, donor, index, selected],
                            data["score"]["PACT"][seed, donor, index, selected],
                            y,
                            support,
                            native_posterior,
                            data["budget"][seed, donor, index, selected],
                            data["budget"][seed, donor, arm_index["native"], selected],
                        )
                    )
            for contrast, (left, right) in CONTRASTS.items():
                for metric in METRICS:
                    donor_contrasts = np.asarray(
                        [
                            donor_bundles[right][donor][metric] - donor_bundles[left][donor][metric]
                            for donor in range(data["views"])
                        ],
                        dtype=np.float64,
                    )
                    per_seed[(contrast, metric, "donor_macro")].append(float(np.mean(donor_contrasts)))
                    right_values = np.asarray(
                        [donor_bundles[right][donor][metric] for donor in range(data["views"])],
                        dtype=np.float64,
                    )
                    if metric == "budget_ratio" and contrast == "coarsening_boundary":
                        worst = int(np.argmin(right_values))
                    else:
                        worst = int(np.argmax(right_values) if metric in LOSS_METRICS else np.argmin(right_values))
                    per_seed[(contrast, metric, "worst_donor")].append(float(donor_contrasts[worst]))
        for key, seed_values in per_seed.items():
            samples[key][replicate] = float(np.mean(seed_values))
    rows = []
    for (contrast, metric, aggregation), values in samples.items():
        rows.append(
            {
                "contrast": contrast,
                "left_arm": CONTRASTS[contrast][0],
                "right_arm": CONTRASTS[contrast][1],
                "metric": metric,
                "aggregation": aggregation,
                "bootstrap_mean": float(np.mean(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "draws": draws,
                "bootstrap_seed": seed_value,
            }
        )
    return rows


def legacy_reproduction(data: dict[str, Any], released: Path, tolerance: float) -> dict[str, Any]:
    released_rows = json.loads(released.read_text(encoding="utf-8"))
    published = {(row["method"], row["arm"]): row for row in released_rows}
    arm_names = {
        "native": "A0_native",
        "registered_copy_m8": "A1_conserved_exact_m8",
        "false_split_m8": "A2_false_split_m8",
        "all_view_merge": "A3_false_merge",
    }
    residuals = []
    donor_for_record = data["record_ids"] % data["views"]
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    for method in METHODS:
        for arm in ARMS:
            posteriors = []
            scores = []
            labels = []
            budgets = []
            components = []
            drifts = []
            score_drifts = []
            for seed in range(len(data["seeds"])):
                selected_posterior = np.asarray(
                    [
                        data["posterior"][method][seed, donor_for_record[index], arm_index[arm], index]
                        for index in range(len(data["record_ids"]))
                    ]
                )
                selected_score = np.asarray(
                    [
                        data["score"][method][seed, donor_for_record[index], arm_index[arm], index]
                        for index in range(len(data["record_ids"]))
                    ]
                )
                posteriors.append(selected_posterior)
                scores.append(selected_score)
                labels.append(data["labels"])
                if method == "PACT":
                    native_posterior = np.asarray(
                        [
                            data["posterior"][method][seed, donor_for_record[index], arm_index["native"], index]
                            for index in range(len(data["record_ids"]))
                        ]
                    )
                    native_score = np.asarray(
                        [
                            data["score"][method][seed, donor_for_record[index], arm_index["native"], index]
                            for index in range(len(data["record_ids"]))
                        ]
                    )
                    selected_budget = np.asarray(
                        [
                            data["budget"][seed, donor_for_record[index], arm_index[arm], index]
                            for index in range(len(data["record_ids"]))
                        ]
                    )
                    selected_components = np.asarray(
                        [
                            data["components"][seed, donor_for_record[index], arm_index[arm], index]
                            for index in range(len(data["record_ids"]))
                        ]
                    )
                    budgets.append(selected_budget)
                    components.append(selected_components)
                    drifts.append(np.sum(np.abs(selected_posterior - native_posterior), axis=1))
                    score_drifts.append(selected_score - native_score)
            posterior = np.concatenate(posteriors)
            score = np.concatenate(scores)
            y = np.concatenate(labels)
            pred = posterior.argmax(axis=1)
            confidence = posterior.max(axis=1)
            one_hot = np.eye(posterior.shape[1])[y]
            ece = 0.0
            for lower in np.linspace(0.0, 0.9, 10):
                mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
                if np.any(mask):
                    ece += float(np.mean(mask)) * abs(float(np.mean(pred[mask] == y[mask])) - float(np.mean(confidence[mask])))
            observed = {
                "accuracy": float(np.mean(pred == y)),
                "macro_f1": macro_f1(y, pred, posterior.shape[1]),
                "nll": float(-np.mean(np.log(np.clip(posterior[np.arange(len(y)), y], 1e-12, 1.0)))),
                "brier": float(np.mean(np.sum(np.square(posterior - one_hot), axis=1))),
                "ece_10bin": float(ece),
                "ncs_aurc_full_support": old_full_support_ncs(score, pred != y),
            }
            if method == "PACT":
                observed |= {
                    "mean_component_count": float(np.mean(np.concatenate(components))),
                    "mean_evidence_budget": float(np.mean(np.concatenate(budgets))),
                    "mean_posterior_l1_from_A0": float(np.mean(np.concatenate(drifts))),
                    "mean_score_drift_from_A0": float(np.mean(np.concatenate(score_drifts))),
                }
            expected = published[(method, arm_names[arm])]
            for metric, value in observed.items():
                residuals.append(
                    {
                        "method": method,
                        "arm": arm,
                        "metric": metric,
                        "observed": value,
                        "released": float(expected[metric]),
                        "absolute_residual": abs(value - float(expected[metric])),
                    }
                )
    return {
        "pass": max(row["absolute_residual"] for row in residuals) <= tolerance,
        "tolerance": tolerance,
        "max_absolute_residual": max(row["absolute_residual"] for row in residuals),
        "rows": residuals,
    }


def self_test() -> None:
    scores = np.asarray([0.9, 0.9, 0.2, 0.1])
    wrong = np.asarray([False, True, False, True])
    risk = fractional_risk(scores, wrong, np.asarray([0.25, 0.50, 1.00]))
    assert np.allclose(risk, [0.5, 0.5, 0.5])
    labels = np.asarray([0, 0, 1, 1, 1, 2])
    draw = sampled_indices(labels, np.random.default_rng(7))
    assert len(draw) == len(labels)
    assert sorted(np.bincount(labels[draw], minlength=3)) == sorted(np.bincount(labels, minlength=3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    if sha256(args.input) != "2245CD2B272542F0AA1BD0E4DAFF94DC962FCBE8B6B620247D5828E83D8EAB98":
        raise AssertionError("frozen evidence hash mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    pair_specs = (
        ("HandWritten-Mfeat__TMC", "TMC"),
        ("HandWritten-Mfeat__RCML", "RCML"),
    )
    all_points = []
    all_bootstrap = []
    gates = {}
    supports = {}
    legacy = {}
    for pair_offset, (pair_id, backbone) in enumerate(pair_specs):
        data = build_arrays(load_rows(args.input, backbone))
        support, support_info = support_for_pair(data)
        gate = invariance_gate(data, support, 1e-12)
        gates[pair_id] = gate
        supports[pair_id] = support_info
        if not gate["pass"]:
            write_json(args.output / "INVARIANCE_GATE.json", gates)
            raise AssertionError(f"registered-copy invariance failed for {pair_id}")
        points, lookup = point_metrics(data, support)
        contrast_points = point_contrasts(data, support)
        for row in points:
            row["pair_id"] = pair_id
            row["support_low"] = support_info["gamma_min"]
            row["support_high"] = support_info["gamma_max"]
            all_points.append(row)
        boot = bootstrap(data, support, args.draws, 20260731 + pair_offset)
        for row in boot:
            row["pair_id"] = pair_id
            row["point"] = contrast_points[
                (row["contrast"], row["metric"], row["aggregation"])
            ]
            row["support_low"] = support_info["gamma_min"]
            row["support_high"] = support_info["gamma_max"]
            all_bootstrap.append(row)
        released = Path(__file__).resolve().parents[1] / "results" / "extended_evidence" / "n3_topology" / pair_id / "RESULTS.json"
        legacy[pair_id] = legacy_reproduction(data, released, 1e-12)
        if not legacy[pair_id]["pass"]:
            write_json(args.output / "LEGACY_REPRODUCTION.json", legacy)
            raise AssertionError(f"legacy point estimates do not reproduce for {pair_id}")

    write_csv(args.output / "PAIR_OUTCOMES.csv", all_points)
    write_csv(args.output / "PAIRED_CONTRASTS.csv", all_bootstrap)
    write_json(args.output / "PAIR_SUPPORT.json", supports)
    write_json(args.output / "INVARIANCE_GATE.json", gates)
    write_json(args.output / "LEGACY_REPRODUCTION.json", legacy)
    write_json(
        args.output / "BLOCKED_INPUTS.json",
        {
            "PIE__RCML": {
                "status": "BLOCKED_INPUT_NOT_RECOVERED",
                "reason": "Frozen donor-level per-view evidence is unavailable; aggregate rows cannot reconstruct the intervention and emitter retraining is prohibited by the lock."
            }
        },
    )
    primary = [
        row
        for row in all_bootstrap
        if row["contrast"] == "provenance_error"
        and row["metric"] in {"ncsAURC", "posterior_l1_from_native"}
        and row["aggregation"] == "donor_macro"
    ]
    write_json(
        args.output / "PRIMARY_RESULTS.json",
        {
            "status": "COMPLETE_FOR_TWO_RECOVERABLE_PAIRS",
            "primary_and_mechanistic_results": primary,
            "cross_pair_pooling_performed": False,
            "null_or_reversed_results_retained": True,
            "manuscript_modified": False,
        },
    )
    manifest = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "MANIFEST.sha256":
            manifest.append(f"{sha256(path)}  {path.name}")
    (args.output / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
