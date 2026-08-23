"""Reproduce the frozen P0 estimand-closure analyses.

This runner does not modify the manuscript or any shared release metadata.  It
writes only versioned results under ``results/p0_estimand_closure/v1``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "p0_estimand_closure" / "v1"
OOF = ROOT / "outputs" / "pcecf_study" / "oof_predictions.jsonl.gz"
FOLD_THRESHOLDS = ROOT / "outputs" / "pcecf_study" / "fold_thresholds.jsonl.gz"
CONTROLLED_DATA = ROOT / "data" / "controlled" / "source_records.jsonl.gz"
CONTROLLED_LABELS = ROOT / "data" / "controlled" / "evaluation_labels.jsonl.gz"
CONTROLLED_CONFIG = ROOT / "configs" / "controlled_study.json"
PCECF_CONFIG = ROOT / "configs" / "pcecf_study.json"
FM_ROOT = ROOT / "results" / "balanced_fm_panel"
FM_CASES = FM_ROOT / "protocol" / "FM_PANEL_FULL_CASES.csv"
FM_INPUTS = FM_ROOT / "outputs"
FM_ANALYSIS = FM_ROOT / "protocol" / "analyze_fm_panel.py"
FM_PUBLISHED = FM_ROOT / "analysis" / "FUSION_METRICS.csv"

VERSION = "p0-estimand-closure-v1"
CONTROLLED_METHODS = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
    "pcecf",
)
CONTROLLED_TARGETS = np.asarray([index / 100.0 for index in range(1, 61)])
CONTROLLED_SUPPORT = np.linspace(0.10, 0.35, 36)
CONTROLLED_BOOTSTRAP_REPLICATES = 2000
CONTROLLED_BOOTSTRAP_SEED = 1886
CONTROLLED_CONTRASTS = (
    (
        "fusion_no_verifier",
        "pcecf",
        "product_evidence_fusion",
    ),
    (
        "fusion_no_verifier",
        "pcecf",
        "registered_lineage_pooling",
    ),
    (
        "shared_verifier",
        "pcecf",
        "nested_evidential_composition",
    ),
)

REPEATED_SPLITS = 50
REPEATED_BASE_SEED = 1930
FOLD_COUNT = 5
POLICY_TARGETS = np.asarray((0.10, 0.13, 0.15))
CONCENTRATIONS = (4.0, 8.0, 12.0, 16.0, 24.0)
BASE_CANDIDATES = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
)
NESTED_CANDIDATES = tuple(
    f"nested_evidential_composition_c{int(value)}" for value in CONCENTRATIONS
)
SELECTION_CANDIDATES = BASE_CANDIDATES + NESTED_CANDIDATES

FM_FULL_SUPPORT = np.linspace(0.10, 0.90, 36)
FM_COMPARABILITY_SUPPORT = np.linspace(0.10, 0.39, 36)
FM_BOOTSTRAP_REPLICATES = 2000
FM_BOOTSTRAP_SEED = 5401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p025": float(np.quantile(array, 0.025)),
        "p975": float(np.quantile(array, 0.975)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def direction(point: float, left: str, right: str) -> str:
    if point < 0.0:
        return f"favors_{left}"
    if point > 0.0:
        return f"favors_{right}"
    return "tie"


def grouped_ncs_aurc(
    coverage: Sequence[float],
    risk: Sequence[float],
    support: np.ndarray,
) -> float:
    grouped: dict[float, list[float]] = defaultdict(list)
    for x, y in zip(coverage, risk, strict=True):
        grouped[float(x)].append(float(y))
    x = np.asarray(sorted(grouped), dtype=np.float64)
    y = np.asarray([np.mean(grouped[value]) for value in x], dtype=np.float64)
    tolerance = 1e-12
    if x[0] > support[0] + tolerance or x[-1] < support[-1] - tolerance:
        raise ValueError(
            f"curve support [{x[0]}, {x[-1]}] misses "
            f"[{support[0]}, {support[-1]}]"
        )
    interpolated = np.interp(support, x, y)
    return float(np.trapezoid(interpolated, support) / (support[-1] - support[0]))


def thresholds(
    score: np.ndarray,
    available: np.ndarray,
    train: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    values = np.sort(score[available & train])[::-1]
    counts = np.rint(targets * int(np.sum(train))).astype(int)
    result = np.full(len(targets), np.inf, dtype=np.float64)
    positive = counts > 0
    if len(values) and np.any(positive):
        positions = np.minimum(counts[positive], len(values)) - 1
        result[positive] = values[positions]
    return result


def load_controlled_predictions() -> tuple[
    dict[str, dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(OOF, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            verifier = raw["verifier_outcome"]
            rows[str(raw["method"])].append(
                {
                    "record_id": str(raw["record_id"]),
                    "scene_id": str(raw["scene_id"]),
                    "fold": int(raw["outer_fold"]),
                    "score": float(raw["score"]),
                    "common": bool(raw["common_eligible"]),
                    "native": bool(raw["native_eligible"]),
                    "verifier": bool(verifier["admissible"]),
                    "wrong": str(raw["prediction"]) != str(raw["y"]),
                    "truth": str(raw["y"]),
                    "concentration": raw.get("fold_local_concentration"),
                }
            )
    if tuple(rows) != CONTROLLED_METHODS:
        raise AssertionError(f"unexpected controlled method order: {tuple(rows)}")
    ordered = {
        method: sorted(values, key=lambda row: row["record_id"])
        for method, values in rows.items()
    }
    record_ids = tuple(row["record_id"] for row in ordered["pcecf"])
    if len(record_ids) != 31_200 or len(set(record_ids)) != len(record_ids):
        raise AssertionError("controlled OOF denominator is not 31,200 unique records")
    for method in CONTROLLED_METHODS:
        if tuple(row["record_id"] for row in ordered[method]) != record_ids:
            raise AssertionError(f"record alignment differs for {method}")
    scene_names = tuple(sorted({row["scene_id"] for row in ordered["pcecf"]}))
    if len(scene_names) != 48:
        raise AssertionError(f"expected 48 scenes, found {len(scene_names)}")
    scene_lookup = {scene: index for index, scene in enumerate(scene_names)}
    scene_index = np.asarray(
        [scene_lookup[row["scene_id"]] for row in ordered["pcecf"]], dtype=np.int8
    )
    truth = np.asarray([row["truth"] for row in ordered["pcecf"]], dtype=object)
    result: dict[str, dict[str, np.ndarray]] = {}
    for method, values in ordered.items():
        result[method] = {
            "score": np.asarray([row["score"] for row in values], dtype=np.float64),
            "common": np.asarray([row["common"] for row in values], dtype=bool),
            "native": np.asarray([row["native"] for row in values], dtype=bool),
            "verifier": np.asarray([row["verifier"] for row in values], dtype=bool),
            "wrong": np.asarray([row["wrong"] for row in values], dtype=bool),
            "fold": np.asarray([row["fold"] for row in values], dtype=np.int8),
            "concentration": np.asarray(
                [
                    float(row["concentration"])
                    if row["concentration"] is not None
                    else np.nan
                    for row in values
                ],
                dtype=np.float64,
            ),
        }
    return result, scene_index, truth, record_ids


def load_frozen_thresholds() -> dict[tuple[str, str, int, float], float]:
    result: dict[tuple[str, str, int, float], float] = {}
    with gzip.open(FOLD_THRESHOLDS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("protocol_role") != "primary":
                continue
            key = (
                str(row["table"]),
                str(row["method"]),
                int(row["outer_fold"]),
                round(float(row["target_coverage"]), 2),
            )
            threshold = row["threshold"]
            result[key] = float(threshold) if threshold is not None else float("inf")
    expected = 2 * len(CONTROLLED_METHODS) * FOLD_COUNT * len(CONTROLLED_TARGETS)
    if len(result) != expected:
        raise AssertionError(
            f"expected {expected} frozen primary thresholds, found {len(result)}"
        )
    return result


def controlled_counts(
    methods: Mapping[str, Mapping[str, np.ndarray]],
    scene_index: np.ndarray,
    frozen_thresholds: Mapping[tuple[str, str, int, float], float],
    *,
    source_table: str,
    verifier: bool,
    accepted_count: bool,
) -> np.ndarray:
    fields = 3  # n, admitted, wrong
    counts = np.zeros(
        (len(CONTROLLED_METHODS), len(CONTROLLED_TARGETS), 48, fields),
        dtype=np.float64,
    )
    for method_index, method in enumerate(CONTROLLED_METHODS):
        data = methods[method]
        available = data["common"] & (data["verifier"] if verifier else True)
        for fold in range(FOLD_COUNT):
            test = data["fold"] == fold
            train = ~test
            n_by_scene = np.bincount(scene_index[test], minlength=48)
            counts[method_index, :, :, 0] += n_by_scene[None, :]
            if accepted_count:
                candidates = np.flatnonzero(test & available)
                order = np.lexsort((candidates, -data["score"][candidates]))
                ranked = candidates[order]
                requested = np.rint(CONTROLLED_TARGETS * int(np.sum(test))).astype(int)
                for target_index, count in enumerate(requested):
                    selected = ranked[: min(int(count), len(ranked))]
                    counts[method_index, target_index, :, 1] += np.bincount(
                        scene_index[selected], minlength=48
                    )
                    counts[method_index, target_index, :, 2] += np.bincount(
                        scene_index[selected],
                        weights=data["wrong"][selected],
                        minlength=48,
                    )
            else:
                cutoff = np.asarray(
                    [
                        frozen_thresholds[
                            (source_table, method, fold, round(float(target), 2))
                        ]
                        for target in CONTROLLED_TARGETS
                    ],
                    dtype=np.float64,
                )
                test_indices = np.flatnonzero(test)
                admitted = (
                    available[test_indices, None]
                    & (data["score"][test_indices, None] >= cutoff[None, :])
                )
                wrong = admitted & data["wrong"][test_indices, None]
                for target_index in range(len(CONTROLLED_TARGETS)):
                    counts[method_index, target_index, :, 1] += np.bincount(
                        scene_index[test_indices],
                        weights=admitted[:, target_index],
                        minlength=48,
                    )
                    counts[method_index, target_index, :, 2] += np.bincount(
                        scene_index[test_indices],
                        weights=wrong[:, target_index],
                        minlength=48,
                    )
    return counts


def ncs_from_totals(totals: np.ndarray, support: np.ndarray) -> float:
    n = totals[:, 0]
    admitted = totals[:, 1]
    wrong = totals[:, 2]
    coverage = np.divide(admitted, n, out=np.zeros_like(admitted), where=n > 0)
    risk = np.divide(
        wrong, admitted, out=np.zeros_like(wrong), where=admitted > 0
    )
    return grouped_ncs_aurc(coverage, risk, support)


def controlled_bootstrap(
    counts: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(
        [ncs_from_totals(value.sum(axis=1), CONTROLLED_SUPPORT) for value in counts]
    )
    totals = np.einsum("rs,mtsf->rmtf", weights, counts, optimize=True)
    draws = np.empty((len(weights), len(CONTROLLED_METHODS)), dtype=np.float64)
    for replicate in range(len(weights)):
        for method_index in range(len(CONTROLLED_METHODS)):
            draws[replicate, method_index] = ncs_from_totals(
                totals[replicate, method_index], CONTROLLED_SUPPORT
            )
    return point, draws


def run_controlled(
    methods: Mapping[str, Mapping[str, np.ndarray]],
    scene_index: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(CONTROLLED_BOOTSTRAP_SEED)
    weights = rng.multinomial(
        48,
        np.full(48, 1.0 / 48.0),
        size=CONTROLLED_BOOTSTRAP_REPLICATES,
    )
    estimate_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    count_tables: dict[str, np.ndarray] = {}
    frozen_thresholds = load_frozen_thresholds()
    for table, source_table, verifier in (
        ("fusion_no_verifier", "table1_no_verifier", False),
        ("shared_verifier", "table2_shared_verifier", True),
    ):
        for estimator, accepted_count in (
            ("outer_train_threshold", False),
            ("heldout_exact_count_diagnostic", True),
        ):
            counts = controlled_counts(
                methods,
                scene_index,
                frozen_thresholds,
                source_table=source_table,
                verifier=verifier,
                accepted_count=accepted_count,
            )
            point, draws = controlled_bootstrap(counts, weights)
            result[(table, estimator)] = point, draws
            if accepted_count:
                count_tables[table] = counts
            for method_index, method in enumerate(CONTROLLED_METHODS):
                values = draws[:, method_index]
                estimate_rows.append(
                    {
                        "table": table,
                        "estimator": estimator,
                        "method": method,
                        "support_min": float(CONTROLLED_SUPPORT[0]),
                        "support_max": float(CONTROLLED_SUPPORT[-1]),
                        "grid_points": len(CONTROLLED_SUPPORT),
                        "ncsAURC_point": float(point[method_index]),
                        "scene_bootstrap_mean": float(np.mean(values)),
                        "scene_bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                        "scene_bootstrap_ci95_high": float(np.quantile(values, 0.975)),
                        "bootstrap_replicates": CONTROLLED_BOOTSTRAP_REPLICATES,
                        "bootstrap_seed": CONTROLLED_BOOTSTRAP_SEED,
                        "statistical_unit": "scene",
                        "scene_units": 48,
                        "record_rows": 31_200,
                    }
                )
    method_index = {method: index for index, method in enumerate(CONTROLLED_METHODS)}
    for estimator in (
        "outer_train_threshold",
        "heldout_exact_count_diagnostic",
    ):
        for table, left, right in CONTROLLED_CONTRASTS:
            point, draws = result[(table, estimator)]
            left_index = method_index[left]
            right_index = method_index[right]
            differences = draws[:, left_index] - draws[:, right_index]
            point_difference = float(point[left_index] - point[right_index])
            contrast_rows.append(
                {
                    "table": table,
                    "estimator": estimator,
                    "contrast": f"{left}-minus-{right}",
                    "left_method": left,
                    "right_method": right,
                    "metric": "ncsAURC",
                    "metric_direction": "lower_is_better",
                    "support_min": float(CONTROLLED_SUPPORT[0]),
                    "support_max": float(CONTROLLED_SUPPORT[-1]),
                    "point_difference": point_difference,
                    "paired_bootstrap_mean": float(np.mean(differences)),
                    "paired_bootstrap_ci95_low": float(
                        np.quantile(differences, 0.025)
                    ),
                    "paired_bootstrap_ci95_high": float(
                        np.quantile(differences, 0.975)
                    ),
                    "fraction_below_zero": float(np.mean(differences < 0.0)),
                    "observed_direction": direction(point_difference, left, right),
                    "bootstrap_replicates": CONTROLLED_BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": CONTROLLED_BOOTSTRAP_SEED,
                    "statistical_unit": "scene",
                }
            )
    fold_sizes = [
        int(np.sum(methods["pcecf"]["fold"] == fold)) for fold in range(FOLD_COUNT)
    ]
    accepted_count_pass = True
    for table, counts in count_tables.items():
        totals = counts[:, :, :, 1].sum(axis=2).astype(int)
        for target_index, target in enumerate(CONTROLLED_TARGETS):
            if target < CONTROLLED_SUPPORT[0] - 1e-12 or target > CONTROLLED_SUPPORT[-1] + 1e-12:
                continue
            requested = sum(int(round(float(target) * size)) for size in fold_sizes)
            observed = {
                method: int(totals[index, target_index])
                for index, method in enumerate(CONTROLLED_METHODS)
            }
            exact = all(value == requested for value in observed.values())
            accepted_count_pass &= exact
            accepted_rows.append(
                {
                    "table": table,
                    "target_coverage": float(target),
                    "requested_total": requested,
                    "minimum_admitted_across_methods": min(observed.values()),
                    "maximum_admitted_across_methods": max(observed.values()),
                    "all_methods_exact": exact,
                    "admitted_by_method_json": json.dumps(
                        observed, sort_keys=True, separators=(",", ":")
                    ),
                    "selection": "heldout score rank only",
                    "tie_break": "record_id ascending",
                    "uses_evaluation_labels_for_selection": False,
                }
            )
    write_csv(OUTPUT / "CONTROLLED_ESTIMATES.csv", estimate_rows)
    write_csv(OUTPUT / "CONTROLLED_PAIRED_CONTRASTS.csv", contrast_rows)
    write_csv(OUTPUT / "CONTROLLED_ACCEPTED_COUNTS.csv", accepted_rows)
    primary_contrasts = [
        row for row in contrast_rows if row["estimator"] == "outer_train_threshold"
    ]
    gate = {
        "version": VERSION,
        "status": "EXECUTED_COMPLETE",
        "frozen_predictions_sha256": sha256(OOF),
        "frozen_outer_train_thresholds_sha256": sha256(FOLD_THRESHOLDS),
        "denominator": {"records": 31_200, "scenes": 48, "outer_folds": 5},
        "fixed_support": [0.10, 0.35],
        "support_grid_points": len(CONTROLLED_SUPPORT),
        "threshold_curve_targets": [0.01, 0.60, 0.01],
        "primary_estimator": (
            "frozen method-specific score thresholds fitted on outer-training scenes; "
            "all held-out ties at the fitted threshold are admitted"
        ),
        "accepted_count_diagnostic": {
            "role": "secondary retrospective score-only diagnostic",
            "selection": "eligible held-out rows ranked within outer fold",
            "tie_break": "record_id ascending",
            "uses_evaluation_labels_for_selection": False,
            "all_requested_counts_met": bool(accepted_count_pass),
        },
        "paired_bootstrap": {
            "unit": "scene",
            "replicates": CONTROLLED_BOOTSTRAP_REPLICATES,
            "seed": CONTROLLED_BOOTSTRAP_SEED,
            "paired_draws_shared_across_methods": True,
            "all_deterministic_records_retained_with_sampled_scene": True,
        },
        "support_selection": (
            "fixed before this run for comparability; no PACT outcome or "
            "contrast sign was consulted"
        ),
        "primary_contrasts": primary_contrasts,
        "verdict": (
            "COMPLETE" if accepted_count_pass else "ACCEPTED_COUNT_CHECK_FAILED"
        ),
    }
    write_json(OUTPUT / "CONTROLLED_GATE.json", gate)
    if not accepted_count_pass:
        raise AssertionError("accepted-count diagnostic did not equalize counts")
    return gate


def load_project_module():
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
    import pcecf_study as study  # type: ignore

    return study


def nested_concentration_arrays(
    controlled: Mapping[str, Mapping[str, np.ndarray]],
    truth: np.ndarray,
    record_ids: Sequence[str],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[float, dict[str, np.ndarray]],
    dict[float, dict[str, np.ndarray]],
]:
    study = load_project_module()
    records = sorted(
        study.read_records(CONTROLLED_DATA), key=lambda row: str(row["record_id"])
    )
    labels = {
        str(row["record_id"]): str(row["preferred_contract"])
        for row in study.read_records(CONTROLLED_LABELS)
    }
    observed_ids = tuple(str(record["record_id"]) for record in records)
    if observed_ids != tuple(record_ids) or set(labels) != set(record_ids):
        raise AssertionError("released records, labels, and frozen OOF IDs differ")
    if any(labels[record_id] != truth[index] for index, record_id in enumerate(record_ids)):
        raise AssertionError("released labels differ from frozen OOF labels")

    candidates = {
        method: {
            "score": controlled[method]["score"],
            "available": controlled[method]["native"]
            & controlled[method]["verifier"],
            "wrong": controlled[method]["wrong"],
        }
        for method in BASE_CANDIDATES
    }
    nested: dict[float, dict[str, np.ndarray]] = {
        concentration: {
            "score": np.empty(len(records), dtype=np.float64),
            "native": np.empty(len(records), dtype=bool),
            "common": controlled["pcecf"]["common"].copy(),
            "verifier": np.empty(len(records), dtype=bool),
            "wrong": np.empty(len(records), dtype=bool),
        }
        for concentration in CONCENTRATIONS
    }
    for index, record in enumerate(records):
        graph = study.graph_from_parent_sets(study.source_parents(record))
        for concentration in CONCENTRATIONS:
            predicted, _, score, native = study.predict_record(
                record,
                "nested_evidential_composition",
                concentration,
            )
            verification = study.verify_source_state(record, predicted, graph)
            nested[concentration]["score"][index] = score
            nested[concentration]["native"][index] = native
            nested[concentration]["verifier"][index] = bool(
                verification.admissible
            )
            nested[concentration]["wrong"][index] = predicted != truth[index]
    for concentration in CONCENTRATIONS:
        candidate = f"nested_evidential_composition_c{int(concentration)}"
        candidates[candidate] = {
            "score": nested[concentration]["score"],
            "available": nested[concentration]["native"]
            & nested[concentration]["verifier"],
            "wrong": nested[concentration]["wrong"],
        }

    base = controlled["pcecf"]
    original_concentration = base["concentration"]
    if np.any(~np.isfinite(original_concentration)):
        raise AssertionError("PACT OOF rows lack fold-local concentrations")
    if np.any((base["score"] < 0.0) | (base["score"] >= 1.0)):
        raise AssertionError("PACT non-vacuity score is outside [0,1)")
    base_evidence_ratio = np.divide(
        base["score"],
        (1.0 - base["score"]) * original_concentration,
        out=np.zeros_like(base["score"]),
        where=base["score"] > 0.0,
    )
    pact: dict[float, dict[str, np.ndarray]] = {}
    for concentration in CONCENTRATIONS:
        scaled = concentration * base_evidence_ratio
        score = scaled / (1.0 + scaled)
        same = original_concentration == concentration
        if np.any(same) and float(np.max(np.abs(score[same] - base["score"][same]))) > 1e-12:
            raise AssertionError("PACT concentration score transform failed")
        pact[concentration] = {
            "score": score,
            "common": base["common"],
            "verifier": base["verifier"],
            "wrong": base["wrong"],
        }
    return candidates, nested, pact


def fold_mapping(scene_count: int, seed: int) -> np.ndarray:
    scenes = list(range(scene_count))
    random.Random(seed).shuffle(scenes)
    mapping = np.empty(scene_count, dtype=np.int8)
    for index, scene in enumerate(scenes):
        mapping[scene] = index % FOLD_COUNT
    return mapping


def selection_counts(
    data: Mapping[str, np.ndarray],
    folds: np.ndarray,
    outer: int,
) -> dict[str, np.ndarray]:
    totals = {
        "n": np.zeros(len(POLICY_TARGETS), dtype=np.float64),
        "admitted": np.zeros(len(POLICY_TARGETS), dtype=np.float64),
        "wrong": np.zeros(len(POLICY_TARGETS), dtype=np.float64),
    }
    for inner in range(FOLD_COUNT):
        if inner == outer:
            continue
        train = (folds != outer) & (folds != inner)
        validate = folds == inner
        cutoff = thresholds(
            data["score"], data["available"], train, POLICY_TARGETS
        )
        accepted = (
            validate[:, None]
            & data["available"][:, None]
            & (data["score"][:, None] >= cutoff[None, :])
        )
        totals["n"] += int(np.sum(validate))
        totals["admitted"] += np.sum(accepted, axis=0)
        totals["wrong"] += np.sum(accepted & data["wrong"][:, None], axis=0)
    totals["correct"] = totals["admitted"] - totals["wrong"]
    return totals


def select_candidates(
    candidates: Mapping[str, Mapping[str, np.ndarray]],
    folds: np.ndarray,
    repeat: int,
    seed: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    order = {candidate: index for index, candidate in enumerate(SELECTION_CANDIDATES)}
    winners: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for outer in range(FOLD_COUNT):
        scored = []
        for candidate in SELECTION_CANDIDATES:
            counts = selection_counts(candidates[candidate], folds, outer)
            n = counts["n"]
            expected_cost = (
                5.0 * counts["wrong"] + 0.5 * (n - counts["admitted"])
            ) / n
            wrong_all = counts["wrong"] / n
            correct_all = counts["correct"] / n
            row = {
                "repeat": repeat,
                "seed": seed,
                "outer_fold": outer,
                "candidate": candidate,
                "mean_inner_expected_cost": float(np.mean(expected_cost)),
                "mean_inner_wrong_all": float(np.mean(wrong_all)),
                "mean_inner_correct_all": float(np.mean(correct_all)),
            }
            scored.append(row)
        winner = min(
            scored,
            key=lambda row: (
                row["mean_inner_expected_cost"],
                row["mean_inner_wrong_all"],
                -row["mean_inner_correct_all"],
                order[str(row["candidate"])],
            ),
        )
        name = str(winner["candidate"])
        winners[outer] = name
        concentration = (
            float(name.rsplit("c", 1)[1])
            if name.startswith("nested_evidential_composition_c")
            else None
        )
        rows.append(
            {
                **winner,
                "selected_candidate": name,
                "selected_concentration": concentration,
                "complete_inner_reselection": True,
            }
        )
    return winners, rows


def evaluate_repeated_curve(
    data_by_outer: Mapping[int, Mapping[str, np.ndarray]],
    folds: np.ndarray,
    *,
    verifier: bool,
) -> float:
    n = np.zeros(len(CONTROLLED_TARGETS), dtype=np.float64)
    admitted = np.zeros(len(CONTROLLED_TARGETS), dtype=np.float64)
    wrong = np.zeros(len(CONTROLLED_TARGETS), dtype=np.float64)
    for outer in range(FOLD_COUNT):
        data = data_by_outer[outer]
        train = folds != outer
        test = folds == outer
        available = data["common"] & (data["verifier"] if verifier else True)
        cutoff = thresholds(data["score"], available, train, CONTROLLED_TARGETS)
        accepted = (
            test[:, None]
            & available[:, None]
            & (data["score"][:, None] >= cutoff[None, :])
        )
        n += int(np.sum(test))
        admitted += np.sum(accepted, axis=0)
        wrong += np.sum(accepted & data["wrong"][:, None], axis=0)
    coverage = admitted / n
    risk = np.divide(
        wrong, admitted, out=np.zeros_like(wrong), where=admitted > 0
    )
    return grouped_ncs_aurc(coverage, risk, CONTROLLED_SUPPORT)


def run_repeated_splits(
    controlled: Mapping[str, Mapping[str, np.ndarray]],
    scene_index: np.ndarray,
    truth: np.ndarray,
    record_ids: Sequence[str],
) -> dict[str, Any]:
    required = (
        CONTROLLED_DATA,
        CONTROLLED_LABELS,
        CONTROLLED_CONFIG,
        ROOT / "experiments" / "nested_selection.py",
        ROOT / "experiments" / "controlled_study.py",
        ROOT / "experiments" / "pcecf_study.py",
        ROOT / "src" / "action_admission" / "pcecf.py",
        OOF,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        gate = {
            "version": VERSION,
            "status": "NOT_ELIGIBLE",
            "missing_inputs": missing,
            "reason": "complete train-only reselection cannot be executed from the released assets",
        }
        write_json(OUTPUT / "REPEATED_SPLIT_GATE.json", gate)
        return gate

    candidates, nested, pact = nested_concentration_arrays(
        controlled, truth, record_ids
    )
    configured = json.loads(CONTROLLED_CONFIG.read_text(encoding="utf-8"))
    expected_original = {
        int(fold): float(value)
        for fold, value in configured["dirichlet_concentration_by_fold"].items()
    }
    original_folds = controlled["pcecf"]["fold"]
    original_winners, _ = select_candidates(
        candidates, original_folds, -1, -1
    )
    reproduced = {
        fold: float(name.rsplit("c", 1)[1])
        if name.startswith("nested_evidential_composition_c")
        else None
        for fold, name in original_winners.items()
    }
    if reproduced != expected_original:
        gate = {
            "version": VERSION,
            "status": "NOT_ELIGIBLE",
            "missing_inputs": [
                "a released selector contract that reproduces the configured original-fold concentrations"
            ],
            "expected_original_concentrations": expected_original,
            "reselected_original_concentrations": reproduced,
            "reason": "released selector reproduction failed; alternate-split results are not admitted",
        }
        write_json(OUTPUT / "REPEATED_SPLIT_GATE.json", gate)
        return gate

    repeat_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    undefined: list[dict[str, Any]] = []
    product = controlled["product_evidence_fusion"]
    product_data = {
        "score": product["score"],
        "common": product["common"],
        "verifier": product["verifier"],
        "wrong": product["wrong"],
    }
    for repeat in range(REPEATED_SPLITS):
        seed = REPEATED_BASE_SEED + repeat
        scene_folds = fold_mapping(48, seed)
        folds = scene_folds[scene_index]
        winners, selected = select_candidates(candidates, folds, repeat, seed)
        selection_rows.extend(selected)
        selected_concentrations: dict[int, float] = {}
        for outer, winner in winners.items():
            if not winner.startswith("nested_evidential_composition_c"):
                undefined.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "outer_fold": outer,
                        "selected_candidate": winner,
                    }
                )
            else:
                selected_concentrations[outer] = float(winner.rsplit("c", 1)[1])
        if len(selected_concentrations) != FOLD_COUNT:
            continue
        pact_by_outer = {
            outer: pact[selected_concentrations[outer]] for outer in range(FOLD_COUNT)
        }
        nested_by_outer = {
            outer: nested[selected_concentrations[outer]]
            for outer in range(FOLD_COUNT)
        }
        product_by_outer = {outer: product_data for outer in range(FOLD_COUNT)}
        fusion_pact = evaluate_repeated_curve(
            pact_by_outer, folds, verifier=False
        )
        fusion_product = evaluate_repeated_curve(
            product_by_outer, folds, verifier=False
        )
        verifier_pact = evaluate_repeated_curve(
            pact_by_outer, folds, verifier=True
        )
        verifier_nested = evaluate_repeated_curve(
            nested_by_outer, folds, verifier=True
        )
        repeat_rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "fusion_pact_ncsAURC": fusion_pact,
                "fusion_product_ncsAURC": fusion_product,
                "fusion_pact_minus_product": fusion_pact - fusion_product,
                "verifier_pact_ncsAURC": verifier_pact,
                "verifier_nested_ncsAURC": verifier_nested,
                "verifier_pact_minus_nested": verifier_pact - verifier_nested,
                "selected_concentrations_by_outer_fold": json.dumps(
                    selected_concentrations,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    write_csv(OUTPUT / "REPEATED_SPLIT_OUTER_SELECTION.csv", selection_rows)
    if undefined:
        gate = {
            "version": VERSION,
            "status": "NOT_ELIGIBLE",
            "missing_inputs": [
                "a predeclared mapping from a non-nested selector winner to the PACT concentration"
            ],
            "undefined_outer_folds": undefined,
            "reason": (
                "complete reselection selected a concentration-free backbone in at least one outer fold; "
                "inventing a PACT concentration would violate the released selector contract"
            ),
        }
        write_json(OUTPUT / "REPEATED_SPLIT_GATE.json", gate)
        return gate
    write_csv(OUTPUT / "REPEATED_SPLIT_ESTIMATES.csv", repeat_rows)

    fusion_delta = [float(row["fusion_pact_minus_product"]) for row in repeat_rows]
    verifier_delta = [float(row["verifier_pact_minus_nested"]) for row in repeat_rows]
    frequencies = Counter(
        str(row["selected_candidate"]) for row in selection_rows
    )
    gate = {
        "version": VERSION,
        "status": "EXECUTED_COMPLETE",
        "repeats": REPEATED_SPLITS,
        "seeds": [REPEATED_BASE_SEED, REPEATED_BASE_SEED + REPEATED_SPLITS - 1],
        "scene_groups": 48,
        "outer_folds": FOLD_COUNT,
        "fixed_support": [0.10, 0.35],
        "selection_policy_targets": POLICY_TARGETS.tolist(),
        "selection_candidates": list(SELECTION_CANDIDATES),
        "selection_objective": [
            "mean inner-fold expected cost",
            "mean inner-fold wrong/all",
            "negative mean inner-fold correct/all",
            "predeclared candidate order",
        ],
        "complete_reselection": (
            "scene folds, candidate backbone/concentration, and every score threshold are refit within each repeat"
        ),
        "original_split_self_check": {
            "passed": True,
            "configured_concentrations": expected_original,
            "reselected_concentrations": reproduced,
        },
        "selection_frequency": dict(sorted(frequencies.items())),
        "fusion_pact_minus_product": quantile_summary(fusion_delta),
        "fusion_negative_direction_repeats": sum(value < 0.0 for value in fusion_delta),
        "shared_verifier_pact_minus_nested": quantile_summary(verifier_delta),
        "shared_verifier_negative_direction_repeats": sum(
            value < 0.0 for value in verifier_delta
        ),
        "claim_boundary": (
            "repeat quantiles describe sensitivity to scene-fold assignment; "
            "they are not independent-sample confidence intervals"
        ),
        "support_selection": (
            "fixed before this run for comparability; no repeated-split PACT outcome was consulted"
        ),
        "verdict": "EXECUTED_WITHOUT_SIGN_FILTER",
    }
    write_json(OUTPUT / "REPEATED_SPLIT_GATE.json", gate)
    return gate


def load_fm_module():
    specification = importlib.util.spec_from_file_location(
        "released_fm_analysis", FM_ANALYSIS
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load released balanced-panel analysis")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def fractional_risk(
    scores: np.ndarray,
    reference: np.ndarray,
    coverages: np.ndarray,
) -> np.ndarray:
    unique = np.unique(scores)[::-1]
    group_total = np.asarray([np.sum(scores == value) for value in unique], dtype=float)
    group_wrong = np.asarray(
        [np.sum((scores == value) & (reference == 0)) for value in unique],
        dtype=float,
    )
    cumulative_total = np.cumsum(group_total)
    cumulative_wrong = np.cumsum(group_wrong)
    result = np.empty(len(coverages), dtype=np.float64)
    for index, coverage in enumerate(coverages):
        target = coverage * len(scores)
        group = int(np.searchsorted(cumulative_total, target, side="left"))
        before_total = cumulative_total[group - 1] if group else 0.0
        before_wrong = cumulative_wrong[group - 1] if group else 0.0
        fraction = (target - before_total) / group_total[group]
        wrong = before_wrong + fraction * group_wrong[group]
        result[index] = wrong / target
    return result


def fractional_ncs(
    scores: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> float:
    risk = fractional_risk(scores, reference, support)
    return float(np.trapezoid(risk, support) / (support[-1] - support[0]))


def run_balanced_panel() -> dict[str, Any]:
    fm = load_fm_module()
    cases, probabilities, input_hashes = fm.load_inputs(FM_CASES, FM_INPUTS)
    reference = np.asarray([int(row["reference_ready"]) for row in cases], dtype=int)
    predictions = {
        "product": fm.product(probabilities),
        "nested_dirichlet": fm.nested_dirichlet(probabilities),
        "hierarchy_matched_cautious": fm.hierarchy_matched_cautious(probabilities),
    }
    predictions["pact_registered_family"], _ = fm.pact(probabilities, fm.REGISTERED)
    predictions["pact_false_split"], _ = fm.pact(
        probabilities, tuple((index,) for index in range(8))
    )
    predictions["pact_false_merge"], _ = fm.pact(
        probabilities, (tuple(range(8)),)
    )
    methods = tuple(fm.METHODS)
    supports = {
        "registered_full_fractional_tie": FM_FULL_SUPPORT,
        "controlled_comparability_fractional_tie": FM_COMPARABILITY_SUPPORT,
    }
    point = np.empty((len(supports), len(methods)), dtype=np.float64)
    for support_index, support in enumerate(supports.values()):
        for method_index, method in enumerate(methods):
            point[support_index, method_index] = fractional_ncs(
                predictions[method], reference, support
            )

    published: dict[str, float] = {}
    with FM_PUBLISHED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            published[str(row["method"])] = float(row["ncsAURC"])
    full_index = 0
    residuals = {
        method: float(abs(point[full_index, index] - published[method]))
        for index, method in enumerate(methods)
    }
    if max(residuals.values()) > 1e-12:
        raise AssertionError("fractional-tie full-support points do not reproduce release")

    rng = np.random.default_rng(FM_BOOTSTRAP_SEED)
    draws = np.empty(
        (FM_BOOTSTRAP_REPLICATES, len(supports), len(methods)), dtype=np.float64
    )
    for replicate in range(FM_BOOTSTRAP_REPLICATES):
        indices = fm.cluster_sample_indices(cases, rng)
        sampled_reference = reference[indices]
        for support_index, support in enumerate(supports.values()):
            for method_index, method in enumerate(methods):
                draws[replicate, support_index, method_index] = fractional_ncs(
                    predictions[method][indices], sampled_reference, support
                )

    estimate_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for support_index, (estimand, support) in enumerate(supports.items()):
        for method_index, method in enumerate(methods):
            values = draws[:, support_index, method_index]
            estimate_rows.append(
                {
                    "estimand": estimand,
                    "method": method,
                    "support_min": float(support[0]),
                    "support_max": float(support[-1]),
                    "grid_points": len(support),
                    "tie_estimand": "analytic fractional inclusion within equal-score group",
                    "ncsAURC_point": float(point[support_index, method_index]),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": FM_BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": FM_BOOTSTRAP_SEED,
                    "resampling_unit": "episode within task",
                    "rows": len(cases),
                    "events": len({row["event_id"] for row in cases}),
                    "episodes": len({row["episode_id"] for row in cases}),
                    "tasks": len({row["task_id"] for row in cases}),
                }
            )
        pact_index = methods.index("pact_registered_family")
        for method_index, method in enumerate(methods):
            if method == "pact_registered_family":
                continue
            differences = (
                draws[:, support_index, pact_index]
                - draws[:, support_index, method_index]
            )
            point_difference = float(
                point[support_index, pact_index] - point[support_index, method_index]
            )
            contrast_rows.append(
                {
                    "estimand": estimand,
                    "contrast": f"pact_registered_family-minus-{method}",
                    "left_method": "pact_registered_family",
                    "right_method": method,
                    "metric": "ncsAURC",
                    "metric_direction": "lower_is_better",
                    "support_min": float(support[0]),
                    "support_max": float(support[-1]),
                    "point_difference": point_difference,
                    "paired_bootstrap_mean": float(np.mean(differences)),
                    "paired_bootstrap_ci95_low": float(
                        np.quantile(differences, 0.025)
                    ),
                    "paired_bootstrap_ci95_high": float(
                        np.quantile(differences, 0.975)
                    ),
                    "fraction_below_zero": float(np.mean(differences < 0.0)),
                    "observed_direction": direction(
                        point_difference, "pact_registered_family", method
                    ),
                    "bootstrap_replicates": FM_BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": FM_BOOTSTRAP_SEED,
                    "resampling_unit": "episode within task",
                }
            )
    write_csv(OUTPUT / "BALANCED_PANEL_ESTIMATES.csv", estimate_rows)
    write_csv(OUTPUT / "BALANCED_PANEL_PAIRED_CONTRASTS.csv", contrast_rows)
    ordering = {
        estimand: [
            methods[index]
            for index in np.argsort(point[support_index], kind="stable")
        ]
        for support_index, estimand in enumerate(supports)
    }
    gate = {
        "version": VERSION,
        "status": "EXECUTED_COMPLETE",
        "denominator": {
            "rows": len(cases),
            "events": len({row["event_id"] for row in cases}),
            "episodes": len({row["episode_id"] for row in cases}),
            "tasks": len({row["task_id"] for row in cases}),
            "checkpoints": len(fm.MODELS),
        },
        "estimands": {
            "registered": {
                "support": [0.10, 0.90],
                "grid_points": 36,
                "tie_rule": "analytic fractional inclusion within equal-score group",
            },
            "controlled_comparability_sensitivity": {
                "support": [0.10, 0.39],
                "grid_points": 36,
                "tie_rule": "analytic fractional inclusion within equal-score group",
                "role": "comparability sensitivity; it does not replace the registered panel estimand",
            },
        },
        "paired_bootstrap": {
            "unit": "episode within task",
            "replicates": FM_BOOTSTRAP_REPLICATES,
            "seed": FM_BOOTSTRAP_SEED,
            "paired_draws_shared_across_methods": True,
            "all rows of each sampled episode retained": True,
        },
        "full_support_release_reproduction_max_abs_residual": max(
            residuals.values()
        ),
        "method_order_lower_ncsAURC_first": ordering,
        "support_selection": (
            "the 0.10--0.39 sensitivity was fixed for controlled-study comparability; "
            "no PACT panel outcome or contrast sign was consulted"
        ),
        "input_sha256": {
            str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/"): digest
            for path, digest in input_hashes.items()
        },
        "verdict": "COMPLETE_WITH_UNFAVORABLE_AND_REVERSED_ORDERINGS_RETAINED",
    }
    write_json(OUTPUT / "BALANCED_PANEL_GATE.json", gate)
    return gate


def input_hashes() -> dict[str, str]:
    paths = (
        OOF,
        FOLD_THRESHOLDS,
        CONTROLLED_DATA,
        CONTROLLED_LABELS,
        CONTROLLED_CONFIG,
        PCECF_CONFIG,
        ROOT / "experiments" / "controlled_study.py",
        ROOT / "experiments" / "nested_selection.py",
        ROOT / "experiments" / "pcecf_study.py",
        ROOT / "src" / "action_admission" / "pcecf.py",
        FM_CASES,
        FM_ANALYSIS,
        FM_PUBLISHED,
        Path(__file__),
        ROOT / "analysis" / "verify_p0_estimand_closure.py",
    )
    result = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in paths
        if path.is_file()
    }
    for path in sorted(FM_INPUTS.glob("*.jsonl")):
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    return result


def write_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{sha256(path)}  {path.name}")
    with (OUTPUT / "MANIFEST.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(rows) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    controlled, scene_index, truth, record_ids = load_controlled_predictions()
    controlled_gate = run_controlled(controlled, scene_index)
    repeated_gate = run_repeated_splits(
        controlled, scene_index, truth, record_ids
    )
    balanced_gate = run_balanced_panel()
    hashes = input_hashes()
    protocol = {
        "version": VERSION,
        "status": "LOCKED",
        "scope": "P0-2 estimand closure only",
        "controlled": {
            "fixed_support": [0.10, 0.35],
            "support_grid_points": 36,
            "threshold_targets": [0.01, 0.60, 0.01],
            "bootstrap_replicates": CONTROLLED_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": CONTROLLED_BOOTSTRAP_SEED,
        },
        "repeated_scene_splits": {
            "repeats": REPEATED_SPLITS,
            "base_seed": REPEATED_BASE_SEED,
            "outer_folds": FOLD_COUNT,
            "policy_targets": POLICY_TARGETS.tolist(),
            "concentration_grid": list(CONCENTRATIONS),
            "complete_reselection_required": True,
        },
        "balanced_panel": {
            "registered_fractional_tie_support": [0.10, 0.90],
            "comparability_fractional_tie_support": [0.10, 0.39],
            "grid_points_each": 36,
            "bootstrap_replicates": FM_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": FM_BOOTSTRAP_SEED,
            "bootstrap_unit": "episode within task",
        },
        "anti_selection_rule": (
            "supports, seeds, grids, candidate order, and contrast directions are fixed in code; "
            "no result is omitted or support changed based on PACT outcomes"
        ),
        "input_sha256": hashes,
    }
    write_json(OUTPUT / "PROTOCOL_LOCK.json", protocol)
    overall = (
        "COMPLETE"
        if repeated_gate["status"] == "EXECUTED_COMPLETE"
        else "COMPLETE_WITH_REPEATED_SPLIT_NOT_ELIGIBLE"
    )
    final = {
        "version": VERSION,
        "status": overall,
        "controlled": {
            "status": controlled_gate["status"],
            "verdict": controlled_gate["verdict"],
            "fixed_support": controlled_gate["fixed_support"],
        },
        "repeated_scene_splits": {
            "status": repeated_gate["status"],
            "verdict": repeated_gate.get("verdict", "NOT_ELIGIBLE"),
            "missing_inputs": repeated_gate.get("missing_inputs", []),
        },
        "balanced_panel": {
            "status": balanced_gate["status"],
            "verdict": balanced_gate["verdict"],
            "registered_support": [0.10, 0.90],
            "comparability_support": [0.10, 0.39],
        },
        "no_sign_filter": True,
        "no_fabricated_results": True,
    }
    write_json(OUTPUT / "FINAL_VERDICT.json", final)
    write_manifest()
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
