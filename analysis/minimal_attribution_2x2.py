#!/usr/bin/env python3
"""Secondary post-hoc PACT topology-fusion x provenance-verifier attribution."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
for search_path in (REPO / "src", REPO / "experiments"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import pcecf_study as controlled  # noqa: E402
from action_admission import (  # noqa: E402
    CONTRACT_CLASSES,
    graph_from_parent_sets,
    verify_source_state,
)
from action_admission.pcecf import SourceEvidence, forward  # noqa: E402
from action_admission.verifier import DEFAULT_CONFIG as DEFAULT_VERIFIER_CONFIG  # noqa: E402
from controlled_study import (  # noqa: E402
    cluster_id,
    read_records,
    scene_fold_map,
    source_parents,
    uncompressed_sha256,
)


OUT = ROOT / "outputs" / "minimal_attribution_2x2"
DATA = REPO / "data" / "controlled" / "source_records.jsonl.gz"
LABELS = REPO / "data" / "controlled" / "evaluation_labels.jsonl.gz"
CONFIG = REPO / "configs" / "controlled_study.json"
SOURCE_NAMES = ("language", "geometry", "risk")
SOURCE_ROLES = {
    "language": "command",
    "geometry": "geometry",
    "risk": "risk",
}
FUSIONS = ("F0_singleton", "F1_registered")
VERIFIERS = ("V0_no_provenance", "V1_full")
CELLS = tuple(f"{fusion}:{verifier}" for fusion in FUSIONS for verifier in VERIFIERS)
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
SUPPORTS = {
    "primary_0.10_0.39": (0.10, 0.39),
    "fixed_0.10_0.35": (0.10, 0.35),
}
TRAINING_TARGET = 0.13
BOOTSTRAPS = 2000
SEED = 51402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probability_vector(payload: Mapping[str, Any]) -> np.ndarray:
    probabilities = payload.get("probabilities", {})
    vector = np.asarray(
        [float(probabilities.get(label, 0.0)) for label in CONTRACT_CLASSES],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("source probabilities must be finite and nonnegative")
    total = float(vector.sum())
    return vector / total if total > 0.0 else np.zeros_like(vector)


def observed_count(record: Mapping[str, Any]) -> int:
    return sum(
        bool(record.get("sources", {}).get(name, {}))
        and not bool(record["sources"][name].get("missing", False))
        for name in SOURCE_NAMES
    )


def is_complete(record: Mapping[str, Any]) -> bool:
    return observed_count(record) == len(SOURCE_NAMES)


def pact_sources(record: Mapping[str, Any], fusion: str) -> list[SourceEvidence]:
    registered = source_parents(record)
    result = []
    for name in SOURCE_NAMES:
        payload = record.get("sources", {}).get(name, {})
        missing = not payload or bool(payload.get("missing", False))
        probabilities = probability_vector(payload)
        valid = (
            payload.get("schema_valid", True) is not False
            and float(probabilities.sum()) > 0.0
        )
        parents = (
            tuple(registered[name])
            if fusion == "F1_registered"
            else (f"singleton:{name}",)
        )
        result.append(
            SourceEvidence(
                source_id=name,
                probabilities=probabilities,
                quality=float(payload.get("quality", 0.0)),
                conflict=float(payload.get("conflict", 0.0)),
                missing=missing,
                parents=parents,
                valid=valid,
            )
        )
    return result


def make_rows(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    folds: Mapping[str, int],
    concentrations: Mapping[int, float],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    v0_config = replace(DEFAULT_VERIFIER_CONFIG, minimum_registered_components=0)
    unique_concentrations = tuple(sorted(set(concentrations.values())))
    cache: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for fusion in FUSIONS:
        for concentration in unique_concentrations:
            rows = []
            for record in records:
                record_id = str(record["record_id"])
                sources = pact_sources(record, fusion)
                output = forward(
                    sources,
                    concentration=float(concentration),
                    expected_source_ids=SOURCE_NAMES,
                )
                prediction = CONTRACT_CLASSES[output.predicted_index]
                graph = graph_from_parent_sets(source_parents(record))
                v0 = verify_source_state(record, prediction, graph, config=v0_config)
                v1 = verify_source_state(
                    record,
                    prediction,
                    graph,
                    config=DEFAULT_VERIFIER_CONFIG,
                )
                rows.append(
                    {
                        "record_id": record_id,
                        "scene_id": str(record["metadata"]["scene_id"]),
                        "fold": int(folds[cluster_id(record_id)]),
                        "complete": is_complete(record),
                        "score": float(output.selection_score),
                        "eligible": observed_count(record) >= 2,
                        "native_eligible": observed_count(record) >= 2,
                        "fold_local_concentration": float(concentration),
                        "predicted_contract": prediction,
                        "preferred_contract": labels[record_id],
                        "probabilities": {
                            label: float(output.posterior[index])
                            for index, label in enumerate(CONTRACT_CLASSES)
                        },
                        "components": tuple(tuple(group) for group in output.group_ids),
                        "v0_pass": bool(v0.admissible),
                        "v0_route": str(v0.route),
                        "v0_reason": str(v0.reason),
                        "v1_pass": bool(v1.admissible),
                        "v1_route": str(v1.route),
                        "v1_reason": str(v1.reason),
                    }
                )
            cache[(fusion, concentration)] = rows
    return {
        fusion: {
            fold: cache[(fusion, float(concentrations[fold]))]
            for fold in sorted(concentrations)
        }
        for fusion in FUSIONS
    }


def cell_rows_by_outer(
    fusion_rows: Mapping[int, list[dict[str, Any]]],
    verifier: str,
    stratum: str,
) -> dict[int, list[dict[str, Any]]]:
    pass_key = "v1_pass" if verifier == "V1_full" else "v0_pass"
    route_key = "v1_route" if verifier == "V1_full" else "v0_route"
    reason_key = "v1_reason" if verifier == "V1_full" else "v0_reason"
    result = {}
    for outer_fold, rows in fusion_rows.items():
        result[outer_fold] = [
            {
                **row,
                "verifier_pass": bool(row[pass_key]),
                "verifier_route": str(row[route_key]),
                "verifier_reason": str(row[reason_key]),
            }
            for row in rows
            if stratum == "all_records" or bool(row["complete"])
        ]
    return result


def held_out_rows(rows_by_outer: Mapping[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = [
        row
        for outer_fold, values in sorted(rows_by_outer.items())
        for row in values
        if int(row["fold"]) == int(outer_fold)
    ]
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise AssertionError("held-out records are not unique")
    return rows


def fold_allocation(total: int, fold_sizes: Sequence[int]) -> tuple[int, ...]:
    size = sum(fold_sizes)
    desired = [total * value / size for value in fold_sizes]
    allocated = [int(math.floor(value)) for value in desired]
    order = sorted(
        range(len(fold_sizes)),
        key=lambda index: (-(desired[index] - allocated[index]), index),
    )
    for index in order[: total - sum(allocated)]:
        allocated[index] += 1
    return tuple(allocated)


def exact_step_ncsaurc(
    rows: Sequence[Mapping[str, Any]],
    low: float,
    high: float,
) -> dict[str, float | int | str | bool]:
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    fold_sizes: dict[int, int] = defaultdict(int)
    for row in rows:
        fold = int(row["fold"])
        fold_sizes[fold] += 1
        if bool(row["eligible"]) and bool(row["verifier_pass"]):
            by_fold[fold].append(row)
    folds = tuple(sorted(fold_sizes))
    sizes = tuple(fold_sizes[fold] for fold in folds)
    total_n = sum(sizes)
    low_count = int(math.ceil(low * total_n))
    high_count = int(math.floor(high * total_n))
    ranked_wrong: list[np.ndarray] = []
    ranked_correct: list[np.ndarray] = []
    for fold, size in zip(folds, sizes):
        ranked = sorted(
            by_fold[fold],
            key=lambda row: (-float(row["score"]), str(row["record_id"])),
        )
        required = int(math.ceil(high * size))
        if len(ranked) < required:
            raise AssertionError(
                f"fold {fold} has {len(ranked)} eligible rows, below {required}"
            )
        wrong = np.fromiter(
            (
                row["predicted_contract"] != row["preferred_contract"]
                for row in ranked
            ),
            dtype=np.int64,
        )
        correct = 1 - wrong
        ranked_wrong.append(np.concatenate(([0], np.cumsum(wrong))))
        ranked_correct.append(np.concatenate(([0], np.cumsum(correct))))

    coverages = np.empty(high_count - low_count + 1, dtype=np.float64)
    risks = np.empty_like(coverages)
    wrong_all = np.empty_like(coverages)
    correct_all = np.empty_like(coverages)
    for offset, total in enumerate(range(low_count, high_count + 1)):
        allocation = fold_allocation(total, sizes)
        wrong = sum(
            int(ranked_wrong[index][count])
            for index, count in enumerate(allocation)
        )
        correct = sum(
            int(ranked_correct[index][count])
            for index, count in enumerate(allocation)
        )
        coverages[offset] = total / total_n
        risks[offset] = wrong / total
        wrong_all[offset] = wrong / total_n
        correct_all[offset] = correct / total_n
    width = float(coverages[-1] - coverages[0])
    return {
        "score": "native",
        "verifier": True,
        "records": total_n,
        "coverage_low": float(coverages[0]),
        "coverage_high": float(coverages[-1]),
        "curve_points": int(coverages.size),
        "exact_right_step_naurc": float(
            np.sum(risks[1:] * np.diff(coverages)) / width
        ),
        "exact_trapezoidal_naurc": float(
            np.trapezoid(risks, coverages) / width
        ),
        "wrong_all_at_low": float(wrong_all[0]),
        "wrong_all_at_high": float(wrong_all[-1]),
        "correct_all_at_low": float(correct_all[0]),
        "correct_all_at_high": float(correct_all[-1]),
        "selection_uses_labels": False,
    }


def point_bootstrap(
    counts: Mapping[str, Mapping[tuple[float, str], Mapping[str, int]]],
    scenes: Sequence[str],
    target: float,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, np.ndarray]]:
    cells = tuple(counts)
    fields = ("n", "admitted", "wrong", "correct")
    values = np.zeros((len(cells), len(scenes), len(fields)), dtype=np.float64)
    for cell_index, cell in enumerate(cells):
        for scene_index, scene in enumerate(scenes):
            source = counts[cell][(target, scene)]
            values[cell_index, scene_index, :] = [source[field] for field in fields]
    rng = np.random.default_rng(seed)
    draws = {
        "coverage": np.zeros((replicates, len(cells))),
        "wrong_all": np.zeros((replicates, len(cells))),
        "correct_all": np.zeros((replicates, len(cells))),
    }
    for replicate in range(replicates):
        weights = rng.multinomial(len(scenes), np.full(len(scenes), 1.0 / len(scenes)))
        totals = np.tensordot(values, weights, axes=(1, 0))
        n = totals[:, 0]
        draws["coverage"][replicate] = totals[:, 1] / n
        draws["wrong_all"][replicate] = totals[:, 2] / n
        draws["correct_all"][replicate] = totals[:, 3] / n
    summaries = []
    for cell_index, cell in enumerate(cells):
        row: dict[str, Any] = {"cell": cell, "replicates": replicates, "seed": seed}
        for metric, matrix in draws.items():
            row[metric] = float(np.mean(matrix[:, cell_index]))
            row[f"{metric}_ci_low"] = float(np.quantile(matrix[:, cell_index], 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(matrix[:, cell_index], 0.975))
        summaries.append(row)
    return summaries, cells, draws


def factorial_effects(
    methods: Sequence[str],
    draws: np.ndarray,
    estimand: str,
) -> list[dict[str, Any]]:
    index = {method: methods.index(method) for method in methods}
    contrasts = {
        "fusion_effect_at_V0": ("F1_registered:V0_no_provenance", "F0_singleton:V0_no_provenance"),
        "fusion_effect_at_V1": ("F1_registered:V1_full", "F0_singleton:V1_full"),
        "verifier_effect_at_F0": ("F0_singleton:V1_full", "F0_singleton:V0_no_provenance"),
        "verifier_effect_at_F1": ("F1_registered:V1_full", "F1_registered:V0_no_provenance"),
    }
    output = []
    deltas = {}
    for name, (left, right) in contrasts.items():
        delta = draws[:, index[left]] - draws[:, index[right]]
        deltas[name] = delta
        output.append(
            {
                "contrast": name,
                "estimand": estimand,
                "left": left,
                "right": right,
                "mean": float(np.mean(delta)),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "fraction_below_zero": float(np.mean(delta < 0.0)),
            }
        )
    interaction = deltas["fusion_effect_at_V1"] - deltas["fusion_effect_at_V0"]
    output.append(
        {
            "contrast": "difference_in_differences",
            "estimand": estimand,
            "left": "(F1V1-F0V1)",
            "right": "(F1V0-F0V0)",
            "mean": float(np.mean(interaction)),
            "ci_low": float(np.quantile(interaction, 0.025)),
            "ci_high": float(np.quantile(interaction, 0.975)),
            "fraction_below_zero": float(np.mean(interaction < 0.0)),
        }
    )
    return output


def route_transitions(
    fusion: str,
    fusion_rows: Mapping[int, list[dict[str, Any]]],
    stratum: str,
) -> list[dict[str, Any]]:
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    by_scene: dict[tuple[str, str, str], int] = defaultdict(int)
    for outer_fold, raw_rows in sorted(fusion_rows.items()):
        rows = [
            row
            for row in raw_rows
            if stratum == "all_records" or bool(row["complete"])
        ]
        train = [row for row in rows if int(row["fold"]) != int(outer_fold)]
        test = [row for row in rows if int(row["fold"]) == int(outer_fold)]
        fit_rows = [{**row, "verifier_pass": bool(row["v0_pass"])} for row in train]
        cutoff = controlled.threshold(fit_rows, TRAINING_TARGET, verifier=True)
        for row in test:
            if not bool(row["eligible"]) or float(row["score"]) < cutoff:
                left = right = "score_withhold"
            else:
                left = "admit" if bool(row["v0_pass"]) else str(row["v0_route"])
                right = "admit" if bool(row["v1_pass"]) else str(row["v1_route"])
            transitions[(left, right)] += 1
            by_scene[(str(row["scene_id"]), left, right)] += 1
    return [
        {
            "fusion": fusion,
            "stratum": stratum,
            "threshold_protocol": "V0 outer-train-fitted target 0.13, held fixed for V0-to-V1 transition",
            "from_V0": left,
            "to_V1": right,
            "records": count,
            "scenes_with_transition": sum(
                by_scene.get((scene, left, right), 0) > 0
                for scene in {key[0] for key in by_scene}
            ),
        }
        for (left, right), count in sorted(transitions.items())
    ]


def exact_count_summary(
    rows: Sequence[Mapping[str, Any]],
    low: float,
    high: float,
) -> dict[str, Any]:
    return exact_step_ncsaurc(rows, low, high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol = {
        "analysis_status": "secondary post-hoc mechanistic attribution",
        "directional_acceptance_rule": None,
        "factor_F0": "PACT fusion with each fixed catalog slot assigned a singleton parent",
        "factor_F1": "PACT fusion with the current registered partition",
        "verifier_partition_fixed": (
            "current registered partition in both fusion arms; "
            "F changes only operator aggregation"
        ),
        "factor_V0": "current verifier with only the provenance predicate disabled",
        "factor_V1": "current full verifier under primary policy A",
        "fixed": [
            "controlled records and labels",
            "scene folds",
            "fold-local evidence concentration",
            "evidence adapter and prior",
            "native non-vacuity score",
            "common eligibility",
            "non-provenance verifier predicates",
            "train-only threshold protocol",
            "denominator within each declared stratum",
        ],
        "primary_support": [0.10, 0.39],
        "fixed_support_sensitivity": [0.10, 0.35],
        "operating_target": TRAINING_TARGET,
        "bootstrap_unit": "scene",
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "retain_regardless_of_sign": True,
    }
    write_json(output / "PROTOCOL_LOCK.json", protocol)

    if uncompressed_sha256(DATA) != config["source_records_sha256"]:
        raise AssertionError("controlled source records changed")
    if uncompressed_sha256(LABELS) != config["evaluation_labels_sha256"]:
        raise AssertionError("controlled labels changed")
    records = read_records(DATA)
    label_rows = read_records(LABELS)
    if len(records) != 31_200 or len(label_rows) != 31_200:
        raise AssertionError("controlled denominator changed")
    labels = {str(row["record_id"]): str(row["preferred_contract"]) for row in label_rows}
    folds = scene_fold_map(records, int(config["fold_count"]))
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    rows_by_fusion = make_rows(records, labels, folds, concentrations)
    expected_counts = {
        "all_records": 31_200,
        "complete_records": 21_600,
    }
    summary_rows = []
    effect_rows = []
    operating_rows = []
    exact_rows = []
    route_rows = []
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "analysis_status": protocol["analysis_status"],
        "strata": {},
    }
    for stratum in expected_counts:
        cell_outer: dict[str, dict[int, list[dict[str, Any]]]] = {}
        cell_oof: dict[str, list[dict[str, Any]]] = {}
        curves = {}
        counts = {}
        for fusion in FUSIONS:
            for verifier in VERIFIERS:
                cell = f"{fusion}:{verifier}"
                outer = cell_rows_by_outer(rows_by_fusion[fusion], verifier, stratum)
                oof = held_out_rows(outer)
                if len(oof) != expected_counts[stratum]:
                    raise AssertionError(f"{cell}/{stratum}: {len(oof)} rows")
                curve, scene_counts, _ = controlled.evaluate_curve(
                    outer,
                    TARGETS,
                    len(concentrations),
                    method=cell,
                    table=f"2x2_{stratum}",
                    verifier=True,
                )
                cell_outer[cell] = outer
                cell_oof[cell] = oof
                curves[cell] = curve
                counts[cell] = scene_counts

        scenes = sorted({str(row["scene_id"]) for row in next(iter(cell_oof.values()))})
        if len(scenes) != 48:
            raise AssertionError(f"{stratum}: expected 48 scenes, found {len(scenes)}")
        stratum_result: dict[str, Any] = {"records": expected_counts[stratum], "supports": {}}
        for support_name, (low, high) in SUPPORTS.items():
            grid = np.linspace(low, high, 36)
            bootstrap, method_order, draws = controlled.bootstrap_naurc(
                counts,
                TARGETS,
                scenes,
                grid,
                replicates=BOOTSTRAPS,
                seed=SEED,
            )
            cells_payload = {}
            for cell in CELLS:
                point = controlled.interpolate_naurc(curves[cell], grid)
                row = {
                    "stratum": stratum,
                    "support": support_name,
                    "cell": cell,
                    "ncsAURC": point,
                    "bootstrap_mean": bootstrap[cell]["mean"],
                    "ci_low": bootstrap[cell]["ci_low"],
                    "ci_high": bootstrap[cell]["ci_high"],
                }
                summary_rows.append(row)
                cells_payload[cell] = row
                exact = exact_count_summary(cell_oof[cell], low, high)
                exact_row = {
                    "stratum": stratum,
                    "support": support_name,
                    "cell": cell,
                    **exact,
                }
                exact_rows.append(exact_row)
            effects = factorial_effects(
                method_order,
                draws,
                f"ncsAURC_{low:.2f}_{high:.2f}",
            )
            for row in effects:
                row.update({"stratum": stratum, "support": support_name})
                effect_rows.append(row)
            stratum_result["supports"][support_name] = {
                "cells": cells_payload,
                "effects": effects,
            }

        point_summaries, point_order, point_draws = point_bootstrap(
            counts,
            scenes,
            TRAINING_TARGET,
            BOOTSTRAPS,
            SEED + 1,
        )
        point_curve = {
            cell: next(row for row in curves[cell] if math.isclose(row["target"], TRAINING_TARGET))
            for cell in CELLS
        }
        for row in point_summaries:
            cell = str(row["cell"])
            row.update(
                {
                    "stratum": stratum,
                    "target": TRAINING_TARGET,
                    "admitted": int(point_curve[cell]["admitted"]),
                    "wrong": int(point_curve[cell]["wrong"]),
                    "correct": int(point_curve[cell]["correct"]),
                }
            )
            operating_rows.append(row)
        point_effects = {
            metric: factorial_effects(point_order, matrix, metric)
            for metric, matrix in point_draws.items()
        }
        for fusion in FUSIONS:
            route_rows.extend(route_transitions(fusion, rows_by_fusion[fusion], stratum))
        stratum_result["operating_point"] = point_summaries
        stratum_result["operating_point_effects"] = point_effects
        result["strata"][stratum] = stratum_result

    anchor_cell = next(
        row
        for row in summary_rows
        if row["stratum"] == "all_records"
        and row["support"] == "primary_0.10_0.39"
        and row["cell"] == "F1_registered:V1_full"
    )
    anchor_point = next(
        row
        for row in operating_rows
        if row["stratum"] == "all_records"
        and row["cell"] == "F1_registered:V1_full"
    )
    anchor = {
        "expected_ncsAURC": 0.08612194505919865,
        "observed_ncsAURC": float(anchor_cell["ncsAURC"]),
        "expected_admitted_at_target_0p13": 4043,
        "observed_admitted_at_target_0p13": int(anchor_point["admitted"]),
        "expected_wrong_at_target_0p13": 0,
        "observed_wrong_at_target_0p13": int(anchor_point["wrong"]),
    }
    anchor["pass"] = bool(
        math.isclose(
            anchor["observed_ncsAURC"],
            anchor["expected_ncsAURC"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and anchor["observed_admitted_at_target_0p13"]
        == anchor["expected_admitted_at_target_0p13"]
        and anchor["observed_wrong_at_target_0p13"]
        == anchor["expected_wrong_at_target_0p13"]
    )
    if not anchor["pass"]:
        raise AssertionError(f"registered PACT anchor changed: {anchor}")
    result["canonical_anchor"] = anchor
    write_json(output / "CANONICAL_ANCHOR_CHECK.json", anchor)

    write_csv(output / "ncsaurc_cells.csv", summary_rows)
    write_csv(output / "ncsaurc_factorial_effects.csv", effect_rows)
    write_csv(output / "operating_point_cells.csv", operating_rows)
    write_csv(output / "accepted_count_cells.csv", exact_rows)
    write_csv(output / "route_transitions.csv", route_rows)
    write_json(output / "MINIMAL_ATTRIBUTION_RESULT.json", result)
    inputs = {
        "source_records": {"path": DATA.relative_to(ROOT).as_posix(), "sha256": sha256(DATA)},
        "evaluation_labels": {"path": LABELS.relative_to(ROOT).as_posix(), "sha256": sha256(LABELS)},
        "config": {"path": CONFIG.relative_to(ROOT).as_posix(), "sha256": sha256(CONFIG)},
        "operator": {
            "path": (REPO / "src" / "action_admission" / "pcecf.py").relative_to(ROOT).as_posix(),
            "sha256": sha256(REPO / "src" / "action_admission" / "pcecf.py"),
        },
        "verifier": {
            "path": (REPO / "src" / "action_admission" / "verifier.py").relative_to(ROOT).as_posix(),
            "sha256": sha256(REPO / "src" / "action_admission" / "verifier.py"),
        },
        "analysis": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__).resolve())},
    }
    write_json(output / "INPUT_HASHES.json", inputs)
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "OUTPUT_MANIFEST.json"
    }
    write_json(output / "OUTPUT_MANIFEST.json", manifest)
    print(json.dumps({"status": "COMPLETE", "output": str(output), "anchor": anchor}, indent=2))


if __name__ == "__main__":
    main()
