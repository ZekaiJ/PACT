"""Run compact sanity and shared-score controls for the controlled study."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path[:0] = [str(REPOSITORY / "src"), str(REPOSITORY / "experiments")]

import pcecf_study as study  # noqa: E402
from action_admission import CONTRACT_CLASSES  # noqa: E402
from controlled_study import cluster_id, read_records, scene_fold_map  # noqa: E402

OUTPUT = REPOSITORY / "outputs" / "strong_paper_controls"
OOF = REPOSITORY / "outputs" / "pcecf_study" / "oof_predictions.jsonl.gz"
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
GRID = np.linspace(0.10, 0.39, 30)
SOURCE_ORDER = ("language", "geometry", "risk")


def read_oof() -> list[dict[str, Any]]:
    with gzip.open(OOF, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    verifier = row["verifier_outcome"]
    return {
        "record_id": str(row["record_id"]),
        "scene_id": str(row["scene_id"]),
        "fold": int(row["outer_fold"]),
        "score": float(row["score"]),
        "eligible": bool(row["common_eligible"]),
        "common_eligible": bool(row["common_eligible"]),
        "native_eligible": bool(row["native_eligible"]),
        "verifier_pass": bool(verifier["admissible"]),
        "verifier_route": str(verifier["route"]),
        "verifier_reason": str(verifier["reason"]),
        "fold_local_concentration": row.get("fold_local_concentration"),
        "predicted_contract": str(row["prediction"]),
        "preferred_contract": str(row["y"]),
        "probabilities": dict(row["probabilities"]),
    }


def method_rows(oof: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof:
        grouped[str(row["method"])].append(evaluation_row(row))
    expected = 31_200
    if any(len(rows) != expected for rows in grouped.values()):
        raise AssertionError("each method must contain 31,200 OOF rows")
    return dict(grouped)


def curve_summary(rows_by_outer: Mapping[int, list[dict[str, Any]]], method: str, *, verifier: bool = False) -> dict[str, Any]:
    curve, _, _ = study.evaluate_curve(
        rows_by_outer,
        TARGETS,
        5,
        method=method,
        table="strong_paper_control",
        verifier=verifier,
    )
    return {
        "method": method,
        "verifier": verifier,
        "csaurc_0p10_0p39": study.interpolate_naurc(curve, GRID),
        "support_min": min(float(row["coverage"]) for row in curve),
        "support_max": max(float(row["coverage"]) for row in curve),
    }


def source_rows(records: list[dict[str, Any]], labels: Mapping[str, str], folds: Mapping[str, int], source: str, *, common_gate: bool) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        record_id = str(record["record_id"])
        payload = record.get("sources", {}).get(source, {})
        missing = not payload or bool(payload.get("missing", False))
        probabilities = payload.get("probabilities", {}) if isinstance(payload, Mapping) else {}
        values = {label: float(probabilities.get(label, 0.0)) for label in CONTRACT_CLASSES}
        total = sum(values.values())
        valid = (not missing and total > 0.0 and payload.get("schema_valid", True) is not False)
        if total > 0.0:
            values = {label: value / total for label, value in values.items()}
        else:
            values = {label: 1.0 / len(CONTRACT_CLASSES) for label in CONTRACT_CLASSES}
        predicted = max(CONTRACT_CLASSES, key=values.__getitem__)
        eligible = study.common_eligibility(record) if common_gate else valid
        rows.append({
            "record_id": record_id,
            "scene_id": study.scene_id(record),
            "fold": int(folds[cluster_id(record_id)]),
            "score": float(values[predicted]),
            "eligible": bool(eligible),
            "common_eligible": bool(eligible),
            "native_eligible": bool(valid),
            "verifier_pass": True,
            "verifier_route": "none",
            "verifier_reason": "not_applied",
            "fold_local_concentration": None,
            "predicted_contract": predicted,
            "preferred_contract": labels[record_id],
            "probabilities": values,
        })
    return rows


def training_accuracy(rows: list[dict[str, Any]], held_out_fold: int) -> float:
    train = [row for row in rows if int(row["fold"]) != held_out_fold and row["eligible"]]
    return sum(row["predicted_contract"] == row["preferred_contract"] for row in train) / len(train)


def sanity_controls(records: list[dict[str, Any]], labels: Mapping[str, str], folds: Mapping[str, int]) -> list[dict[str, Any]]:
    per_source = {
        source: source_rows(records, labels, folds, source, common_gate=False)
        for source in SOURCE_ORDER
    }
    language_common = source_rows(records, labels, folds, "language", common_gate=True)
    controls: dict[str, dict[int, list[dict[str, Any]]]] = {
        "language_only": {fold: per_source["language"] for fold in range(5)},
        "language_plus_common_completeness": {fold: language_common for fold in range(5)},
    }
    selected = {}
    best_by_outer = {}
    for fold in range(5):
        best = max(SOURCE_ORDER, key=lambda source: (training_accuracy(per_source[source], fold), -SOURCE_ORDER.index(source)))
        best_by_outer[str(fold)] = best
        selected[fold] = per_source[best]
    controls["best_single_source_outer_train_accuracy"] = selected
    result = [curve_summary(rows, name) for name, rows in controls.items()]
    for row in result:
        row["selected_source_by_outer_fold"] = best_by_outer if row["method"].startswith("best_single") else None
    return result


def posterior_features(row: Mapping[str, Any], observed_count: int) -> list[float]:
    probabilities = np.asarray([float(row["probabilities"][label]) for label in CONTRACT_CLASSES])
    ordered = np.sort(probabilities)[::-1]
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return [*probabilities.tolist(), entropy, float(ordered[0]), float(ordered[0] - ordered[1]), float(observed_count)]


def fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    for _ in range(30):
        linear = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        variance = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = design.T @ (probability - y) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return weights, mean, scale


def predict_logistic(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, scale = model
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))


def shared_score_controls(grouped: Mapping[str, list[dict[str, Any]]], observed: Mapping[str, int]) -> list[dict[str, Any]]:
    methods = tuple(sorted(grouped))
    by_outer: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in methods}
    for outer in range(5):
        train_rows = [row for method in methods for row in grouped[method] if int(row["fold"]) != outer]
        x_train = np.asarray([posterior_features(row, observed[row["record_id"]]) for row in train_rows])
        y_train = np.asarray([row["predicted_contract"] == row["preferred_contract"] for row in train_rows], dtype=int)
        model = fit_logistic(x_train, y_train)
        for method in methods:
            method_rows = grouped[method]
            features = np.asarray(
                [posterior_features(row, observed[row["record_id"]]) for row in method_rows]
            )
            scores = predict_logistic(features, model)
            scored = []
            for row, score in zip(method_rows, scores, strict=True):
                copy = dict(row)
                copy["score"] = float(score)
                scored.append(copy)
            by_outer[method][outer] = scored
    result = []
    for method in methods:
        result.append(curve_summary(by_outer[method], f"{method}:shared_error_score", verifier=False))
        result.append(curve_summary(by_outer[method], f"{method}:shared_error_score", verifier=True))
    return result


def subset_controls(grouped: Mapping[str, list[dict[str, Any]]], complete: Mapping[str, bool]) -> list[dict[str, Any]]:
    result = []
    for method in ("pcecf", "nested_evidential_composition", "product_evidence_fusion"):
        for subset_name, keep in (("complete", True), ("missing", False)):
            rows = [row for row in grouped[method] if complete[row["record_id"]] is keep]
            by_outer = {fold: rows for fold in range(5)}
            result.append({"subset": subset_name, **curve_summary(by_outer, method)})
        gated = []
        for row in grouped[method]:
            copy = dict(row)
            copy["eligible"] = bool(complete[row["record_id"]])
            gated.append(copy)
        result.append({"subset": "all_with_complete_catalog_gate", **curve_summary({fold: gated for fold in range(5)}, method)})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = read_records(study.DATA)
    label_rows = read_records(study.LABELS)
    labels = {str(row["record_id"]): str(row["preferred_contract"]) for row in label_rows}
    folds = scene_fold_map(records, 5)
    oof = read_oof()
    grouped = method_rows(oof)
    observed = {str(record["record_id"]): study.observed_source_count(record) for record in records}
    complete = {record_id: count == len(study.SOURCE_NAMES) for record_id, count in observed.items()}

    sanity = sanity_controls(records, labels, folds)
    subsets = subset_controls(grouped, complete)
    shared = shared_score_controls(grouped, observed)
    write_csv(OUTPUT / "sanity_baselines.csv", sanity)
    write_csv(OUTPUT / "complete_missing_attribution.csv", subsets)
    write_csv(OUTPUT / "shared_score_control.csv", shared)
    summary = {
        "status": "PASS",
        "records": len(records),
        "scenes": len({study.scene_id(record) for record in records}),
        "shared_score": "one outer-training-fitted logistic correctness predictor using posterior, entropy, peak, margin, and observed-source count; method identity excluded",
        "sanity_baselines": sanity,
        "complete_missing_attribution": subsets,
        "shared_score_control": shared,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert len(records) == 31_200 and len(oof) == 218_400
    assert all(math.isfinite(float(row["csaurc_0p10_0p39"])) for row in sanity + subsets + shared)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
