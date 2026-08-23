"""Reproduce the scene-grouped inner selection used by the controlled study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from controlled_study import (
    CONFIG,
    DATA,
    LABELS,
    ROOT,
    cluster_id,
    join_evaluation_labels,
    prediction_row,
    read_records,
    scene_fold_map,
    selection_threshold,
    uncompressed_sha256,
)

OUTPUT = ROOT / "outputs" / "nested_selection"
POLICY_TARGETS = (0.10, 0.13, 0.15)
CONCENTRATIONS = (4.0, 8.0, 12.0, 16.0, 24.0)
BASE_METHODS = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
)
CANDIDATES = BASE_METHODS + tuple(
    f"nested_evidential_composition_c{int(value)}"
    for value in CONCENTRATIONS
)


def candidate_spec(name: str) -> tuple[str, float | None]:
    prefix = "nested_evidential_composition_c"
    if name.startswith(prefix):
        return "nested_evidential_composition", float(name.removeprefix(prefix))
    return name, None


def counts_at_threshold(
    rows: list[Mapping[str, Any]],
    threshold: float,
) -> dict[str, int]:
    counts = {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
    for row in rows:
        accepted = bool(
            row["eligible"]
            and row["verifier_pass"]
            and row["score"] >= threshold
        )
        correct = bool(
            accepted
            and row["predicted_contract"] == row["preferred_contract"]
        )
        counts["n"] += 1
        counts["admitted"] += int(accepted)
        counts["wrong"] += int(accepted and not correct)
        counts["correct"] += int(correct)
    return counts


def add_counts(total: dict[str, int], update: Mapping[str, int]) -> None:
    for field in total:
        total[field] += int(update[field])


def operating_metrics(counts: Mapping[str, int]) -> dict[str, float]:
    n = float(counts["n"])
    admitted = float(counts["admitted"])
    wrong = float(counts["wrong"])
    correct = float(counts["correct"])
    return {
        "coverage": admitted / n,
        "wrong_all": wrong / n,
        "correct_all": correct / n,
        "expected_cost": (5.0 * wrong + 0.5 * (n - admitted)) / n,
    }


def inner_point(
    rows: list[dict[str, Any]],
    outer: int,
    target: float,
    fold_count: int,
) -> dict[str, float]:
    total = {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
    for inner in range(fold_count):
        if inner == outer:
            continue
        train = [row for row in rows if row["fold"] not in {outer, inner}]
        validate = [row for row in rows if row["fold"] == inner]
        threshold = selection_threshold(train, target)
        add_counts(total, counts_at_threshold(validate, threshold))
    return operating_metrics(total)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if uncompressed_sha256(arguments.data) != config["source_records_sha256"]:
        raise ValueError("Source-record checksum does not match the configuration")
    if uncompressed_sha256(arguments.labels) != config["evaluation_labels_sha256"]:
        raise ValueError("Evaluation-label checksum does not match the configuration")

    records = read_records(arguments.data)
    labels = read_records(arguments.labels)
    labels_by_id = {str(row["record_id"]): row for row in labels}
    if len(records) != 31_200 or len(labels_by_id) != len(records):
        raise ValueError("Expected 31,200 one-to-one source and label records")

    fold_count = int(config["fold_count"])
    folds = scene_fold_map(records, fold_count)
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        method, concentration = candidate_spec(candidate)
        predictions = [
            prediction_row(
                record,
                folds[cluster_id(str(record["record_id"]))],
                method=method,
                concentration=concentration,
            )
            for record in records
        ]
        rows_by_candidate[candidate] = join_evaluation_labels(
            predictions,
            labels_by_id,
        )

    order = {candidate: index for index, candidate in enumerate(CANDIDATES)}
    score_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    expected = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    fixed_target = float(config["fixed_target"])

    for outer in range(fold_count):
        fold_scores = []
        for candidate in CANDIDATES:
            points = [
                inner_point(
                    rows_by_candidate[candidate],
                    outer,
                    target,
                    fold_count,
                )
                for target in POLICY_TARGETS
            ]
            row: dict[str, Any] = {
                "outer_fold": outer,
                "candidate": candidate,
                "mean_expected_cost": sum(
                    point["expected_cost"] for point in points
                ) / len(points),
                "mean_wrong_all": sum(
                    point["wrong_all"] for point in points
                ) / len(points),
                "mean_correct_all": sum(
                    point["correct_all"] for point in points
                ) / len(points),
            }
            for target, point in zip(POLICY_TARGETS, points):
                suffix = str(target).replace(".", "p")
                for field, value in point.items():
                    row[f"{field}_{suffix}"] = value
            score_rows.append(row)
            fold_scores.append(row)

        winner = min(
            fold_scores,
            key=lambda row: (
                row["mean_expected_cost"],
                row["mean_wrong_all"],
                -row["mean_correct_all"],
                order[row["candidate"]],
            ),
        )
        method, concentration = candidate_spec(str(winner["candidate"]))
        if method != "nested_evidential_composition":
            raise AssertionError(f"Unexpected selected backbone: {winner['candidate']}")
        if concentration != expected[outer]:
            raise AssertionError(
                f"Fold {outer} selected c={concentration}, expected c={expected[outer]}"
            )
        train = [
            row
            for row in rows_by_candidate[str(winner["candidate"])]
            if row["fold"] != outer
        ]
        selection_rows.append(
            {
                "outer_fold": outer,
                "selected_candidate": winner["candidate"],
                "dirichlet_concentration": concentration,
                "mean_inner_expected_cost": winner["mean_expected_cost"],
                "mean_inner_wrong_all": winner["mean_wrong_all"],
                "mean_inner_correct_all": winner["mean_correct_all"],
                "fixed_target_threshold": selection_threshold(train, fixed_target),
            }
        )

    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output / "inner_candidate_scores.csv", score_rows)
    write_csv(arguments.output / "outer_fold_selection.csv", selection_rows)
    summary = {
        "candidate_order": list(CANDIDATES),
        "concentration_grid": list(CONCENTRATIONS),
        "fold_count": fold_count,
        "policy_targets": list(POLICY_TARGETS),
        "selection_objective": [
            "mean inner-fold expected cost",
            "mean wrong/all",
            "negative mean correct/all",
            "candidate order",
        ],
        "selected_concentrations": [
            row["dirichlet_concentration"] for row in selection_rows
        ],
        "source_records_sha256": config["source_records_sha256"],
        "evaluation_labels_sha256": config["evaluation_labels_sha256"],
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("Nested scene-grouped selection reproduced successfully.")


if __name__ == "__main__":
    main()
