"""Reproduce the primary shared-verifier comparison from source records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from action_admission import (
    CONTRACT_CLASSES,
    SourceOpinion,
    dirichlet_predict,
    graph_from_parent_sets,
    log_linear_posterior,
    restrict_dirichlet_input,
    verify_source_state,
)
from action_admission.baselines import (
    cautious_evidence_prediction,
    product_evidence_prediction,
    quality_weighted_prediction,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "controlled" / "source_records.jsonl.gz"
LABELS = ROOT / "data" / "controlled" / "evaluation_labels.jsonl.gz"
CONFIG = ROOT / "configs" / "controlled_study.json"
OUTPUT = ROOT / "outputs" / "controlled_study"
METHODS = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
)
SOURCES = ("language", "geometry", "risk")
FUSION_SOURCE_ROLES = ("language", "vision", "geometry", "risk")


def read_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def uncompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_id(record_id: str) -> str:
    return record_id.split("__", 1)[0]


def scene_id(record: Mapping[str, Any]) -> str:
    return str(record["metadata"]["scene_id"])


def scene_fold_map(
    records: list[Mapping[str, Any]],
    fold_count: int,
) -> dict[str, int]:
    scenes = sorted({scene_id(record) for record in records})
    if len(scenes) != 48:
        raise ValueError(f"Expected 48 scenes, found {len(scenes)}")
    scene_folds = {
        scene: index % fold_count
        for index, scene in enumerate(scenes)
    }
    return {
        cluster_id(str(record["record_id"])): scene_folds[scene_id(record)]
        for record in records
    }


def source_parents(record: Mapping[str, Any]) -> dict[str, list[str]]:
    metadata = record["metadata"]
    scene = str(metadata["scene_id"])
    occlusion = str(metadata["occlusion_band"])
    language = record["sources"]["language"]
    route = language.get("source_route", {})
    command_hash = str(
        route.get("payload_text_hash")
        or route.get("current_command_hash")
        or ""
    )
    return {
        "language": [f"command:{command_hash}"],
        "geometry": [
            f"scene:{scene}",
            f"relation:{scene}:{metadata['relation_label']}",
            f"occlusion:{scene}:{occlusion}",
        ],
        "risk": [
            f"scene:{scene}",
            f"risk:{scene}:{metadata['risk_band']}",
            f"occlusion:{scene}:{occlusion}",
        ],
    }


def source_opinions(record: Mapping[str, Any]) -> list[SourceOpinion]:
    opinions = []
    for source in FUSION_SOURCE_ROLES:
        payload = record["sources"].get(source, {})
        opinions.append(
            SourceOpinion(
                source=source,
                probabilities=payload.get("probabilities", {}),
                quality=float(payload.get("quality", 0.0)),
                conflict=float(payload.get("conflict", 0.0)),
                missing=not payload or bool(payload.get("missing", False)),
            )
        )
    return opinions


def prediction_row(
    record: Mapping[str, Any],
    fold: int,
    *,
    method: str,
    concentration: float | None = None,
) -> dict[str, Any]:
    registered_graph = graph_from_parent_sets(source_parents(record))
    if method == "nested_evidential_composition":
        if concentration is None:
            raise ValueError("Dirichlet concentration is required")
        prediction = dirichlet_predict(
            restrict_dirichlet_input(record),
            concentration=concentration,
        )
        predicted = prediction.predicted_contract
        score = prediction.selection_score
        eligible = prediction.eligible
    elif method == "quality_weighted_fusion":
        prediction = quality_weighted_prediction(record)
        predicted = prediction.predicted_contract
        score = prediction.confidence
        eligible = True
    elif method == "product_evidence_fusion":
        prediction = product_evidence_prediction(record)
        predicted = prediction.predicted_contract
        score = prediction.confidence
        eligible = True
    elif method == "cautious_evidence_fusion":
        prediction = cautious_evidence_prediction(record)
        predicted = prediction.predicted_contract
        score = prediction.confidence
        eligible = True
    elif method in {"lineage_unaware_pooling", "registered_lineage_pooling"}:
        fusion_graph = (
            {} if method == "lineage_unaware_pooling" else registered_graph
        )
        posterior = log_linear_posterior(
            source_opinions(record),
            fusion_graph,
        )
        predicted = max(CONTRACT_CLASSES, key=posterior.__getitem__)
        score = posterior[predicted]
        eligible = max(
            float(record["sources"].get(source, {}).get("conflict", 0.0))
            for source in SOURCES
        ) <= 0.65
    else:
        raise ValueError(method)

    verification = verify_source_state(
        record,
        predicted,
        registered_graph,
    )
    return {
        "record_id": str(record["record_id"]),
        "scene_id": scene_id(record),
        "fold": fold,
        "score": float(score),
        "eligible": bool(eligible),
        "verifier_pass": verification.admissible,
        "predicted_contract": predicted,
    }


def join_evaluation_labels(
    predictions: list[dict[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    joined = []
    for prediction in predictions:
        record_id = str(prediction["record_id"])
        if record_id not in labels_by_id:
            raise ValueError(f"Missing evaluation label for {record_id}")
        label = labels_by_id[record_id]
        if str(label["scene_id"]) != str(prediction["scene_id"]):
            raise ValueError(f"Scene identifier mismatch for {record_id}")
        joined.append(
            {
                **prediction,
                "preferred_contract": str(label["preferred_contract"]),
            }
        )
    if len(joined) != len(labels_by_id):
        raise ValueError("Prediction and evaluation-label counts differ")
    return joined


def selection_threshold(
    rows: list[dict[str, Any]],
    target: float,
) -> float:
    eligible = [
        row
        for row in rows
        if row["eligible"] and row["verifier_pass"]
    ]
    eligible.sort(key=lambda row: (-row["score"], row["record_id"]))
    count = int(round(target * len(rows)))
    if not eligible or count <= 0:
        return float("inf")
    return float(eligible[min(count, len(eligible)) - 1]["score"])


def metrics(counts: Mapping[str, float]) -> dict[str, float]:
    n = float(counts["n"])
    admitted = float(counts["admitted"])
    wrong = float(counts["wrong"])
    correct = float(counts["correct"])
    return {
        "n": n,
        "admitted": admitted,
        "wrong": wrong,
        "correct": correct,
        "coverage": admitted / n,
        "wrong_all": wrong / n,
        "wrong_admitted": wrong / admitted if admitted else 0.0,
        "correct_all": correct / n,
    }


def evaluate(
    method: str,
    rows_by_outer_fold: Mapping[int, list[dict[str, Any]]],
    targets: tuple[float, ...],
    fold_count: int,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[float, str], dict[str, float]],
]:
    curve = []
    scene_counts: dict[tuple[float, str], dict[str, float]] = {}
    for target in targets:
        total = defaultdict(float)
        by_scene: dict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for fold in range(fold_count):
            rows = rows_by_outer_fold[fold]
            train = [row for row in rows if row["fold"] != fold]
            test = [row for row in rows if row["fold"] == fold]
            threshold = selection_threshold(train, target)
            for row in test:
                accepted = bool(
                    row["eligible"]
                    and row["verifier_pass"]
                    and row["score"] >= threshold
                )
                correct = bool(
                    accepted
                    and row["predicted_contract"] == row["preferred_contract"]
                )
                wrong = bool(accepted and not correct)
                for bucket in (total, by_scene[row["scene_id"]]):
                    bucket["n"] += 1
                    bucket["admitted"] += int(accepted)
                    bucket["wrong"] += int(wrong)
                    bucket["correct"] += int(correct)
        curve.append(
            {
                "method": method,
                "target_coverage": target,
                **metrics(total),
            }
        )
        for scene, counts in by_scene.items():
            scene_counts[(target, scene)] = dict(counts)
    return curve, scene_counts


def frontier(
    curve: list[Mapping[str, Any]],
    common_grid: np.ndarray,
) -> np.ndarray:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in curve:
        grouped[float(row["coverage"])].append(
            float(row["wrong_admitted"])
        )
    coverage = np.asarray(sorted(grouped))
    risk = np.asarray(
        [
            sum(grouped[value]) / len(grouped[value])
            for value in coverage
        ]
    )
    if coverage[0] > common_grid[0] or coverage[-1] < common_grid[-1]:
        raise ValueError(
            "Risk-coverage curve does not span the configured interval"
        )
    return np.interp(common_grid, coverage, risk)


def bootstrap_aurc(
    counts_by_method: Mapping[
        str,
        Mapping[tuple[float, str], Mapping[str, float]],
    ],
    targets: tuple[float, ...],
    scenes: list[str],
    common_grid: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    fields = ("n", "admitted", "wrong", "correct")
    values = np.zeros(
        (len(METHODS), len(targets), len(scenes), len(fields)),
        dtype=float,
    )
    for method_index, method in enumerate(METHODS):
        for target_index, target in enumerate(targets):
            for scene_index, scene in enumerate(scenes):
                counts = counts_by_method[method][(target, scene)]
                values[method_index, target_index, scene_index, :] = [
                    counts[field] for field in fields
                ]

    rng = np.random.default_rng(seed)
    draws = np.zeros((replicates, len(METHODS)), dtype=float)
    for replicate in range(replicates):
        weights = rng.multinomial(
            len(scenes),
            np.full(len(scenes), 1.0 / len(scenes)),
        )
        totals = np.tensordot(values, weights, axes=(2, 0))
        n = totals[:, :, 0]
        admitted = totals[:, :, 1]
        wrong = totals[:, :, 2]
        coverage = np.divide(
            admitted,
            n,
            out=np.zeros_like(admitted),
            where=n > 0,
        )
        risk = np.divide(
            wrong,
            admitted,
            out=np.zeros_like(wrong),
            where=admitted > 0,
        )
        for method_index in range(len(METHODS)):
            interpolated = frontier(
                [
                    {
                        "coverage": coverage[method_index, target_index],
                        "wrong_admitted": risk[
                            method_index,
                            target_index,
                        ],
                    }
                    for target_index in range(len(targets))
                ],
                common_grid,
            )
            draws[replicate, method_index] = (
                np.trapezoid(interpolated, common_grid)
                / (common_grid[-1] - common_grid[0])
            )

    return [
        {
            "method": method,
            "common_coverage_min": float(common_grid[0]),
            "common_coverage_max": float(common_grid[-1]),
            "normalized_aurc": float(np.mean(draws[:, index])),
            "ci_low": float(np.quantile(draws[:, index], 0.025)),
            "ci_high": float(np.quantile(draws[:, index], 0.975)),
            "bootstrap_replicates": replicates,
        }
        for index, method in enumerate(METHODS)
    ]


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
    observed_hash = uncompressed_sha256(arguments.data)
    observed_label_hash = uncompressed_sha256(arguments.labels)
    if observed_hash != config["source_records_sha256"]:
        raise ValueError("Source-record checksum does not match the configuration")
    if observed_label_hash != config["evaluation_labels_sha256"]:
        raise ValueError("Evaluation-label checksum does not match the configuration")

    records = read_records(arguments.data)
    labels = read_records(arguments.labels)
    if len(records) != 31_200:
        raise ValueError(f"Expected 31,200 records, found {len(records)}")
    if len(labels) != len(records):
        raise ValueError("Source-record and evaluation-label counts differ")
    labels_by_id = {str(row["record_id"]): row for row in labels}
    if len(labels_by_id) != len(labels):
        raise ValueError("Evaluation-label identifiers are not unique")
    record_ids = {str(record["record_id"]) for record in records}
    if set(labels_by_id) != record_ids:
        raise ValueError("Source-record and evaluation-label identifiers differ")
    for record in records:
        label = labels_by_id[str(record["record_id"])]
        if str(label["scene_id"]) != scene_id(record):
            raise ValueError("Scene identifiers differ between data files")

    fold_count = int(config["fold_count"])
    folds = scene_fold_map(records, fold_count)
    concentrations = {
        int(fold): float(value)
        for fold, value in config[
            "dirichlet_concentration_by_fold"
        ].items()
    }

    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]] = {}
    nested_by_concentration = {
        concentration: join_evaluation_labels(
            [
                prediction_row(
                    record,
                    folds[cluster_id(str(record["record_id"]))],
                    method="nested_evidential_composition",
                    concentration=concentration,
                )
                for record in records
            ],
            labels_by_id,
        )
        for concentration in sorted(set(concentrations.values()))
    }
    rows_by_method["nested_evidential_composition"] = {
        fold: nested_by_concentration[concentrations[fold]]
        for fold in range(fold_count)
    }
    for method in METHODS:
        if method == "nested_evidential_composition":
            continue
        rows = join_evaluation_labels(
            [
                prediction_row(
                    record,
                    folds[cluster_id(str(record["record_id"]))],
                    method=method,
                )
                for record in records
            ],
            labels_by_id,
        )
        rows_by_method[method] = {
            fold: rows
            for fold in range(fold_count)
        }

    targets = tuple(round(index / 100.0, 2) for index in range(1, 61))
    curve: list[dict[str, Any]] = []
    counts_by_method = {}
    for method in METHODS:
        method_curve, method_counts = evaluate(
            method,
            rows_by_method[method],
            targets,
            fold_count,
        )
        curve.extend(method_curve)
        counts_by_method[method] = method_counts

    common_grid = np.linspace(*config["common_coverage"], 36)
    summary = bootstrap_aurc(
        counts_by_method,
        targets,
        sorted({scene_id(record) for record in records}),
        common_grid,
        int(config["bootstrap_replicates"]),
        int(config["bootstrap_seed"]),
    )
    fixed_target = float(config["fixed_target"])
    fixed_rows = [
        row
        for row in curve
        if float(row["target_coverage"]) == fixed_target
    ]

    expected = config["expected"]
    by_method = {row["method"]: row for row in summary}
    for method in METHODS:
        if not np.isclose(
            by_method[method]["normalized_aurc"],
            expected[method]["normalized_aurc"],
            atol=1e-12,
        ):
            raise AssertionError(f"nAURC changed for {method}")
    nested_fixed = next(
        row
        for row in fixed_rows
        if row["method"] == "nested_evidential_composition"
    )
    nested_expected = expected["nested_evidential_composition"]
    if (
        int(nested_fixed["admitted"])
        != int(nested_expected["fixed_target_admitted"])
        or int(nested_fixed["wrong"])
        != int(nested_expected["fixed_target_wrong"])
    ):
        raise AssertionError("Fixed-target decision counts changed")

    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output / "risk_coverage.csv", curve)
    write_csv(arguments.output / "aurc_summary.csv", summary)
    write_csv(arguments.output / "fixed_target_summary.csv", fixed_rows)
    (arguments.output / "summary.json").write_text(
        json.dumps(
            {
                "records": len(records),
                "source_records_sha256": observed_hash,
                "evaluation_labels_sha256": observed_label_hash,
                "aurc": summary,
                "fixed_target": fixed_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("Primary controlled-study reproduction completed successfully.")


if __name__ == "__main__":
    main()
