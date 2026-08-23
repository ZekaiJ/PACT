"""Apply the frozen shared score to the two topology-aware comparators."""

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
from typing import Any, Mapping

import numpy as np


MAIN_METHODS = (
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "pcecf",
)
TARGET_METHODS = (
    "registered_lineage_pooling",
    "hierarchical_cautious_cumulative",
)
STATIC_SCORES = (
    "native",
    "posterior_peak",
    "top_two_margin",
    "inverse_normalized_entropy",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = [
        str(repository / "analysis"),
        str(repository / "experiments"),
        str(repository / "src"),
    ]
    import pcecf_study as study  # noqa: PLC0415
    import score_verifier_factorial as factorial  # noqa: PLC0415
    import strong_paper_controls as controls  # noqa: PLC0415
    from controlled_study import read_records  # noqa: PLC0415

    oof_path = repository / "results" / "p0_estimand_closure" / "v1" / "inputs" / "oof_predictions.jsonl.gz"
    reference_path = repository / "results" / "score_verifier_factorial" / "score_factorial.csv"
    fusion_reference_path = repository / "results" / "reference" / "pcecf_fusion_only.csv"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_pcecf: list[dict[str, Any]] = []
    with gzip.open(oof_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            method = str(raw["method"])
            if method in {*MAIN_METHODS, "registered_lineage_pooling"}:
                grouped[method].append(controls.evaluation_row(raw))
            if method == "pcecf":
                raw_pcecf.append(raw)

    assert all(len(grouped[method]) == 31_200 for method in (*MAIN_METHODS, "registered_lineage_pooling"))
    folds = {row["record_id"]: int(row["fold"]) for row in grouped["product_evidence_fusion"]}
    concentrations: dict[int, float] = {}
    for row in raw_pcecf:
        fold = int(row["outer_fold"])
        concentration = float(row["fold_local_concentration"])
        if fold in concentrations:
            assert concentrations[fold] == concentration
        concentrations[fold] = concentration
    assert set(concentrations) == set(range(5))

    records = read_records(study.DATA)
    labels = {
        str(row["record_id"]): str(row["preferred_contract"])
        for row in read_records(study.LABELS)
    }
    observed = {
        str(record["record_id"]): study.observed_source_count(record)
        for record in records
    }

    def generate(method: str, concentration: float) -> list[dict[str, Any]]:
        rows = []
        for record in records:
            record_id = str(record["record_id"])
            prediction, probabilities, score, native_eligible = study.predict_record(
                record, method, concentration
            )
            common_eligible = study.common_eligibility(record)
            rows.append(
                {
                    "record_id": record_id,
                    "scene_id": study.scene_id(record),
                    "fold": folds[record_id],
                    "score": float(score),
                    "eligible": bool(common_eligible),
                    "common_eligible": bool(common_eligible),
                    "native_eligible": bool(native_eligible),
                    "verifier_pass": True,
                    "verifier_route": "not_applied",
                    "verifier_reason": "not_applied",
                    "fold_local_concentration": concentration if method in {"pcecf", "nested_evidential_composition"} else None,
                    "predicted_contract": str(prediction),
                    "preferred_contract": labels[record_id],
                    "probabilities": dict(probabilities),
                }
            )
        return rows

    generated = {
        (method, concentration): generate(method, concentration)
        for method in ("nested_evidential_composition", "pcecf")
        for concentration in sorted(set(concentrations.values()))
    }
    hierarchy = generate("hierarchical_cautious_cumulative", 4.0)

    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]] = {
        "product_evidence_fusion": {outer: grouped["product_evidence_fusion"] for outer in range(5)},
        "cautious_evidence_fusion": {outer: grouped["cautious_evidence_fusion"] for outer in range(5)},
        "registered_lineage_pooling": {outer: grouped["registered_lineage_pooling"] for outer in range(5)},
        "hierarchical_cautious_cumulative": {outer: hierarchy for outer in range(5)},
        "nested_evidential_composition": {
            outer: generated[("nested_evidential_composition", concentrations[outer])]
            for outer in range(5)
        },
        "pcecf": {
            outer: generated[("pcecf", concentrations[outer])]
            for outer in range(5)
        },
    }

    models = {}
    model_rows = []
    for outer in range(5):
        train = [
            row
            for method in MAIN_METHODS
            for row in rows_by_method[method][outer]
            if int(row["fold"]) != outer
        ]
        x_train = np.asarray(
            [controls.posterior_features(row, observed[row["record_id"]]) for row in train]
        )
        y_train = np.asarray(
            [row["predicted_contract"] == row["preferred_contract"] for row in train],
            dtype=int,
        )
        model = controls.fit_logistic(x_train, y_train)
        models[outer] = model
        weights, mean, scale = model
        model_rows.append(
            {
                "outer_fold": outer,
                "training_methods": ";".join(MAIN_METHODS),
                "training_rows": len(train),
                "concentration": concentrations[outer],
                "weights": json.dumps(weights.tolist()),
                "feature_mean": json.dumps(mean.tolist()),
                "feature_scale": json.dumps(scale.tolist()),
            }
        )

    def shared_row(method: str) -> dict[str, Any]:
        scored_by_outer = {}
        for outer in range(5):
            rows = rows_by_method[method][outer]
            features = np.asarray(
                [controls.posterior_features(row, observed[row["record_id"]]) for row in rows]
            )
            scores = controls.predict_logistic(features, models[outer])
            scored = []
            for row, score in zip(rows, scores, strict=True):
                copy = dict(row)
                copy["score"] = float(score)
                scored.append(copy)
            scored_by_outer[outer] = scored
        return factorial.curve_row(
            scored_by_outer,
            method=method,
            score_name="shared_outer_train_logistic",
            verifier=False,
            subset="all",
        )

    all_rows = []
    for method in (*MAIN_METHODS, *TARGET_METHODS):
        for score_name in STATIC_SCORES:
            scored = {
                outer: factorial.rescore(rows_by_method[method][outer], score_name)
                for outer in range(5)
            }
            all_rows.append(
                factorial.curve_row(
                    scored,
                    method=method,
                    score_name=score_name,
                    verifier=False,
                    subset="all",
                )
            )
        all_rows.append(shared_row(method))

    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference = {
            (row["method"], row["score"]): float(row["csaurc_0p10_0p39"])
            for row in csv.DictReader(handle)
            if row["verifier"] == "False" and row["method"] in MAIN_METHODS
        }
    reproduced = {
        (row["method"], row["score"]): float(row["csaurc_0p10_0p39"])
        for row in all_rows
        if row["method"] in MAIN_METHODS
    }
    assert reference.keys() == reproduced.keys()
    assert max(abs(reference[key] - reproduced[key]) for key in reference) <= 1e-12

    with fusion_reference_path.open(encoding="utf-8", newline="") as handle:
        native_reference = {
            row["method"]: float(row["normalized_aurc_point"])
            for row in csv.DictReader(handle)
            if row["method"] in TARGET_METHODS
        }
    native_reproduced = {
        row["method"]: float(row["csaurc_0p10_0p39"])
        for row in all_rows
        if row["method"] in TARGET_METHODS and row["score"] == "native"
    }
    assert native_reference.keys() == native_reproduced.keys()
    assert max(abs(native_reference[key] - native_reproduced[key]) for key in native_reference) <= 1e-12

    parity_rows = [row for row in all_rows if row["method"] in TARGET_METHODS]
    panel_rows = []
    for method in TARGET_METHODS:
        values = {row["score"]: row["csaurc_0p10_0p39"] for row in parity_rows if row["method"] == method}
        panel_rows.append(
            {
                "method": method,
                "native": values["native"],
                "peak": values["posterior_peak"],
                "margin": values["top_two_margin"],
                "entropy": values["inverse_normalized_entropy"],
                "shared": values["shared_outer_train_logistic"],
            }
        )

    write_csv(output / "common_score_parity.csv", parity_rows)
    write_csv(output / "table_panel_b_rows.csv", panel_rows)
    write_csv(output / "shared_score_models.csv", model_rows)
    protocol = {
        "status": "PASS",
        "analysis": "frozen-output common-score parity",
        "support": [0.10, 0.39],
        "outer_folds": 5,
        "score_training_methods": list(MAIN_METHODS),
        "score_training_excludes_target_comparators": True,
        "target_comparators": list(TARGET_METHODS),
        "features": ["posterior coordinates", "entropy", "peak", "margin", "observed-source count"],
        "thresholds": "refitted on each comparator's outer-training rows for every readout",
        "reproduction_tolerance": 1e-12,
        "main_factorial_max_abs_error": max(abs(reference[key] - reproduced[key]) for key in reference),
        "target_native_max_abs_error": max(abs(native_reference[key] - native_reproduced[key]) for key in native_reference),
        "inputs": {
            str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
            for path in (oof_path, study.DATA, study.LABELS, reference_path, fusion_reference_path)
        },
    }
    (output / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    hashes = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "sha256.json"
    }
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": protocol, "panel_rows": panel_rows}, indent=2))


if __name__ == "__main__":
    main()
