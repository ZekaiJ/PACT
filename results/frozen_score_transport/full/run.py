"""Evaluate the m=1 shared score after same-source multiplicity shifts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT
E2_RUNNER = ROOT / "results" / "topology_multiplicity" / "run.py"
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments"), str(REPO / "analysis")]

import pcecf_study as study  # noqa: E402
import score_verifier_factorial as score  # noqa: E402
import strong_paper_controls as controls  # noqa: E402
from action_admission import CONTRACT_CLASSES, restrict_dirichlet_input  # noqa: E402
from action_admission.dirichlet import DirichletInput, predict as nested_predict  # noqa: E402
from action_admission.pcecf import discounted_evidence  # noqa: E402
from controlled_study import cluster_id, read_records, scene_fold_map  # noqa: E402

SPEC = importlib.util.spec_from_file_location("e2_stress", E2_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the frozen E2 runner")
e2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2)

MULTIPLICITIES = tuple(e2.MULTIPLICITIES)
METHODS = ("pact_registered", "pact_singleton", "pact_all_merge", "nested_unaware")
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
GRID = np.linspace(0.10, 0.39, 36)
BOOTSTRAPS = 2000
SEED = 58002
K = len(CONTRACT_CLASSES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_matrix(probabilities: np.ndarray, observed_count: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    return np.column_stack(
        (probabilities, entropy, ordered[:, 0], ordered[:, 0] - ordered[:, 1], observed_count)
    )


def fit_frozen_models(
    rows_by_method: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    observed: Mapping[str, int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    models = {}
    summaries = []
    for outer in range(5):
        train = [
            row
            for method in score.MAIN_METHODS
            for row in rows_by_method[method][outer]
            if int(row["fold"]) != outer
        ]
        x_train = np.asarray(
            [controls.posterior_features(row, observed[str(row["record_id"])]) for row in train]
        )
        y_train = np.asarray(
            [row["predicted_contract"] == row["preferred_contract"] for row in train],
            dtype=int,
        )
        model = controls.fit_logistic(x_train, y_train)
        models[outer] = model
        weights, mean, scale = model
        summaries.append(
            {
                "outer_fold": outer,
                "training_rows": len(train),
                "positive_fraction": float(y_train.mean()),
                "observed_count_mean": float(mean[-1]),
                "observed_count_scale": float(scale[-1]),
                "observed_count_standardized_coefficient": float(weights[-1]),
                "observed_count_raw_logit_coefficient": float(weights[-1] / scale[-1]),
            }
        )
    return models, summaries


def nested_probabilities(
    inputs: Sequence[DirichletInput],
    multiplicity: int,
    concentration: float,
) -> np.ndarray:
    source_names = ["language", "geometry", "risk"] + [
        f"geometry_copy_{index}" for index in range(2, multiplicity + 1)
    ]
    result = np.empty((len(inputs), K), dtype=np.float64)
    for index, base in enumerate(inputs):
        geometry = base.sources["geometry"]
        expanded = dict(base.sources)
        for copy_index in range(2, multiplicity + 1):
            expanded[f"geometry_copy_{copy_index}"] = geometry
        prediction = nested_predict(
            DirichletInput(sources=expanded),
            concentration=concentration,
            sources=source_names,
        )
        result[index] = [prediction.probabilities[label] for label in CONTRACT_CLASSES]
    return result


def evaluation_rows(
    records: Sequence[Mapping[str, Any]],
    preferred: np.ndarray,
    folds_by_record: np.ndarray,
    eligible: np.ndarray,
    probabilities: np.ndarray,
    observed_count: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    concentration: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    scores = controls.predict_logistic(feature_matrix(probabilities, observed_count), model)
    predicted = np.argmax(probabilities, axis=1)
    rows = [
        {
            "record_id": str(record["record_id"]),
            "scene_id": str(record["metadata"]["scene_id"]),
            "fold": int(folds_by_record[index]),
            "score": float(scores[index]),
            "eligible": bool(eligible[index]),
            "native_eligible": bool(eligible[index]),
            "verifier_pass": True,
            "fold_local_concentration": concentration,
            "predicted_contract": CONTRACT_CLASSES[int(predicted[index])],
            "preferred_contract": str(preferred[index]),
        }
        for index, record in enumerate(records)
    ]
    return rows, scores


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO / "configs" / "controlled_study.json").read_text(encoding="utf-8"))
    records = read_records(study.DATA)
    label_rows = read_records(study.LABELS)
    labels_by_id = {str(row["record_id"]): row for row in label_rows}
    preferred = np.asarray([str(labels_by_id[str(record["record_id"])]["preferred_contract"]) for record in records])
    if len(records) != 31_200 or len(labels_by_id) != 31_200:
        raise AssertionError("controlled denominator changed")

    folds = scene_fold_map(records, int(config["fold_count"]))
    fold_index = np.asarray([folds[cluster_id(str(record["record_id"]))] for record in records])
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    unique_concentrations = tuple(sorted(set(concentrations.values())))
    eligible = np.asarray([study.common_eligibility(record) for record in records], dtype=bool)
    base_observed = np.asarray([study.observed_source_count(record) for record in records], dtype=float)
    geometry_observed = np.asarray(
        [
            bool(
                record["sources"].get("geometry", {})
                and not record["sources"]["geometry"].get("missing", False)
                and study.structurally_valid(record["sources"]["geometry"])
            )
            for record in records
        ],
        dtype=float,
    )

    base_rows = study.make_rows(records, labels_by_id, folds, concentrations)
    observed_by_id = {
        str(record["record_id"]): int(base_observed[index])
        for index, record in enumerate(records)
    }
    models, model_rows = fit_frozen_models(base_rows, observed_by_id)

    unit = np.stack(
        [
            np.stack([discounted_evidence(source, 1.0) for source in study.pcecf_sources(record)])
            for record in records
        ]
    )
    nested_inputs = [restrict_dirichlet_input(record) for record in records]
    scenes = sorted({str(record["metadata"]["scene_id"]) for record in records})

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    baseline_oof_scores: dict[str, np.ndarray] = {}
    baseline_oof_probabilities: dict[str, np.ndarray] = {}
    gap_by_m: dict[int, float] = {}
    formula_residual = 0.0

    for multiplicity in MULTIPLICITIES:
        expanded_count = base_observed + (multiplicity - 1) * geometry_observed
        probabilities: dict[str, dict[float, np.ndarray]] = {method: {} for method in METHODS}
        for concentration in unique_concentrations:
            for arm in ("registered", "singleton", "all_merge"):
                probabilities[f"pact_{arm}"][concentration] = e2.posterior(
                    e2.fused_unit(unit, arm, multiplicity), concentration
                )[0]
            probabilities["nested_unaware"][concentration] = nested_probabilities(
                nested_inputs, multiplicity, concentration
            )

        counts: dict[str, Any] = {}
        curves: dict[str, list[dict[str, Any]]] = {}
        oof_scores: dict[str, np.ndarray] = {}
        oof_probabilities: dict[str, np.ndarray] = {}
        for method in METHODS:
            rows_by_outer = {}
            method_oof_scores = np.empty(len(records), dtype=np.float64)
            method_oof_probabilities = np.empty((len(records), K), dtype=np.float64)
            for outer in range(5):
                concentration = float(concentrations[outer])
                rows, scores = evaluation_rows(
                    records,
                    preferred,
                    fold_index,
                    eligible,
                    probabilities[method][concentration],
                    expanded_count,
                    models[outer],
                    concentration,
                )
                rows_by_outer[outer] = rows
                held_out = fold_index == outer
                method_oof_scores[held_out] = scores[held_out]
                method_oof_probabilities[held_out] = probabilities[method][concentration][held_out]
            curve, scene_counts, _ = study.evaluate_curve(
                rows_by_outer,
                TARGETS,
                5,
                method=method,
                table=f"frozen_shared_score_m{multiplicity}",
                verifier=False,
            )
            curves[method] = curve
            counts[method] = scene_counts
            oof_scores[method] = method_oof_scores
            oof_probabilities[method] = method_oof_probabilities

        if multiplicity == 1:
            baseline_oof_scores = {method: values.copy() for method, values in oof_scores.items()}
            baseline_oof_probabilities = {
                method: values.copy() for method, values in oof_probabilities.items()
            }
            pact_anchor = study.interpolate_naurc(curves["pact_registered"], GRID)
            nested_anchor = study.interpolate_naurc(curves["nested_unaware"], GRID)
            if not math.isclose(pact_anchor, 0.6488988790621325, abs_tol=1e-12):
                raise AssertionError(f"PACT shared-score anchor changed: {pact_anchor}")
            if not math.isclose(nested_anchor, 0.3795544515360131, abs_tol=1e-12):
                raise AssertionError(f"nested shared-score anchor changed: {nested_anchor}")

            for outer in range(5):
                held_out = fold_index == outer
                for method, base_name in (
                    ("pact_registered", "pcecf"),
                    ("nested_unaware", "nested_evidential_composition"),
                ):
                    base = base_rows[base_name][outer]
                    base_probabilities = np.asarray(
                        [
                            [float(row["probabilities"][label]) for label in CONTRACT_CLASSES]
                            for row in base
                        ]
                    )
                    formula_residual = max(
                        formula_residual,
                        float(
                            np.max(
                                np.abs(
                                    probabilities[method][float(concentrations[outer])][held_out]
                                    - base_probabilities[held_out]
                                )
                            )
                        ),
                    )
            if formula_residual > 1e-12:
                raise AssertionError(f"m=1 posterior reproduction failed: {formula_residual}")

        bootstrap, order, draws = study.bootstrap_naurc(
            counts,
            TARGETS,
            scenes,
            GRID,
            replicates=BOOTSTRAPS,
            seed=SEED + multiplicity,
        )
        index = {name: order.index(name) for name in order}
        point_ncs = {method: study.interpolate_naurc(curves[method], GRID) for method in METHODS}
        gap_by_m[multiplicity] = point_ncs["pact_registered"] - point_ncs["nested_unaware"]

        for method in METHODS:
            point = next(row for row in curves[method] if math.isclose(float(row["target"]), 0.13))
            score_drift = np.abs(oof_scores[method] - baseline_oof_scores.get(method, oof_scores[method]))
            posterior_drift = np.abs(
                oof_probabilities[method] - baseline_oof_probabilities.get(method, oof_probabilities[method])
            ).sum(axis=1)
            summary_rows.append(
                {
                    "multiplicity": multiplicity,
                    "method": method,
                    "ncsaurc_0p10_0p39": point_ncs[method],
                    "bootstrap_mean": bootstrap[method]["mean"],
                    "ci_low": bootstrap[method]["ci_low"],
                    "ci_high": bootstrap[method]["ci_high"],
                    "coverage_at_0p13": point["coverage"],
                    "wrong_all_at_0p13": point["wrong_all"],
                    "correct_all_at_0p13": point["correct_all"],
                    "mean_absolute_score_drift_vs_m1": float(score_drift.mean()),
                    "maximum_absolute_score_drift_vs_m1": float(score_drift.max()),
                    "mean_posterior_l1_drift_vs_m1": float(posterior_drift.mean()),
                    "maximum_posterior_l1_drift_vs_m1": float(posterior_drift.max()),
                    "mean_expanded_observed_opinion_count": float(expanded_count.mean()),
                }
            )

        for name, left, right in (
            ("pact_registered_minus_nested_unaware", "pact_registered", "nested_unaware"),
            ("pact_singleton_minus_pact_registered", "pact_singleton", "pact_registered"),
        ):
            delta = draws[:, index[left]] - draws[:, index[right]]
            contrast_rows.append(
                {
                    "multiplicity": multiplicity,
                    "contrast": name,
                    "point": point_ncs[left] - point_ncs[right],
                    "bootstrap_mean": float(delta.mean()),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "fraction_above_zero": float(np.mean(delta > 0.0)),
                }
            )

    write_csv(HERE / "fold_models.csv", model_rows)
    write_csv(HERE / "stress_summary.csv", summary_rows)
    write_csv(HERE / "matched_contrast.csv", contrast_rows)
    gate = {
        "status": "PASS",
        "records": len(records),
        "scenes": len(scenes),
        "multiplicities": list(MULTIPLICITIES),
        "score_fit": "m=1 outer-training only; coefficients frozen for every larger multiplicity",
        "threshold_fit": "outer-training score quantiles at each multiplicity; no labels",
        "verifier": "disabled",
        "m1_pact_shared_score_ncsaurc": 0.6488988790621325,
        "m1_nested_shared_score_ncsaurc": 0.3795544515360131,
        "pact_minus_nested_gap_by_multiplicity": {
            str(key): value for key, value in gap_by_m.items()
        },
        "gap_sign_change": any(value <= 0.0 for value in gap_by_m.values()),
        "gap_change_m1_to_m32": gap_by_m[32] - gap_by_m[1],
        "m1_posterior_reproduction_max_residual": formula_residual,
        "directional_success_gate": "none; all multiplicities and signs retained",
    }
    (HERE / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        path.name: sha256(path)
        for path in sorted(HERE.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    (HERE / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in manifest.items()),
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
