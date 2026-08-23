"""Compare PACT's within-component meet with the duplication-invariant join."""

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
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
E2_RUNNER = ROOT / "results" / "topology_multiplicity" / "run.py"
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments")]

import pcecf_study as study  # noqa: E402
from action_admission import CONTRACT_CLASSES  # noqa: E402
from action_admission.pcecf import discounted_evidence, registered_components  # noqa: E402
from controlled_study import cluster_id, read_records, scene_fold_map  # noqa: E402

SPEC = importlib.util.spec_from_file_location("e2_stress", E2_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the frozen E2 runner")
e2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2)

METHODS = ("pact_meet", "pact_join")
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
GRID = np.linspace(0.10, 0.39, 36)
BOOTSTRAPS = 2000
SEED = 59001
K = len(CONTRACT_CLASSES)
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fused_unit(
    unit: np.ndarray, method: str, scene_complete: np.ndarray
) -> np.ndarray:
    language, geometry, risk = (unit[:, index, :] for index in range(3))
    reducer = np.minimum if method == "pact_meet" else np.maximum
    scene_budget = reducer(geometry, risk)
    scene_budget = np.where(scene_complete[:, None], scene_budget, 0.0)
    return language + scene_budget


def make_rows(
    records: Sequence[Mapping[str, Any]],
    preferred: np.ndarray,
    folds_by_record: np.ndarray,
    eligible: np.ndarray,
    unit: np.ndarray,
    scene_complete: np.ndarray,
    method: str,
    concentrations: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    variants: dict[float, list[dict[str, Any]]] = {}
    fused = fused_unit(unit, method, scene_complete)
    for concentration in sorted(set(concentrations.values())):
        probabilities, scores, _ = e2.posterior(fused, concentration)
        predicted = np.argmax(probabilities, axis=1)
        variants[float(concentration)] = [
            {
                "record_id": str(record["record_id"]),
                "scene_id": str(record["metadata"]["scene_id"]),
                "fold": int(folds_by_record[index]),
                "score": float(scores[index]),
                "eligible": bool(eligible[index]),
                "native_eligible": bool(eligible[index]),
                "verifier_pass": True,
                "fold_local_concentration": float(concentration),
                "predicted_contract": CONTRACT_CLASSES[int(predicted[index])],
                "preferred_contract": str(preferred[index]),
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(CONTRACT_CLASSES)
                },
            }
            for index, record in enumerate(records)
        ]
    return {
        outer: variants[float(concentrations[outer])]
        for outer in sorted(concentrations)
    }


def oof_arrays(
    unit: np.ndarray,
    folds_by_record: np.ndarray,
    concentrations: Mapping[int, float],
    method: str,
    scene_complete: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.empty((len(unit), K), dtype=np.float64)
    scores = np.empty(len(unit), dtype=np.float64)
    budgets = np.empty(len(unit), dtype=np.float64)
    fused = fused_unit(unit, method, scene_complete)
    for fold, concentration in concentrations.items():
        mask = folds_by_record == fold
        probabilities[mask], scores[mask], budgets[mask] = e2.posterior(
            fused[mask], concentration
        )
    return probabilities, scores, budgets


def structural_checks(unit: np.ndarray, scene_complete: np.ndarray) -> dict[str, Any]:
    language, geometry, risk = (unit[:, index, :] for index in range(3))
    meet = np.minimum(geometry, risk)
    join = np.maximum(geometry, risk)
    fine = language + geometry + risk
    base = language + geometry
    meet_full = language + np.where(scene_complete[:, None], meet, 0.0)
    join_full = language + np.where(scene_complete[:, None], join, 0.0)

    duplicate_meet = np.minimum(np.minimum(geometry, risk), geometry)
    duplicate_join = np.maximum(np.maximum(geometry, risk), geometry)
    meet_duplicate_residual = float(np.max(np.abs(duplicate_meet - meet)))
    join_duplicate_residual = float(np.max(np.abs(duplicate_join - join)))
    meet_cap_residual = float(np.max(np.maximum(meet - geometry, meet - risk)))
    join_cap_violation = scene_complete[:, None] & (
        (join > geometry + TOL) | (join > risk + TOL)
    )
    join_increase = join_full.sum(axis=1) - base.sum(axis=1)
    meet_change = meet_full.sum(axis=1) - base.sum(axis=1)
    join_coarsening_residual = float(np.max(join_full - fine))

    checks = {
        "meet_exact_duplicate_max_residual": meet_duplicate_residual,
        "join_exact_duplicate_max_residual": join_duplicate_residual,
        "meet_common_evidence_cap_max_residual": meet_cap_residual,
        "complete_scene_component_records": int(scene_complete.sum()),
        "join_common_evidence_cap_violating_complete_coordinates": int(
            join_cap_violation.sum()
        ),
        "join_common_evidence_cap_violating_complete_record_fraction": float(
            np.mean(np.any(join_cap_violation[scene_complete], axis=1))
        ),
        "join_same_component_insertion_increase_complete_record_fraction": float(
            np.mean(join_increase[scene_complete] > TOL)
        ),
        "join_same_component_insertion_mean_budget_increase_complete": float(
            np.mean(join_increase[scene_complete])
        ),
        "join_same_component_insertion_max_budget_increase_complete": float(
            np.max(join_increase[scene_complete])
        ),
        "meet_same_component_insertion_increase_complete_record_fraction": float(
            np.mean(meet_change[scene_complete] > TOL)
        ),
        "meet_same_component_insertion_mean_budget_change_complete": float(
            np.mean(meet_change[scene_complete])
        ),
        "join_partition_coarsening_max_residual": join_coarsening_residual,
    }
    if meet_duplicate_residual > TOL or join_duplicate_residual > TOL:
        raise AssertionError("exact-copy invariance failed")
    if meet_cap_residual > TOL or np.any(meet_change[scene_complete] > TOL):
        raise AssertionError("meet violated the common-evidence cap")
    if not np.any(join_increase[scene_complete] > TOL) or not np.any(join_cap_violation):
        raise AssertionError("join did not expose the intended non-amplification contrast")
    if join_coarsening_residual > TOL:
        raise AssertionError("join violated partition coarsening monotonicity")
    return checks


def main() -> None:
    protocol = json.loads((HERE / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    locked = {
        "controlled_config_sha256": REPO / "configs" / "controlled_study.json",
        "evaluation_labels_file_sha256": study.LABELS,
        "operator_source_sha256": REPO / "src" / "action_admission" / "pcecf.py",
        "source_records_file_sha256": study.DATA,
    }
    for key, path in locked.items():
        if sha256(path).upper() != str(protocol[key]).upper():
            raise AssertionError(f"locked input changed: {key}")

    config = json.loads((REPO / "configs" / "controlled_study.json").read_text(encoding="utf-8"))
    records = read_records(study.DATA)
    label_rows = read_records(study.LABELS)
    labels = {str(row["record_id"]): str(row["preferred_contract"]) for row in label_rows}
    if len(records) != 31_200 or len(labels) != 31_200:
        raise AssertionError("controlled denominator changed")
    folds = scene_fold_map(records, int(config["fold_count"]))
    fold_index = np.asarray([folds[cluster_id(str(row["record_id"]))] for row in records])
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    preferred = np.asarray([labels[str(row["record_id"])] for row in records])
    eligible = np.asarray([study.common_eligibility(row) for row in records], dtype=bool)
    unit = np.stack(
        [
            np.stack(
                [discounted_evidence(source, 1.0) for source in study.pcecf_sources(record)]
            )
            for record in records
        ]
    )
    scene_complete = np.asarray(
        [
            all(
                not source.missing and source.valid
                for source in study.pcecf_sources(record)
                if source.source_id in {"geometry", "risk"}
            )
            for record in records
        ],
        dtype=bool,
    )
    if {len(registered_components(study.pcecf_sources(row))) for row in records} != {2}:
        raise AssertionError("registered topology is not the locked two-component design")
    formula_check = e2.exact_formula_check(records, unit)
    structure = structural_checks(unit, scene_complete)

    scenes = sorted({str(row["metadata"]["scene_id"]) for row in records})
    rows_by_method = {
        method: make_rows(
            records,
            preferred,
            fold_index,
            eligible,
            unit,
            scene_complete,
            method,
            concentrations,
        )
        for method in METHODS
    }
    curves: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {}
    oof: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summaries = []
    for method in METHODS:
        curve, scene_counts, _ = study.evaluate_curve(
            rows_by_method[method],
            TARGETS,
            len(concentrations),
            method=method,
            table="m1_join_ablation",
            verifier=False,
        )
        curves[method] = curve
        counts[method] = scene_counts
        oof[method] = oof_arrays(
            unit, fold_index, concentrations, method, scene_complete
        )

    bootstrap, order, draws = study.bootstrap_naurc(
        counts,
        TARGETS,
        scenes,
        GRID,
        replicates=BOOTSTRAPS,
        seed=SEED,
    )
    point = {method: study.interpolate_naurc(curves[method], GRID) for method in METHODS}
    if not math.isclose(point["pact_meet"], 0.6294374684440601, abs_tol=1e-12):
        raise AssertionError(f"canonical PACT anchor changed: {point['pact_meet']}")

    for method in METHODS:
        probabilities, scores, budgets = oof[method]
        point_013 = next(
            row for row in curves[method] if math.isclose(float(row["target"]), 0.13)
        )
        calibration = study.calibration(study.held_out_rows(rows_by_method[method], 5))
        summaries.append(
            {
                "method": method,
                "ncsaurc_0p10_0p39": point[method],
                "bootstrap_mean": bootstrap[method]["mean"],
                "ci_low": bootstrap[method]["ci_low"],
                "ci_high": bootstrap[method]["ci_high"],
                "coverage_at_0p13": point_013["coverage"],
                "wrong_all_at_0p13": point_013["wrong_all"],
                "correct_all_at_0p13": point_013["correct_all"],
                "aggregate_evidence_budget": float(budgets.sum()),
                "mean_selection_score": float(scores.mean()),
                **calibration,
            }
        )

    method_index = {method: order.index(method) for method in order}
    delta = draws[:, method_index["pact_join"]] - draws[:, method_index["pact_meet"]]
    meet_p, meet_s, meet_b = oof["pact_meet"]
    join_p, join_s, join_b = oof["pact_join"]
    contrast = {
        "contrast": "pact_join_minus_pact_meet",
        "point": point["pact_join"] - point["pact_meet"],
        "bootstrap_mean": float(delta.mean()),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "fraction_above_zero": float(np.mean(delta > 0.0)),
        "aggregate_budget_ratio_join_over_meet": float(join_b.sum() / meet_b.sum()),
        "mean_posterior_l1_join_vs_meet": float(np.abs(join_p - meet_p).sum(axis=1).mean()),
        "max_posterior_l1_join_vs_meet": float(np.abs(join_p - meet_p).sum(axis=1).max()),
        "prediction_flip_rate_join_vs_meet": float(
            np.mean(np.argmax(join_p, axis=1) != np.argmax(meet_p, axis=1))
        ),
        "mean_absolute_score_shift_join_vs_meet": float(np.abs(join_s - meet_s).mean()),
    }

    write_csv(HERE / "operator_summary.csv", summaries)
    write_csv(HERE / "paired_contrast.csv", [contrast])
    (HERE / "structural_checks.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate = {
        "status": "PASS",
        "records": len(records),
        "scenes": len(scenes),
        "formula_check": formula_check,
        "canonical_meet_anchor": point["pact_meet"],
        "join_ncsaurc": point["pact_join"],
        "join_minus_meet": contrast,
        "structural_checks": structure,
        "directional_success_gate": protocol["directional_success_gate"],
        "interpretation": (
            "The join is an exact-copy-invariant ablation that retains partition "
            "monotonicity but violates the declared common-evidence cap and insertion "
            "non-amplification. Predictive direction does not define validity."
        ),
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
