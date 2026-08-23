"""Topology-multiplicity stress test with a decision-layer 2x2 readout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments")]

import pcecf_study as study  # noqa: E402
from action_admission import CONTRACT_CLASSES, graph_from_parent_sets, verify_source_state  # noqa: E402
from action_admission.pcecf import (  # noqa: E402
    SourceEvidence,
    discounted_evidence,
    forward,
    registered_components,
)
from action_admission.verifier import DEFAULT_CONFIG  # noqa: E402
from controlled_study import cluster_id, read_records, scene_fold_map, source_parents  # noqa: E402

MULTIPLICITIES = (1, 2, 4, 8, 16, 32)
ARMS = ("registered", "singleton", "all_merge")
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
SUPPORTS = {"primary_0.10_0.39": (0.10, 0.39), "fixed_0.10_0.35": (0.10, 0.35)}
BOOTSTRAPS = 2000
SEED = 52002
K = len(CONTRACT_CLASSES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def posterior(fused_unit: np.ndarray, concentration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fused = concentration * fused_unit
    denominator = fused.sum(axis=1) + K
    return (
        (fused + 1.0) / denominator[:, None],
        1.0 - K / denominator,
        fused.sum(axis=1),
    )


def fused_unit(unit: np.ndarray, arm: str, multiplicity: int) -> np.ndarray:
    language, geometry, risk = (unit[:, index, :] for index in range(3))
    if arm == "registered":
        return language + np.minimum(geometry, risk)
    if arm == "singleton":
        return language + multiplicity * geometry + risk
    if arm == "all_merge":
        return np.minimum(np.minimum(language, geometry), risk)
    raise ValueError(arm)


def explicit_sources(record: Mapping[str, Any], arm: str, multiplicity: int) -> list[SourceEvidence]:
    base = study.pcecf_sources(record)
    geometry = next(source for source in base if source.source_id == "geometry")
    expanded = list(base)
    for index in range(2, multiplicity + 1):
        expanded.append(
            SourceEvidence(
                source_id=f"geometry_copy_{index}",
                probabilities=geometry.probabilities.copy(),
                quality=geometry.quality,
                conflict=geometry.conflict,
                missing=geometry.missing,
                parents=geometry.parents,
                valid=geometry.valid,
            )
        )
    if arm == "registered":
        return expanded
    parent = "merged:all" if arm == "all_merge" else None
    return [
        replace(source, parents=((parent,) if parent else (f"singleton:{source.source_id}",)))
        for source in expanded
    ]


def exact_formula_check(records: Sequence[Mapping[str, Any]], unit: np.ndarray) -> dict[str, Any]:
    maximum = 0.0
    checks = 0
    for record_index in (0, 1, 2400, 9600, 28800):
        record = records[record_index]
        for concentration in (4.0, 8.0):
            for arm in ARMS:
                for multiplicity in (1, 2, 32):
                    expected, expected_score, _ = posterior(
                        fused_unit(unit[record_index : record_index + 1], arm, multiplicity),
                        concentration,
                    )
                    observed = forward(
                        explicit_sources(record, arm, multiplicity),
                        concentration=concentration,
                    )
                    drift = float(np.max(np.abs(observed.posterior - expected[0])))
                    maximum = max(maximum, drift, abs(observed.selection_score - float(expected_score[0])))
                    checks += 1
    if maximum > 1e-12:
        raise AssertionError(f"closed-form stress construction diverges from PACT: {maximum}")
    return {"checks": checks, "maximum_absolute_residual": maximum}


def make_rows(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    folds: Mapping[str, int],
    unit: np.ndarray,
    v0_pass: Mapping[tuple[str, str], bool],
    arm: str,
    multiplicity: int,
    concentrations: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    unique = sorted(set(concentrations.values()))
    variants: dict[float, list[dict[str, Any]]] = {}
    fused = fused_unit(unit, arm, multiplicity)
    eligible = [study.common_eligibility(record) for record in records]
    for concentration in unique:
        probabilities, scores, _ = posterior(fused, concentration)
        predicted = np.argmax(probabilities, axis=1)
        rows = []
        for index, record in enumerate(records):
            record_id = str(record["record_id"])
            label = CONTRACT_CLASSES[int(predicted[index])]
            rows.append(
                {
                    "record_id": record_id,
                    "scene_id": str(record["metadata"]["scene_id"]),
                    "fold": int(folds[cluster_id(record_id)]),
                    "score": float(scores[index]),
                    "eligible": bool(eligible[index]),
                    "native_eligible": bool(eligible[index]),
                    "verifier_pass": bool(v0_pass[(record_id, label)]),
                    "fold_local_concentration": float(concentration),
                    "predicted_contract": label,
                    "preferred_contract": labels[record_id],
                }
            )
        variants[float(concentration)] = rows
    return {
        outer: variants[float(concentrations[outer])]
        for outer in sorted(concentrations)
    }


def structural_summary(
    unit: np.ndarray,
    folds_by_record: np.ndarray,
    concentrations: Mapping[int, float],
    arm: str,
    multiplicity: int,
    baseline_posterior: Mapping[str, np.ndarray],
    baseline_budget: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    fused = fused_unit(unit, arm, multiplicity)
    probabilities = np.empty((len(unit), K), dtype=np.float64)
    budgets = np.empty(len(unit), dtype=np.float64)
    for fold, concentration in concentrations.items():
        mask = folds_by_record == fold
        probabilities[mask], _, budgets[mask] = posterior(fused[mask], concentration)
    drift = np.abs(probabilities - baseline_posterior[arm]).sum(axis=1)
    denominator = float(baseline_budget[arm].sum())
    ratio = float(budgets.sum() / denominator) if denominator > 0 else 1.0
    return {
        "aggregate_budget_ratio_vs_m1": ratio,
        "posterior_l1_drift_mean_vs_m1": float(drift.mean()),
        "posterior_l1_drift_max_vs_m1": float(drift.max()),
        "registered_components": 2 if arm == "registered" else (multiplicity + 2 if arm == "singleton" else 1),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO / "configs" / "controlled_study.json").read_text(encoding="utf-8"))
    records = read_records(study.DATA)
    labels = {
        str(row["record_id"]): str(row["preferred_contract"])
        for row in read_records(study.LABELS)
    }
    if len(records) != 31_200 or len(labels) != 31_200:
        raise AssertionError("controlled denominator changed")
    folds = scene_fold_map(records, int(config["fold_count"]))
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }

    unit = np.stack(
        [
            np.stack([discounted_evidence(source, 1.0) for source in study.pcecf_sources(record)])
            for record in records
        ]
    )
    # The registered graph has two components throughout: language and the shared scene component.
    component_counts = {len(registered_components(study.pcecf_sources(record))) for record in records}
    if component_counts != {2}:
        raise AssertionError(f"registered component count is not fixed at two: {component_counts}")
    formula_check = exact_formula_check(records, unit)

    v0_config = replace(DEFAULT_CONFIG, minimum_registered_components=0)
    v0_pass: dict[tuple[str, str], bool] = {}
    for record in records:
        record_id = str(record["record_id"])
        graph = graph_from_parent_sets(source_parents(record))
        for label in CONTRACT_CLASSES:
            v0_pass[(record_id, label)] = bool(
                verify_source_state(record, label, graph, config=v0_config).admissible
            )

    fold_index = np.asarray([folds[cluster_id(str(record["record_id"]))] for record in records])
    baseline_posterior: dict[str, np.ndarray] = {}
    baseline_budget: dict[str, np.ndarray] = {}
    for arm in ARMS:
        probabilities = np.empty((len(records), K), dtype=np.float64)
        budgets = np.empty(len(records), dtype=np.float64)
        fused = fused_unit(unit, arm, 1)
        for fold, concentration in concentrations.items():
            mask = fold_index == fold
            probabilities[mask], _, budgets[mask] = posterior(fused[mask], concentration)
        baseline_posterior[arm] = probabilities
        baseline_budget[arm] = budgets

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    registered_ncs: list[float] = []
    singleton_contrasts: list[float] = []
    for multiplicity in MULTIPLICITIES:
        curves: dict[str, list[dict[str, Any]]] = {}
        counts: dict[str, Any] = {}
        structural: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            outer = make_rows(records, labels, folds, unit, v0_pass, arm, multiplicity, concentrations)
            curve, scene_counts, _ = study.evaluate_curve(
                outer,
                TARGETS,
                len(concentrations),
                method=arm,
                table=f"topology_multiplicity_m{multiplicity}",
                verifier=True,
            )
            curves[arm] = curve
            counts[arm] = scene_counts
            structural[arm] = structural_summary(
                unit,
                fold_index,
                concentrations,
                arm,
                multiplicity,
                baseline_posterior,
                baseline_budget,
            )

        scenes = sorted({str(record["metadata"]["scene_id"]) for record in records})
        for support_name, (low, high) in SUPPORTS.items():
            grid = np.linspace(low, high, 36)
            bootstrap, order, draws = study.bootstrap_naurc(
                counts,
                TARGETS,
                scenes,
                grid,
                replicates=BOOTSTRAPS,
                seed=SEED + multiplicity,
            )
            index = {name: order.index(name) for name in order}
            delta = draws[:, index["singleton"]] - draws[:, index["registered"]]
            point_delta = study.interpolate_naurc(curves["singleton"], grid) - study.interpolate_naurc(curves["registered"], grid)
            contrast_rows.append(
                {
                    "multiplicity": multiplicity,
                    "support": support_name,
                    "contrast": "F0_singleton_minus_F1_registered_at_V0",
                    "point": point_delta,
                    "bootstrap_mean": float(delta.mean()),
                    "ci_low": float(np.quantile(delta, 0.025)),
                    "ci_high": float(np.quantile(delta, 0.975)),
                    "fraction_above_zero": float(np.mean(delta > 0.0)),
                }
            )
            if support_name == "primary_0.10_0.39":
                singleton_contrasts.append(point_delta)
            for arm in ARMS:
                point = next(row for row in curves[arm] if math.isclose(float(row["target"]), 0.13))
                ncs = study.interpolate_naurc(curves[arm], grid)
                summary_rows.append(
                    {
                        "multiplicity": multiplicity,
                        "arm": arm,
                        "support": support_name,
                        "ncsaurc": ncs,
                        "raw_partial_aurc": ncs * (high - low),
                        "bootstrap_mean": bootstrap[arm]["mean"],
                        "ci_low": bootstrap[arm]["ci_low"],
                        "ci_high": bootstrap[arm]["ci_high"],
                        "coverage_at_0p13": point["coverage"],
                        "wrong_all_at_0p13": point["wrong_all"],
                        "correct_all_at_0p13": point["correct_all"],
                        **structural[arm],
                    }
                )
                if support_name == "primary_0.10_0.39" and arm == "registered":
                    registered_ncs.append(ncs)

    # m=1 reproduces the frozen 2x2 fusion cells.
    m1 = {
        (row["arm"], row["support"]): row
        for row in summary_rows
        if int(row["multiplicity"]) == 1
    }
    assert math.isclose(m1[("singleton", "primary_0.10_0.39")]["ncsaurc"], 0.41346387685055297, abs_tol=1e-12)
    assert math.isclose(m1[("registered", "primary_0.10_0.39")]["ncsaurc"], 0.3891164502870042, abs_tol=1e-12)
    registered_residual = max(abs(value - registered_ncs[0]) for value in registered_ncs)
    monotone = all(right >= left - 1e-12 for left, right in zip(singleton_contrasts, singleton_contrasts[1:]))

    write_csv(HERE / "stress_summary.csv", summary_rows)
    write_csv(HERE / "matched_contrast.csv", contrast_rows)
    gate = {
        "status": "PASS",
        "records": len(records),
        "scenes": 48,
        "true_registered_components": 2,
        "multiplicities": list(MULTIPLICITIES),
        "arms": list(ARMS),
        "formula_check": formula_check,
        "registered_ncsaurc_max_residual": registered_residual,
        "singleton_minus_registered_contrast_monotone_non_decreasing": monotone,
        "contrast_m1": singleton_contrasts[0],
        "contrast_m32": singleton_contrasts[-1],
        "directional_success_gate": "none; all multiplicities and signs retained",
        "interpretation": "topology-multiplicity stress test, not a scaling law",
    }
    (HERE / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
