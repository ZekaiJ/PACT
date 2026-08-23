"""Equal-cardinality topology control for the frozen controlled study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import pcecf_study as study  # noqa: E402
from action_admission import CONTRACT_CLASSES  # noqa: E402
from action_admission.pcecf import (  # noqa: E402
    SourceEvidence,
    discounted_evidence,
    forward,
)
from controlled_study import cluster_id, read_records, scene_fold_map  # noqa: E402


PARTITIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "registered_L_GR": (("language",), ("geometry", "risk")),
    "wrong_LG_R": (("language", "geometry"), ("risk",)),
    "wrong_LR_G": (("language", "risk"), ("geometry",)),
    "singleton_L_G_R": (("language",), ("geometry",), ("risk",)),
}
TARGETS = tuple(round(index / 100.0, 2) for index in range(1, 61))
GRID = np.linspace(0.10, 0.39, 36)
BOOTSTRAPS = 2000
SEED = 56201
K = len(CONTRACT_CLASSES)
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fused_unit(
    unit: np.ndarray,
    partition: Sequence[Sequence[str]],
) -> np.ndarray:
    source_index = {name: index for index, name in enumerate(study.SOURCE_NAMES)}
    return np.sum(
        [
            np.min(unit[:, [source_index[name] for name in component], :], axis=1)
            for component in partition
        ],
        axis=0,
    )


def posterior(
    fused: np.ndarray,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence = concentration * fused
    denominator = evidence.sum(axis=1) + K
    probabilities = (evidence + 1.0) / denominator[:, None]
    score = 1.0 - K / denominator
    return probabilities, score, evidence.sum(axis=1)


def explicit_sources(
    record: Mapping[str, Any],
    partition: Sequence[Sequence[str]],
) -> list[SourceEvidence]:
    component_by_source = {
        name: component_index
        for component_index, component in enumerate(partition)
        for name in component
    }
    return [
        SourceEvidence(
            source_id=source.source_id,
            probabilities=source.probabilities,
            quality=source.quality,
            conflict=source.conflict,
            missing=source.missing,
            parents=(f"m1-component:{component_by_source[source.source_id]}",),
            valid=source.valid,
            evidence=source.evidence,
        )
        for source in study.pcecf_sources(record)
    ]


def formula_check(records: Sequence[Mapping[str, Any]], unit: np.ndarray) -> dict[str, Any]:
    maximum = 0.0
    checks = 0
    for record_index in (0, 1, 2400, 9600, 19200, 28800):
        for concentration in (4.0, 8.0):
            for partition in PARTITIONS.values():
                expected, expected_score, _ = posterior(
                    fused_unit(unit[record_index : record_index + 1], partition),
                    concentration,
                )
                observed = forward(
                    explicit_sources(records[record_index], partition),
                    concentration=concentration,
                    expected_source_ids=study.SOURCE_NAMES,
                )
                maximum = max(
                    maximum,
                    float(np.max(np.abs(observed.posterior - expected[0]))),
                    abs(observed.selection_score - float(expected_score[0])),
                )
                checks += 1
    if maximum > TOL:
        raise AssertionError(f"closed-form topology construction diverged: {maximum}")
    return {"checks": checks, "maximum_absolute_residual": maximum}


def make_rows(
    records: Sequence[Mapping[str, Any]],
    preferred: np.ndarray,
    folds_by_record: np.ndarray,
    eligible: np.ndarray,
    unit: np.ndarray,
    partition: Sequence[Sequence[str]],
    concentrations: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    fused = fused_unit(unit, partition)
    variants: dict[float, list[dict[str, Any]]] = {}
    for concentration in sorted(set(concentrations.values())):
        probabilities, scores, _ = posterior(fused, concentration)
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
        outer_fold: variants[float(concentrations[outer_fold])]
        for outer_fold in sorted(concentrations)
    }


def oof_arrays(
    unit: np.ndarray,
    folds_by_record: np.ndarray,
    concentrations: Mapping[int, float],
    partition: Sequence[Sequence[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.empty((len(unit), K), dtype=np.float64)
    scores = np.empty(len(unit), dtype=np.float64)
    budgets = np.empty(len(unit), dtype=np.float64)
    fused = fused_unit(unit, partition)
    for fold, concentration in concentrations.items():
        mask = folds_by_record == fold
        probabilities[mask], scores[mask], budgets[mask] = posterior(
            fused[mask], concentration
        )
    return probabilities, scores, budgets


def random_reference_by_scene(
    rows: Sequence[Mapping[str, Any]], scenes: Sequence[str]
) -> np.ndarray:
    scene_index = {scene: index for index, scene in enumerate(scenes)}
    counts = np.zeros((len(scenes), 2), dtype=np.float64)
    for row in rows:
        if not bool(row["eligible"]):
            continue
        index = scene_index[str(row["scene_id"])]
        counts[index, 0] += 1.0
        counts[index, 1] += float(
            row["predicted_contract"] != row["preferred_contract"]
        )
    if np.any(counts[:, 0] == 0):
        raise AssertionError("a scene has no common-eligible candidates")
    return counts


def main() -> None:
    protocol = json.loads((HERE / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    locked = {
        "controlled_config_sha256": ROOT / "configs" / "controlled_study.json",
        "evaluation_labels_file_sha256": study.LABELS,
        "operator_source_sha256": ROOT / "src" / "action_admission" / "pcecf.py",
        "source_records_file_sha256": study.DATA,
    }
    for key, path in locked.items():
        if sha256(path) != str(protocol[key]).upper():
            raise AssertionError(f"locked input changed: {key}")
    config = json.loads(
        (ROOT / "configs" / "controlled_study.json").read_text(encoding="utf-8")
    )
    records = read_records(study.DATA)
    label_rows = read_records(study.LABELS)
    labels = {str(row["record_id"]): str(row["preferred_contract"]) for row in label_rows}
    if len(records) != 31_200 or len(labels) != 31_200:
        raise AssertionError("controlled denominator changed")
    folds = scene_fold_map(records, int(config["fold_count"]))
    folds_by_record = np.asarray(
        [folds[cluster_id(str(record["record_id"]))] for record in records]
    )
    concentrations = {
        int(fold): float(value)
        for fold, value in config["dirichlet_concentration_by_fold"].items()
    }
    preferred = np.asarray([labels[str(record["record_id"])] for record in records])
    eligible = np.asarray([study.common_eligibility(record) for record in records])
    unit = np.stack(
        [
            np.stack(
                [discounted_evidence(source, 1.0) for source in study.pcecf_sources(record)]
            )
            for record in records
        ]
    )
    if tuple(study.SOURCE_NAMES) != ("language", "geometry", "risk"):
        raise AssertionError("source order changed")
    if any(
        sorted(name for component in partition for name in component)
        != sorted(study.SOURCE_NAMES)
        for partition in PARTITIONS.values()
    ):
        raise AssertionError("a partition does not cover the fixed source catalog")

    formula = formula_check(records, unit)
    scenes = sorted({str(record["metadata"]["scene_id"]) for record in records})
    rows_by_arm = {
        arm: make_rows(
            records,
            preferred,
            folds_by_record,
            eligible,
            unit,
            partition,
            concentrations,
        )
        for arm, partition in PARTITIONS.items()
    }
    curves: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, Any] = {}
    oof_rows: dict[str, list[dict[str, Any]]] = {}
    oof: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm, partition in PARTITIONS.items():
        curve, scene_counts, _ = study.evaluate_curve(
            rows_by_arm[arm],
            TARGETS,
            len(concentrations),
            method=arm,
            table="m1_equal_cardinality_topology",
            verifier=False,
        )
        curves[arm] = curve
        counts[arm] = scene_counts
        oof_rows[arm] = study.held_out_rows(rows_by_arm[arm], len(concentrations))
        oof[arm] = oof_arrays(unit, folds_by_record, concentrations, partition)

    bootstrap, method_order, draws = study.bootstrap_naurc(
        counts,
        TARGETS,
        scenes,
        GRID,
        replicates=BOOTSTRAPS,
        seed=SEED,
    )
    points = {arm: study.interpolate_naurc(curves[arm], GRID) for arm in PARTITIONS}
    if not math.isclose(points["registered_L_GR"], 0.6294374684440601, abs_tol=1e-12):
        raise AssertionError("registered PACT anchor changed")

    random_scene = {
        arm: random_reference_by_scene(oof_rows[arm], scenes) for arm in PARTITIONS
    }
    rng = np.random.default_rng(SEED)
    random_draws = np.empty((BOOTSTRAPS, len(PARTITIONS)), dtype=np.float64)
    for replicate in range(BOOTSTRAPS):
        weights = rng.multinomial(len(scenes), np.full(len(scenes), 1.0 / len(scenes)))
        for arm_index, arm in enumerate(method_order):
            totals = random_scene[arm].T @ weights
            random_draws[replicate, arm_index] = totals[1] / totals[0]

    summaries: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for arm, partition in PARTITIONS.items():
        probabilities, scores, budgets = oof[arm]
        calibration = study.calibration(oof_rows[arm])
        arm_index = method_order.index(arm)
        random_point = float(random_scene[arm][:, 1].sum() / random_scene[arm][:, 0].sum())
        excess = draws[:, arm_index] - random_draws[:, arm_index]
        target = next(
            row for row in curves[arm] if math.isclose(float(row["target"]), 0.13)
        )
        summaries.append(
            {
                "arm": arm,
                "partition": " | ".join("+".join(component) for component in partition),
                "component_count": len(partition),
                "ncsaurc_0p10_0p39": points[arm],
                "bootstrap_mean": bootstrap[arm]["mean"],
                "ci_low": bootstrap[arm]["ci_low"],
                "ci_high": bootstrap[arm]["ci_high"],
                "eligible_random_reference": random_point,
                "ncsaurc_minus_random": points[arm] - random_point,
                "coverage_at_0p13": target["coverage"],
                "wrong_all_at_0p13": target["wrong_all"],
                "aggregate_evidence_budget": float(budgets.sum()),
                "mean_selection_score": float(scores.mean()),
                **calibration,
            }
        )
        random_rows.append(
            {
                "arm": arm,
                "eligible_random_reference": random_point,
                "bootstrap_mean": float(np.mean(random_draws[:, arm_index])),
                "ci_low": float(np.quantile(random_draws[:, arm_index], 0.025)),
                "ci_high": float(np.quantile(random_draws[:, arm_index], 0.975)),
                "ncsaurc_minus_random_point": points[arm] - random_point,
                "excess_bootstrap_mean": float(np.mean(excess)),
                "excess_ci_low": float(np.quantile(excess, 0.025)),
                "excess_ci_high": float(np.quantile(excess, 0.975)),
                "fraction_excess_above_zero": float(np.mean(excess > 0.0)),
            }
        )

    contrasts: list[dict[str, Any]] = []
    registered_index = method_order.index("registered_L_GR")
    registered_probabilities, registered_scores, registered_budgets = oof[
        "registered_L_GR"
    ]
    for arm in ("wrong_LG_R", "wrong_LR_G", "singleton_L_G_R"):
        arm_index = method_order.index(arm)
        delta = draws[:, arm_index] - draws[:, registered_index]
        probabilities, scores, budgets = oof[arm]
        contrasts.append(
            {
                "contrast": f"{arm}_minus_registered_L_GR",
                "point": points[arm] - points["registered_L_GR"],
                "bootstrap_mean": float(np.mean(delta)),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "fraction_above_zero": float(np.mean(delta > 0.0)),
                "aggregate_budget_ratio": float(budgets.sum() / registered_budgets.sum()),
                "mean_posterior_l1": float(
                    np.abs(probabilities - registered_probabilities).sum(axis=1).mean()
                ),
                "prediction_flip_rate": float(
                    np.mean(
                        np.argmax(probabilities, axis=1)
                        != np.argmax(registered_probabilities, axis=1)
                    )
                ),
                "mean_absolute_score_shift": float(
                    np.abs(scores - registered_scores).mean()
                ),
            }
        )

    write_csv(HERE / "topology_summary.csv", summaries)
    write_csv(HERE / "paired_contrasts.csv", contrasts)
    write_csv(HERE / "random_references.csv", random_rows)
    gate = {
        "status": "PASS",
        "records": len(records),
        "scenes": len(scenes),
        "support": [float(GRID[0]), float(GRID[-1])],
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "formula_check": formula,
        "registered_anchor": points["registered_L_GR"],
        "equal_cardinality_contrasts": contrasts[:2],
        "directional_success_gate": "none; all signs and effect sizes are retained",
        "interpretation_boundary": (
            "The equal-cardinality controls test the registered pairing under fixed "
            "opinions, component count, concentration schedule, eligibility, native score definition, and "
            "verifier-off protocol. They do not authenticate ancestry or identify a "
            "general causal effect of provenance registration."
        ),
    }
    (HERE / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        path.name: sha256(path).lower()
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
