"""Reproduce the manuscript's joint scene--condition PACT comparison.

The script reads the frozen controlled records and evaluated operator code
from this release. It never edits the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
OUT = ROOT / "results" / "joint_scene_configuration_holdout"
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))

import pcecf_study as study  # noqa: E402
from controlled_study import cluster_id, read_records, scene_fold_map  # noqa: E402


CONCENTRATIONS = (4.0, 8.0)
TRANSFERRED_CONCENTRATION = {0: 8.0, 1: 4.0, 2: 8.0, 3: 4.0, 4: 4.0}
CURVE_TARGETS = np.arange(1, 61, dtype=np.float64) / 100.0
SUPPORT = np.linspace(0.10, 0.35, 36)
BOOTSTRAPS = 2_000
BOOTSTRAP_SEED = 1886
FIXED_METHODS = (
    "quality_weighted_fusion",
    "product_evidence_fusion",
    "cautious_evidence_fusion",
    "lineage_unaware_pooling",
    "registered_lineage_pooling",
    "hierarchical_cautious_cumulative",
)
REPORTED_METHODS = (
    "pcecf",
    "nested_evidential_composition",
    "hierarchical_cautious_cumulative",
    "product_evidence_fusion",
    "registered_lineage_pooling",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def threshold(score: np.ndarray, available: np.ndarray, mask: np.ndarray, target: float) -> float:
    values = np.sort(score[available & mask])[::-1]
    count = int(round(target * int(mask.sum())))
    if count <= 0 or not len(values):
        return float("inf")
    return float(values[min(count, len(values)) - 1])


def counts(
    score: np.ndarray,
    available: np.ndarray,
    wrong: np.ndarray,
    mask: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    admitted = mask & available & (score >= cutoff)
    return np.asarray(
        [mask.sum(), admitted.sum(), (admitted & wrong).sum(), (admitted & ~wrong).sum()],
        dtype=np.int64,
    )


def ncs_aurc(counts_by_target: np.ndarray) -> float:
    grouped: dict[float, list[float]] = defaultdict(list)
    for n, admitted, wrong, _ in counts_by_target:
        coverage = float(admitted / n)
        risk = float(wrong / admitted) if admitted else 0.0
        grouped[coverage].append(risk)
    x = np.asarray(sorted(grouped), dtype=np.float64)
    y = np.asarray([np.mean(grouped[value]) for value in x], dtype=np.float64)
    if x[0] > SUPPORT[0] + 1e-12 or x[-1] < SUPPORT[-1] - 1e-12:
        raise ValueError(f"curve support {x[0]:.6f}--{x[-1]:.6f} misses fixed support")
    risk = np.interp(SUPPORT, x, y)
    return float(np.trapezoid(risk, SUPPORT) / (SUPPORT[-1] - SUPPORT[0]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = read_records(REPO / "data" / "controlled" / "source_records.jsonl.gz")
    labels = read_records(REPO / "data" / "controlled" / "evaluation_labels.jsonl.gz")
    labels_by_id = {str(row["record_id"]): str(row["preferred_contract"]) for row in labels}
    folds_by_cluster = scene_fold_map(records, 5)
    conditions = sorted({str(row["record_id"]).split("__")[1] for row in records})
    scenes = sorted({str(row["metadata"]["scene_id"]) for row in records})
    condition_index = {value: index for index, value in enumerate(conditions)}
    scene_index = {value: index for index, value in enumerate(scenes)}
    n = len(records)
    if n != 31_200 or len(conditions) != 13 or len(scenes) != 48:
        raise AssertionError((n, len(conditions), len(scenes)))

    fold = np.empty(n, dtype=np.int8)
    condition = np.empty(n, dtype=np.int8)
    scene = np.empty(n, dtype=np.int8)
    record_ids: list[str] = []
    variants = list(FIXED_METHODS)
    variants += [f"nested_evidential_composition_c{int(c)}" for c in CONCENTRATIONS]
    variants += [f"pcecf_c{int(c)}" for c in CONCENTRATIONS]
    score = {name: np.empty(n, dtype=np.float64) for name in variants}
    available = {name: np.empty(n, dtype=bool) for name in variants}
    wrong = {name: np.empty(n, dtype=bool) for name in variants}

    for index, record in enumerate(records):
        record_id = str(record["record_id"])
        record_ids.append(record_id)
        fold[index] = folds_by_cluster[cluster_id(record_id)]
        condition[index] = condition_index[record_id.split("__")[1]]
        scene[index] = scene_index[str(record["metadata"]["scene_id"])]
        graph = study.graph_from_parent_sets(study.source_parents(record))
        common = study.common_eligibility(record)
        truth = labels_by_id[record_id]
        specs = [(name, name, 4.0) for name in FIXED_METHODS]
        specs += [
            (f"nested_evidential_composition_c{int(c)}", "nested_evidential_composition", c)
            for c in CONCENTRATIONS
        ]
        specs += [(f"pcecf_c{int(c)}", "pcecf", c) for c in CONCENTRATIONS]
        for name, method, concentration in specs:
            prediction, _, value, _ = study.predict_record(record, method, concentration)
            verification = study.verify_source_state(record, prediction, graph)
            score[name][index] = value
            available[name][index] = common and bool(verification.admissible)
            wrong[name][index] = prediction != truth
        if (index + 1) % 4_000 == 0:
            print(f"predictions {index + 1}/{n}", flush=True)

    selection_rows: list[dict[str, object]] = []
    for held, held_name in enumerate(conditions):
        for outer in range(5):
            selection_rows.append(
                {
                    "held_condition": held_name,
                    "outer_fold": outer,
                    "concentration": TRANSFERRED_CONCENTRATION[outer],
                    "selection_scope": "transferred main-study schedule; no joint-holdout reselection",
                }
            )

    aggregate = {
        method: np.zeros((len(CURVE_TARGETS), 4), dtype=np.int64)
        for method in REPORTED_METHODS
    }
    by_condition_scene = {
        method: np.zeros((len(conditions), len(CURVE_TARGETS), len(scenes), 4), dtype=np.int64)
        for method in REPORTED_METHODS
    }
    for held in range(len(conditions)):
        for outer in range(5):
            pact_concentration = TRANSFERRED_CONCENTRATION[outer]
            nested_concentration = TRANSFERRED_CONCENTRATION[outer]
            source = {
                "pcecf": f"pcecf_c{int(pact_concentration)}",
                "nested_evidential_composition": f"nested_evidential_composition_c{int(nested_concentration)}",
                "hierarchical_cautious_cumulative": "hierarchical_cautious_cumulative",
                "product_evidence_fusion": "product_evidence_fusion",
                "registered_lineage_pooling": "registered_lineage_pooling",
            }
            train = (fold != outer) & (condition != held)
            test = (fold == outer) & (condition == held)
            for method, name in source.items():
                for target_index, target in enumerate(CURVE_TARGETS):
                    cutoff = threshold(score[name], available[name], train, float(target))
                    value = counts(score[name], available[name], wrong[name], test, cutoff)
                    aggregate[method][target_index] += value
                    for scene_id in np.unique(scene[test]):
                        mask = test & (scene == scene_id)
                        by_condition_scene[method][held, target_index, scene_id] += counts(
                            score[name], available[name], wrong[name], mask, cutoff
                        )


    summary_rows: list[dict[str, object]] = []
    anchor_index = int(np.where(np.isclose(CURVE_TARGETS, 0.13))[0][0])
    for method in REPORTED_METHODS:
        value = aggregate[method]
        n0, admitted, bad, correct = value[anchor_index]
        summary_rows.append(
            {
                "method": method,
                "ncsAURC": ncs_aurc(value),
                "coverage_at_target_0.13": admitted / n0,
                "wrong_all_at_target_0.13": bad / n0,
                "correct_all_at_target_0.13": correct / n0,
                "admitted_at_target_0.13": int(admitted),
                "wrong_at_target_0.13": int(bad),
                "correct_at_target_0.13": int(correct),
            }
        )

    condition_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[rng.randrange(len(scenes)) for _ in scenes] for _ in range(BOOTSTRAPS)]
    for held, name in enumerate(conditions):
        unsupported = None
        for method in ("pcecf", "nested_evidential_composition"):
            for scene_id in range(len(scenes)):
                try:
                    ncs_aurc(by_condition_scene[method][held, :, scene_id, :])
                except ValueError as error:
                    unsupported = f"{method}, scene {scenes[scene_id]}: {error}"
                    break
            if unsupported:
                break
        if unsupported:
            condition_rows.append({"held_condition": name, "eligible": False, "reason": unsupported})
            continue
        pact_counts = by_condition_scene["pcecf"][held].sum(axis=1)
        nested_counts = by_condition_scene["nested_evidential_composition"][held].sum(axis=1)
        point = ncs_aurc(pact_counts) - ncs_aurc(nested_counts)
        differences = []
        for draw in draws:
            pact_draw = np.take(by_condition_scene["pcecf"][held], draw, axis=1).sum(axis=1)
            nested_draw = np.take(
                by_condition_scene["nested_evidential_composition"][held], draw, axis=1
            ).sum(axis=1)
            differences.append(ncs_aurc(pact_draw) - ncs_aurc(nested_draw))
        condition_rows.append(
            {
                "held_condition": name,
                "eligible": True,
                "pact_minus_nested": point,
                "ci95_low": float(np.quantile(differences, 0.025)),
                "ci95_high": float(np.quantile(differences, 0.975)),
                "valid_bootstrap_draws": len(differences),
            }
        )
        bootstrap_rows.extend(
            {"held_condition": name, "replicate": index, "pact_minus_nested": value}
            for index, value in enumerate(differences)
        )

    aggregate_differences = []
    scene_totals = {
        method: by_condition_scene[method].sum(axis=0) for method in REPORTED_METHODS
    }
    for draw in draws:
        left = np.take(scene_totals["pcecf"], draw, axis=1).sum(axis=1)
        right = np.take(scene_totals["nested_evidential_composition"], draw, axis=1).sum(axis=1)
        aggregate_differences.append(ncs_aurc(left) - ncs_aurc(right))

    write_csv(OUT / "outer_selection.csv", selection_rows)
    write_csv(OUT / "joint_summary.csv", summary_rows)
    write_csv(OUT / "conditionwise_summary.csv", condition_rows)
    if bootstrap_rows:
        write_csv(OUT / "conditionwise_bootstrap.csv", bootstrap_rows)
    payload = {
        "version": "canonical-joint-holdout-v1",
        "status": "COMPLETE",
        "records": n,
        "scenes": len(scenes),
        "conditions": len(conditions),
        "transferred_concentration_by_outer_fold": {
            str(key): value for key, value in TRANSFERRED_CONCENTRATION.items()
        },
        "concentration_selection": "fixed main-study schedule; not reselected in the joint holdout",
        "curve_targets": [float(CURVE_TARGETS[0]), float(CURVE_TARGETS[-1]), 0.01],
        "support": [float(SUPPORT[0]), float(SUPPORT[-1])],
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "scene"},
        "pact_minus_nested": {
            "point": next(row["ncsAURC"] for row in summary_rows if row["method"] == "pcecf")
            - next(row["ncsAURC"] for row in summary_rows if row["method"] == "nested_evidential_composition"),
            "ci95_low": float(np.quantile(aggregate_differences, 0.025)),
            "ci95_high": float(np.quantile(aggregate_differences, 0.975)),
        },
        "input_sha256": {
            "source_records": sha256(REPO / "data" / "controlled" / "source_records.jsonl.gz"),
            "evaluation_labels": sha256(REPO / "data" / "controlled" / "evaluation_labels.jsonl.gz"),
            "pcecf_study": sha256(REPO / "experiments" / "pcecf_study.py"),
            "pact_operator": sha256(REPO / "src" / "action_admission" / "pcecf.py"),
            "hierarchy_matched_cautious": sha256(REPO / "src" / "action_admission" / "hierarchical_cautious.py"),
        },
        "claim_boundary": "Controlled joint scene--condition analysis. The main-study concentration schedule is transferred without reselection; score thresholds exclude the held condition and test scenes.",
    }
    (OUT / "gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "contrast": payload["pact_minus_nested"], "conditions": condition_rows}, indent=2))


if __name__ == "__main__":
    main()
