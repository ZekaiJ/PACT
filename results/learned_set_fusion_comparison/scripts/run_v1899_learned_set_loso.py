#!/usr/bin/env python3
"""Joint scene- and stress-held-out evaluation for the invariant learned fusion model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v1898_learned_set_fusion as v1898  # noqa: E402

JOINT_COMMON_MIN = 0.10
JOINT_COMMON_MAX = 0.35


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = v1898.read_jsonl(args.input)
    if len(records) != 31200:
        raise RuntimeError(f"expected 31,200 records, found {len(records)}")
    array = np.asarray([[v1898.token_features(record, name) for name in v1898.SOURCES] for record in records], dtype=np.float32)
    labels_np = np.asarray([v1898.CLASSES.index(str(record["gold_contract"])) for record in records], dtype=np.int64)
    scenes = np.asarray([str(record.get("metadata", {}).get("scene_id", "")) for record in records])
    stresses = np.asarray([str(record.get("metadata", {}).get("stress_scenario", "")) for record in records])
    unique_scenes = sorted(set(scenes))
    unique_stresses = sorted(set(stresses))
    if len(unique_scenes) != 48 or len(unique_stresses) != 13:
        raise RuntimeError({"scenes": len(unique_scenes), "stresses": len(unique_stresses)})
    scene_to_fold = {scene: index % v1898.OUTER_FOLDS for index, scene in enumerate(unique_scenes)}
    folds = np.asarray([scene_to_fold[scene] for scene in scenes], dtype=np.int64)
    features = torch.from_numpy(array)
    labels = torch.from_numpy(labels_np)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    training_rows: list[dict[str, Any]] = []
    fold_payload: dict[tuple[str, int], dict[str, Any]] = {}
    for heldout in unique_stresses:
        for outer in range(v1898.OUTER_FOLDS):
            validation_fold = (outer + 1) % v1898.OUTER_FOLDS
            train_indices = np.where((stresses != heldout) & (folds != outer) & (folds != validation_fold))[0]
            validation_indices = np.where((stresses != heldout) & (folds == validation_fold))[0]
            test_indices = np.where((stresses == heldout) & (folds == outer))[0]
            validation_probabilities = []
            validation_selections = []
            test_probabilities = []
            test_selections = []
            for seed in v1898.SEEDS:
                model, row = v1898.train_one(
                    "set_transformer",
                    features,
                    labels,
                    train_indices,
                    validation_indices,
                    seed + 100 * outer + 1000 * unique_stresses.index(heldout),
                    device,
                )
                row.update(
                    {
                        "heldout_stress": heldout,
                        "outer_fold": outer,
                        "validation_fold": validation_fold,
                        "train_rows": len(train_indices),
                        "validation_rows": len(validation_indices),
                        "test_rows": len(test_indices),
                    }
                )
                training_rows.append(row)
                val_prob, val_sel = v1898.predict(model, features[validation_indices], 2048, device)
                test_prob, test_sel = v1898.predict(model, features[test_indices], 2048, device)
                validation_probabilities.append(val_prob)
                validation_selections.append(val_sel)
                test_probabilities.append(test_prob)
                test_selections.append(test_sel)
            val_prob = np.mean(np.stack(validation_probabilities), axis=0)
            val_sel = np.mean(np.stack(validation_selections), axis=0)
            test_prob = np.mean(np.stack(test_probabilities), axis=0)
            test_sel = np.mean(np.stack(test_selections), axis=0)
            val_pred = val_prob.argmax(axis=1)
            test_pred = test_prob.argmax(axis=1)
            val_verifier = np.asarray(
                [v1898.common_verifier(records[index], v1898.CLASSES[int(predicted)])[0] for index, predicted in zip(validation_indices, val_pred)],
                dtype=bool,
            )
            test_verifier = np.asarray(
                [v1898.common_verifier(records[index], v1898.CLASSES[int(predicted)])[0] for index, predicted in zip(test_indices, test_pred)],
                dtype=bool,
            )
            fold_payload[(heldout, outer)] = {
                "validation_scores": val_sel,
                "validation_verifier": val_verifier,
                "test_indices": test_indices,
                "test_scores": test_sel,
                "test_verifier": test_verifier,
                "test_prediction": test_pred,
            }

    targets = tuple(sorted(set(v1898.TARGETS + (v1898.ANCHOR,))))
    threshold_rows: list[dict[str, Any]] = []
    decisions_by_target: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        for heldout in unique_stresses:
            for outer in range(v1898.OUTER_FOLDS):
                payload = fold_payload[(heldout, outer)]
                threshold, status = v1898.choose_threshold(payload["validation_scores"], payload["validation_verifier"], target)
                threshold_rows.append(
                    {
                        "heldout_stress": heldout,
                        "outer_fold": outer,
                        "target_coverage": target,
                        "threshold": threshold,
                        "status": status,
                    }
                )
                accepted = (payload["test_scores"] >= threshold) & payload["test_verifier"]
                for position, index in enumerate(payload["test_indices"]):
                    decisions_by_target[target].append(
                        {
                            "scene_id": scenes[index],
                            "stress_scenario": stresses[index],
                            "accepted": bool(accepted[position]),
                            "predicted": v1898.CLASSES[int(payload["test_prediction"][position])],
                            "gold": v1898.CLASSES[int(labels_np[index])],
                            "acceptable_contracts": list(v1898.shared_evaluation.reference_fields(records[index])["acceptable_contracts"]),
                        }
                    )
    if any(len(rows) != 31200 for rows in decisions_by_target.values()):
        raise RuntimeError({target: len(rows) for target, rows in decisions_by_target.items()})
    curve_rows = [{"target_coverage": target, **v1898.counts(decisions_by_target[target])} for target in targets]
    point_aurc = v1898.normalized_aurc(
        [row for row in curve_rows if row["target_coverage"] in v1898.TARGETS],
        common_min=JOINT_COMMON_MIN,
        common_max=JOINT_COMMON_MAX,
    )
    anchor = next(row for row in curve_rows if row["target_coverage"] == v1898.ANCHOR)
    stress_rows = [
        {"heldout_stress": stress, **v1898.counts(row for row in decisions_by_target[v1898.ANCHOR] if row["stress_scenario"] == stress)}
        for stress in unique_stresses
    ]

    rng = np.random.default_rng(1899)
    per_scene_curve: dict[tuple[float, str], dict[str, float]] = {}
    for target in v1898.TARGETS:
        rows = decisions_by_target[target]
        for scene in unique_scenes:
            per_scene_curve[(target, scene)] = v1898.counts(row for row in rows if row["scene_id"] == scene)
    anchor_rows_all = decisions_by_target[v1898.ANCHOR]
    per_scene_anchor = {
        scene: v1898.counts(row for row in anchor_rows_all if row["scene_id"] == scene)
        for scene in unique_scenes
    }
    aurc_draws = []
    anchor_draws = {metric: [] for metric in ("coverage", "wrong_all", "correct_all")}
    for _ in range(v1898.BOOTSTRAPS):
        weights = rng.multinomial(len(unique_scenes), np.full(len(unique_scenes), 1.0 / len(unique_scenes)))
        synthetic_curve = []
        for target in v1898.TARGETS:
            totals = {key: 0.0 for key in ("n", "admitted", "wrong", "correct")}
            for weight, scene in zip(weights, unique_scenes):
                values = per_scene_curve[(target, scene)]
                for key in totals:
                    totals[key] += weight * values[key]
            synthetic_curve.append(
                {
                    "coverage": totals["admitted"] / totals["n"],
                    "wrong_admitted": totals["wrong"] / totals["admitted"] if totals["admitted"] else 0.0,
                }
            )
        aurc_draws.append(
            v1898.normalized_aurc(
                synthetic_curve,
                common_min=JOINT_COMMON_MIN,
                common_max=JOINT_COMMON_MAX,
            )
        )
        anchor_totals = {key: 0.0 for key in ("n", "admitted", "wrong", "correct")}
        for weight, scene in zip(weights, unique_scenes):
            values = per_scene_anchor[scene]
            for key in anchor_totals:
                anchor_totals[key] += weight * values[key]
        anchor_draws["coverage"].append(anchor_totals["admitted"] / anchor_totals["n"])
        anchor_draws["wrong_all"].append(anchor_totals["wrong"] / anchor_totals["n"])
        anchor_draws["correct_all"].append(anchor_totals["correct"] / anchor_totals["n"])

    bootstrap_rows = [
        {
            "metric": "normalized_aurc",
            "estimate": point_aurc,
            "ci_low": v1898.percentile(aurc_draws, 0.025),
            "ci_high": v1898.percentile(aurc_draws, 0.975),
            "bootstraps": v1898.BOOTSTRAPS,
        }
    ] + [
        {
            "metric": metric,
            "estimate": anchor[metric],
            "ci_low": v1898.percentile(draws, 0.025),
            "ci_high": v1898.percentile(draws, 0.975),
            "bootstraps": v1898.BOOTSTRAPS,
        }
        for metric, draws in anchor_draws.items()
    ]
    write_csv(args.output_dir / "training_runs.csv", training_rows)
    write_csv(args.output_dir / "thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "risk_coverage_curve.csv", curve_rows)
    write_csv(args.output_dir / "heldout_stress_anchor.csv", stress_rows)
    write_csv(args.output_dir / "scene_bootstrap.csv", bootstrap_rows)
    gate = {
        "version": "v1899",
        "status": "pass_joint_scene_stress_holdout" if anchor["wrong_all"] <= 0.0021 and anchor["coverage"] >= 0.10 else "retain_diagnostic",
        "architecture": "set_transformer",
        "records": len(records),
        "scenes": len(unique_scenes),
        "heldout_stress_configurations": len(unique_stresses),
        "seeds_per_fold": len(v1898.SEEDS),
        "protocol": "each test record is held out jointly by scene fold and stress configuration; thresholds use non-test scenes and non-test configurations",
        "anchor": anchor,
        "normalized_aurc": point_aurc,
        "normalized_aurc_support": [JOINT_COMMON_MIN, JOINT_COMMON_MAX],
        "scene_bootstrap": bootstrap_rows,
        "claim_boundary": (
            "This is joint scene- and generator-configuration transfer on controlled records, not transfer to naturally occurring degradation, physical deployment, or participant-facing safety."
        ),
        "input_sha256": v1898.file_hash(args.input),
    }
    (args.output_dir / "promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
