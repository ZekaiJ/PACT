#!/usr/bin/env python3
"""Synthesize matched learned-set and structured-fusion holdout results."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "results" / "learned_set_fusion"
LOSO = ROOT / "results" / "learned_set_fusion_loso"
SPLIT = ROOT / "results" / "learned_set_fusion_split_audit"
STRUCTURED = ROOT / "results" / "unseen_stress_generalization"
OUT = ROOT / "results" / "learned_set_fusion_comparison"
SCENE_SUPPORT = (0.10, 0.40)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pooled_naurc(role: str, support: tuple[float, float]) -> float:
    grid = np.linspace(support[0], support[1], 36)
    grouped: dict[float, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in read_csv(STRUCTURED / "train_derived_risk_coverage.csv"):
        if row["role"] != role:
            continue
        totals = grouped[float(row["target_coverage"])]
        totals[0] += int(row["n"])
        totals[1] += int(row["admitted"])
        totals[2] += int(row["wrong"])
    achieved: dict[float, list[float]] = defaultdict(list)
    for n, admitted, wrong in grouped.values():
        achieved[admitted / n].append(wrong / admitted if admitted else 0.0)
    x = np.asarray(sorted(achieved))
    y = np.asarray([float(np.mean(achieved[value])) for value in x])
    if x[0] > grid[0] or x[-1] < grid[-1]:
        raise RuntimeError(f"{role} does not span common support: {x[0]}--{x[-1]}")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(np.interp(grid, x, y), grid) / (grid[-1] - grid[0]))

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene_gate = json.loads((SCENE / "promotion_gate.json").read_text(encoding="utf-8"))
    loso_gate = json.loads((LOSO / "promotion_gate.json").read_text(encoding="utf-8"))
    split_gate = json.loads((SPLIT / "split_audit.json").read_text(encoding="utf-8"))
    joint_support = tuple(float(value) for value in loso_gate["normalized_aurc_support"])
    structured_gate = json.loads((STRUCTURED / "gate.json").read_text(encoding="utf-8"))
    hashes = {
        scene_gate["input_sha256"],
        loso_gate["input_sha256"],
        split_gate["input_sha256"],
        structured_gate["input_sha256"],
    }
    if len(hashes) != 1:
        raise RuntimeError(f"input hash mismatch: {sorted(hashes)}")

    aurc = {row["architecture"]: row for row in scene_gate["normalized_aurc"]}
    anchors = {
        row["architecture"]: row
        for row in scene_gate["anchor_metrics"]
        if row["protocol"] == "verifier_refit"
    }
    rows: list[dict[str, Any]] = []
    labels = {
        "ordered_mlp": "Ordered MLP",
        "deep_sets": "Deep Sets",
        "set_transformer": "Set Transformer",
    }
    for architecture in ("ordered_mlp", "deep_sets", "set_transformer"):
        rows.append(
            {
                "holdout": "scene",
                "model": labels[architecture],
                "normalized_aurc": aurc[architecture]["normalized_aurc"],
                "naurc_support": "0.10--0.40",
                "coverage": anchors[architecture]["coverage"],
                "wrong_all": anchors[architecture]["wrong_all"],
                "correct_all": anchors[architecture]["correct_all"],
            }
        )
    structured = structured_gate["pooled_held_out_summary"]["nested_selected"]
    rows.extend(
        [
            {
                "holdout": "scene_and_configuration",
                "model": "Nested evidential composition",
                "normalized_aurc": pooled_naurc("nested_selected", joint_support),
                "naurc_support": f"{joint_support[0]:.2f}--{joint_support[1]:.2f}",
                "coverage": structured["coverage"],
                "wrong_all": structured["wrong_all"],
                "correct_all": structured["correct_all"],
            },
            {
                "holdout": "scene_and_configuration",
                "model": "Set Transformer",
                "normalized_aurc": loso_gate["normalized_aurc"],
                "naurc_support": f"{joint_support[0]:.2f}--{joint_support[1]:.2f}",
                "coverage": loso_gate["anchor"]["coverage"],
                "wrong_all": loso_gate["anchor"]["wrong_all"],
                "correct_all": loso_gate["anchor"]["correct_all"],
            },
        ]
    )
    write_csv(OUT / "comparison_table.csv", rows)
    payload = {
        "version": "v1902",
        "status": "retain_supervised_comparator",
        "input_sha256": next(iter(hashes)),
        "scene_nAURC_support": list(SCENE_SUPPORT),
        "joint_nAURC_support": list(joint_support),
        "scene_holdout": {
            "protocol": scene_gate["validation_protocol"],
            "models": rows[:3],
            "exact_runtime_feature_overlap_rate": split_gate["scene_holdout_exact_overlap_rate"],
        },
        "joint_scene_configuration_holdout": {
            "protocol": loso_gate["protocol"],
            "models": rows[3:],
            "exact_runtime_feature_overlap_rate": split_gate["joint_scene_stress_exact_overlap_rate"],
            "set_transformer_nAURC_ci": next(
                row for row in loso_gate["scene_bootstrap"] if row["metric"] == "normalized_aurc"
            ),
        },
        "interpretation": (
            "Direct supervised source-set fusion interpolates accurately across held-out scenes when degradation "
            "configurations are represented during training, but it does not transfer reliably to an unseen "
            "configuration. The structured nested evidential composition is retained as the primary backbone."
        ),
        "claim_boundary": (
            "The comparison uses controlled generator configurations and scene-disjoint folds. It does not establish "
            "open-world degradation, physical deployment, or participant-facing safety."
        ),
    }
    (OUT / "gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
