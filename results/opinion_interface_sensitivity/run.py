#!/usr/bin/env python3
"""Replay the frozen opinion-interface stress grid with full PACT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "scripts", ROOT / "research_upgrade" / "pcecf_v2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_v1924_opinion_emitter_assumption_stress as v1924  # noqa: E402
import run_controlled_gates as gates  # noqa: E402
from pcecf_v2 import forward  # noqa: E402


PACT_CURVE = ROOT / "research_upgrade" / "pcecf_v2" / "outputs" / "controlled_hardened" / "table2_shared_verifier_risk_coverage.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fold_concentrations() -> dict[int, float]:
    selected = json.loads(v1924.NESTED_GATE.read_text(encoding="utf-8"))["selected_backbones"]
    return {
        int(row["outer_fold"]): float(str(row["selected_method"]).rsplit("c", 1)[1])
        for row in selected
    }


def pact_prediction_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    folds = v1924.v1851.scene_fold_map(records, v1924.v1885.FOLDS)
    concentrations = fold_concentrations()
    unique_concentrations = sorted(set(concentrations.values()))
    lists = {
        key: [[] for _ in range(v1924.v1885.FOLDS)]
        for key in ("score", "eligible", "verified", "correct")
    }
    fold_values: list[int] = []
    clean_values: list[bool] = []

    for record in records:
        fold = folds[v1924.v1851.cluster_id(str(record["record_id"]))]
        reference = v1924.v1885.reference_fields(record)["preferred_contract"]
        fold_values.append(fold)
        clean_values.append(str(record.get("metadata", {}).get("stress_scenario")) == "clean_control")
        sources = gates.pcecf_sources(record)
        observed = gates.observed_source_count(record)
        predictions: dict[float, tuple[str, float]] = {}
        verifier_cache: dict[str, bool] = {}
        for concentration in unique_concentrations:
            output = forward(
                sources,
                concentration=concentration,
                required_roles_by_class=None,
                expected_source_ids=gates.SOURCE_NAMES,
            )
            label = gates.CONTRACT_CLASSES[output.predicted_index]
            predictions[concentration] = (label, float(output.selection_score))
            if label not in verifier_cache:
                verifier_cache[label] = v1924.v1885.verifier(record, label)[0]
        for outer in range(v1924.v1885.FOLDS):
            label, score = predictions[concentrations[outer]]
            lists["score"][outer].append(score)
            lists["eligible"][outer].append(observed >= 2)
            lists["verified"][outer].append(verifier_cache[label])
            lists["correct"][outer].append(label == reference)

    return {
        "score": np.asarray(lists["score"], dtype=float),
        "eligible": np.asarray(lists["eligible"], dtype=bool),
        "verified": np.asarray(lists["verified"], dtype=bool),
        "correct": np.asarray(lists["correct"], dtype=bool),
        "fold": np.asarray(fold_values, dtype=np.int8),
        "clean": np.asarray(clean_values, dtype=bool),
    }


def archived_pact_naurc(grid: np.ndarray = v1924.STRESS_GRID) -> float:
    grouped: dict[float, list[float]] = {}
    with PACT_CURVE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "pcecf_v2":
                grouped.setdefault(float(row["coverage"]), []).append(float(row["wrong_admitted"]))
    x = np.asarray(sorted(grouped), dtype=float)
    y = np.asarray([sum(grouped[value]) / len(grouped[value]) for value in x])
    if not len(x) or x[0] > grid[0] or x[-1] < grid[-1]:
        raise RuntimeError("archived PACT frontier does not span the stress grid")
    return float(np.trapezoid(np.interp(grid, x, y), grid) / (grid[-1] - grid[0]))


def baseline_gate() -> dict[str, Any]:
    upstream_preflight = v1924.preflight()
    upstream_baseline = v1924.baseline_check()
    records = v1924.read_jsonl(v1924.FROZEN_RECORDS)
    observed = v1924.summarize_method(pact_prediction_arrays(records))
    archived = archived_pact_naurc()
    difference = abs(float(observed["normalized_aurc"]) - archived)
    checks = {
        "upstream_preflight_pass": upstream_preflight["status"] == "pass",
        "nested_and_pooling_baseline_pass": upstream_baseline["status"] == "pass",
        "record_count_31200": len(records) == 31200,
        "pact_stress_naurc_matches_archive": difference <= 2e-6,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tolerance": 2e-6,
        "pact_stress_naurc_observed": float(observed["normalized_aurc"]),
        "pact_stress_naurc_archived": archived,
        "pact_stress_naurc_abs_difference": difference,
        "pact_anchor_metrics": observed,
        "upstream_preflight": upstream_preflight,
        "upstream_baseline": upstream_baseline,
        "input_hashes": {
            "base_records": sha256(v1924.BASE_RECORDS),
            "frozen_records": sha256(v1924.FROZEN_RECORDS),
            "nested_gate": sha256(v1924.NESTED_GATE),
            "verifier": sha256(ROOT / "E3_HRC" / "E3_HRC-main" / "src" / "safe_fuse.py"),
            "pact_operator": sha256(ROOT / "research_upgrade" / "pcecf_v2" / "pcecf_v2.py"),
            "archived_pact_curve": sha256(PACT_CURVE),
        },
    }


def evaluate_setting(setting: Mapping[str, Any]) -> dict[str, Any]:
    base = v1924.read_jsonl(v1924.BASE_RECORDS)
    records, fallbacks = v1924.rebuild_records(base, setting)
    inherited = v1924.prediction_arrays(records)
    nested = v1924.summarize_method(inherited["nested_evidential"])
    pooling = v1924.summarize_method(inherited["registered_lineage"])
    pact = v1924.summarize_method(pact_prediction_arrays(records))
    return {
        **setting,
        "records": len(records),
        "dirichlet_projection_fallbacks": fallbacks,
        "nested_naurc": nested["normalized_aurc"],
        "pooling_naurc": pooling["normalized_aurc"],
        "pact_naurc": pact["normalized_aurc"],
        "pact_minus_nested_naurc": pact["normalized_aurc"] - nested["normalized_aurc"],
        "pact_minus_pooling_naurc": pact["normalized_aurc"] - pooling["normalized_aurc"],
        "nested_coverage": nested["coverage"],
        "pooling_coverage": pooling["coverage"],
        "pact_coverage": pact["coverage"],
        "nested_wrong_all": nested["wrong_all"],
        "pooling_wrong_all": pooling["wrong_all"],
        "pact_wrong_all": pact["wrong_all"],
        "nested_wrong_admitted": nested["wrong_admitted"],
        "pooling_wrong_admitted": pooling["wrong_admitted"],
        "pact_wrong_admitted": pact["wrong_admitted"],
        "nested_correct_all": nested["correct_all"],
        "pooling_correct_all": pooling["correct_all"],
        "pact_correct_all": pact["correct_all"],
        "nested_clean_coverage": nested["clean_coverage"],
        "pooling_clean_coverage": pooling["clean_coverage"],
        "pact_clean_coverage": pact["clean_coverage"],
    }


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "q975": float(np.quantile(array, 0.975)),
        "max": float(np.max(array)),
    }


def summarize(rows: list[dict[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    numeric_fields = (
        "nested_naurc", "pooling_naurc", "pact_naurc",
        "pact_minus_nested_naurc", "pact_minus_pooling_naurc",
        "nested_coverage", "pooling_coverage", "pact_coverage",
        "nested_wrong_all", "pooling_wrong_all", "pact_wrong_all",
    )
    finite = all(math.isfinite(float(row[field])) for row in rows for field in numeric_fields)
    checks = {
        "baseline_gate_pass": baseline["status"] == "pass",
        "requested_setting_count_complete": len(rows) > 0,
        "all_settings_have_31200_records": all(int(row["records"]) == 31200 for row in rows),
        "all_primary_metrics_finite": finite,
        "setting_ids_unique": len({int(row["setting_id"]) for row in rows}) == len(rows),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "settings": len(rows),
        "common_support": [float(v1924.STRESS_GRID[0]), float(v1924.STRESS_GRID[-1])],
        "summary": {
            field: quantiles([float(row[field]) for row in rows])
            for field in numeric_fields
        },
        "counts": {
            "pact_lower_naurc_than_nested": sum(float(row["pact_minus_nested_naurc"]) < 0.0 for row in rows),
            "pact_lower_naurc_than_pooling": sum(float(row["pact_minus_pooling_naurc"]) < 0.0 for row in rows),
        },
        "claim_boundary": (
            "Direction-free Stage-6 robustness replay on the frozen controlled opinion interface. "
            "It does not add native perception, physical execution, or a statistical independence claim."
        ),
    }


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rows.append(f"{sha256(path)}  {path.name}")
    (output / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.settings <= 0 or args.settings & (args.settings - 1):
        raise ValueError("settings must be a positive power of two")
    args.out.mkdir(parents=True, exist_ok=True)

    baseline = baseline_gate()
    (args.out / "baseline_gate.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if baseline["status"] != "pass":
        raise RuntimeError(baseline)

    settings = v1924.sobol_settings(args.settings)
    if args.workers == 1:
        rows = [evaluate_setting(setting) for setting in settings]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(evaluate_setting, settings))
    rows.sort(key=lambda row: int(row["setting_id"]))
    report = summarize(rows, baseline)
    report["requested_settings"] = args.settings
    report["workers"] = args.workers
    write_csv(args.out / "setting_results.csv", rows)
    (args.out / "gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
