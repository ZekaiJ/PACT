"""Separate nuisance emitter assumptions from genuine evidence degradation.

The earlier joint sweep mixed two distinct questions: robustness to probability
parameterization and responsiveness to weaker partially occluded evidence.  This
script evaluates them independently under the frozen folds, verifier, evaluator,
and stress expansion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_v1924_opinion_emitter_assumption_stress as v1924


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "decomposed_emitter_robustness"
NUISANCE_SEED = 1935
RESPONSE_SEED = 1946
FIXED_OCCLUSION_STRENGTH = 1.0
RESPONSE_LEVELS = (0.5, 1.0, 1.5)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nuisance_settings(count: int, seed: int, id_offset: int = 0) -> list[dict[str, Any]]:
    if count <= 0 or count & (count - 1):
        raise ValueError("settings must be a positive power of two")
    unit = v1924.qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(int(math.log2(count)))
    rows = []
    for index, values in enumerate(unit):
        rows.append(
            {
                "setting_id": id_offset + index,
                "language_pmax": v1924.LANGUAGE_MAX[
                    min(int(values[0] * len(v1924.LANGUAGE_MAX)), len(v1924.LANGUAGE_MAX) - 1)
                ],
                "dirichlet_kappa": v1924.DIRICHLET_KAPPA[
                    min(int(values[1] * len(v1924.DIRICHLET_KAPPA)), len(v1924.DIRICHLET_KAPPA) - 1)
                ],
                "quality_multiplier": 0.8 + 0.4 * float(values[2]),
                "conflict_multiplier": 0.8 + 0.4 * float(values[3]),
                "occlusion_strength": FIXED_OCCLUSION_STRENGTH,
            }
        )
    return rows


def detailed_evaluate(setting: Mapping[str, Any]) -> dict[str, Any]:
    base = v1924.read_jsonl(v1924.BASE_RECORDS)
    records, fallbacks = v1924.rebuild_records(base, setting)
    arrays = v1924.prediction_arrays(records)
    nested = v1924.summarize_method(arrays["nested_evidential"])
    registered = v1924.summarize_method(arrays["registered_lineage"])
    accepted = v1924.accepted_at_target(arrays["nested_evidential"], v1924.ANCHOR)
    correct = v1924.selected_correct(arrays["nested_evidential"])
    clean = arrays["nested_evidential"]["clean"]
    occlusion = np.asarray(
        [str(record.get("metadata", {}).get("occlusion_band", "none")) for record in records],
        dtype=object,
    )
    folds = v1924.v1851.scene_fold_map(records, v1924.v1885.FOLDS)
    selected = json.loads(v1924.NESTED_GATE.read_text(encoding="utf-8"))["selected_backbones"]
    concentrations = {
        int(row["outer_fold"]): float(str(row["selected_method"]).rsplit("c", 1)[1])
        for row in selected
    }
    unoccluded_runtime = []
    for record in records:
        metadata = record.get("metadata", {})
        if (
            str(metadata.get("occlusion_band", "none")) != "none"
            or str(metadata.get("stress_scenario")) != "clean_control"
        ):
            continue
        outer = folds[v1924.v1851.cluster_id(str(record["record_id"]))]
        prediction = v1924.tmc.predict(record, concentrations[outer])
        label = str(prediction["predicted_contract"])
        verified, _ = v1924.v1885.verifier(record, label)
        unoccluded_runtime.append(
            (
                str(record["record_id"]),
                label,
                round(1.0 - float(prediction["uncertainty"]), 15),
                bool(prediction["eligible"]),
                bool(verified),
            )
        )
    unoccluded_runtime_hash = hashlib.sha256(
        json.dumps(unoccluded_runtime, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = {
        **dict(setting),
        "records": len(records),
        "dirichlet_projection_fallbacks": fallbacks,
        "nested_naurc": nested["normalized_aurc"],
        "registered_naurc": registered["normalized_aurc"],
        "naurc_gain": registered["normalized_aurc"] - nested["normalized_aurc"],
        "ranking_preserved": nested["normalized_aurc"] < registered["normalized_aurc"],
        "nested_coverage": nested["coverage"],
        "nested_wrong_all": nested["wrong_all"],
        "nested_wrong_admitted": nested["wrong_admitted"],
        "nested_correct_all": nested["correct_all"],
        "nested_clean_coverage": nested["clean_coverage"],
        "registered_coverage": registered["coverage"],
        "registered_wrong_all": registered["wrong_all"],
        "registered_wrong_admitted": registered["wrong_admitted"],
        "registered_correct_all": registered["correct_all"],
        "registered_clean_coverage": registered["clean_coverage"],
        "copy_mass_max_abs_drift": v1924.copy_mass_drift(),
        "unoccluded_clean_runtime_sha256": unoccluded_runtime_hash,
    }
    for band in ("none", "partial"):
        mask = clean & (occlusion == band)
        denominator = int(mask.sum())
        output[f"nested_clean_{band}_coverage"] = float((accepted & mask).sum()) / denominator
        output[f"nested_clean_{band}_correct_all"] = float((accepted & correct & mask).sum()) / denominator
        output[f"nested_clean_{band}_wrong_all"] = float((accepted & ~correct & mask).sum()) / denominator
    return output


def response_settings(count: int) -> list[dict[str, Any]]:
    anchors = nuisance_settings(count, RESPONSE_SEED, id_offset=10000)
    rows = []
    for pair_id, anchor in enumerate(anchors):
        for level in RESPONSE_LEVELS:
            row = dict(anchor)
            row["pair_id"] = pair_id
            row["occlusion_strength"] = level
            rows.append(row)
    return rows


def response_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_level: list[dict[str, Any]] = []
    for level in RESPONSE_LEVELS:
        subset = [row for row in rows if float(row["occlusion_strength"]) == level]
        by_level.append(
            {
                "occlusion_strength": level,
                "settings": len(subset),
                "partial_clean_coverage_mean": float(np.mean([float(row["nested_clean_partial_coverage"]) for row in subset])),
                "partial_clean_coverage_median": float(np.median([float(row["nested_clean_partial_coverage"]) for row in subset])),
                "partial_clean_wrong_all_max": max(float(row["nested_clean_partial_wrong_all"]) for row in subset),
                "unoccluded_clean_coverage_mean": float(np.mean([float(row["nested_clean_none_coverage"]) for row in subset])),
                "nested_naurc_mean": float(np.mean([float(row["nested_naurc"]) for row in subset])),
                "ranking_preservation_rate": float(np.mean([str(row["ranking_preserved"]).lower() == "true" for row in subset])),
            }
        )

    exact_unoccluded = True
    monotone_pairs = 0
    pair_count = len({int(row["pair_id"]) for row in rows})
    for pair_id in range(pair_count):
        subset = sorted(
            [row for row in rows if int(row["pair_id"]) == pair_id],
            key=lambda row: float(row["occlusion_strength"]),
        )
        runtime_hashes = {str(row["unoccluded_clean_runtime_sha256"]) for row in subset}
        exact_unoccluded &= len(runtime_hashes) == 1
        partial = [float(row["nested_clean_partial_coverage"]) for row in subset]
        monotone_pairs += int(partial[0] >= partial[1] >= partial[2])

    partial_means = [float(row["partial_clean_coverage_mean"]) for row in by_level]
    partial_medians = [float(row["partial_clean_coverage_median"]) for row in by_level]
    checks = {
        "direct_unoccluded_clean_outputs_exactly_invariant": exact_unoccluded,
        "partial_clean_coverage_mean_monotone": partial_means[0] >= partial_means[1] >= partial_means[2],
        "partial_clean_coverage_median_monotone": partial_medians[0] >= partial_medians[1] >= partial_medians[2],
        "overall_wrong_all_max_at_most_0_02": max(float(row["nested_wrong_all"]) for row in rows) <= 0.02,
        "ranking_preserved_in_at_least_95pct_response_settings": float(
            np.mean([str(row["ranking_preserved"]).lower() == "true" for row in rows])
        ) >= 0.95,
        "partial_clean_coverage_monotone_in_at_least_80pct_pairs": monotone_pairs / pair_count >= 0.80,
    }
    diagnostics = {
        "checks": checks,
        "monotone_pair_rate": monotone_pairs / pair_count,
        "partial_clean_wrong_all_max": max(float(row["nested_clean_partial_wrong_all"]) for row in rows),
        "overall_wrong_all_max": max(float(row["nested_wrong_all"]) for row in rows),
        "partial_clean_coverage_mean_drop_0_5_to_1_5": partial_means[0] - partial_means[2],
        "partial_clean_coverage_median_drop_0_5_to_1_5": partial_medians[0] - partial_medians[2],
    }
    return by_level, diagnostics


def build_report(
    nuisance_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    response_diagnostics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    gains = np.asarray([float(row["naurc_gain"]) for row in nuisance_rows])
    ranking_rate = float(np.mean([str(row["ranking_preserved"]).lower() == "true" for row in nuisance_rows]))
    wrong_margin = max(
        float(row["nested_wrong_all"]) - float(row["registered_wrong_all"])
        for row in nuisance_rows
    )
    baseline_clean = float(baseline["observed"]["nested_evidential"]["clean_coverage"])
    clean_loss = max(baseline_clean - float(row["nested_clean_coverage"]) for row in nuisance_rows)
    nuisance_checks = {
        "nested_lower_naurc_in_at_least_95pct_settings": ranking_rate >= 0.95,
        "median_naurc_gain_exceeds_0_03": float(np.median(gains)) > 0.03,
        "worst_anchor_wrong_all_margin_at_most_0_01": wrong_margin <= 0.01,
        "worst_clean_coverage_loss_below_0_05": clean_loss < 0.05,
        "copy_mass_invariance": max(float(row["copy_mass_max_abs_drift"]) for row in nuisance_rows) <= 1e-12,
        "no_reference_field_leakage": preflight["checks"]["transform_static_forbidden_hits_zero"]
        and preflight["checks"]["transform_reference_permutation_invariant"],
        "frozen_baseline_reproduced": baseline["status"] == "pass" and preflight["status"] == "pass",
    }
    response_checks = dict(response_diagnostics["checks"])
    all_checks = {**{f"nuisance::{key}": value for key, value in nuisance_checks.items()}, **{f"response::{key}": value for key, value in response_checks.items()}}
    return {
        "version": "v1925",
        "status": "promote" if all(all_checks.values()) else "retain_as_diagnostic",
        "scientific_design": {
            "nuisance_question": "Does the method ranking survive plausible opinion sharpness and quality/conflict parameterization?",
            "evidence_response_question": "Does weaker partially occluded evidence induce selective withholding while leaving unoccluded cases unchanged?",
            "nuisance_occlusion_strength": FIXED_OCCLUSION_STRENGTH,
            "response_levels": RESPONSE_LEVELS,
            "nuisance_sequence": f"scrambled Sobol seed {NUISANCE_SEED}",
            "response_sequence": f"paired scrambled Sobol seed {RESPONSE_SEED}",
            "frozen_components": "scene folds, verifier, evaluator, nested backbone choices, stress expansion, and acceptance protocol",
        },
        "nuisance_settings": len(nuisance_rows),
        "response_settings": len(response_rows),
        "preflight": preflight,
        "baseline_reproduction": baseline,
        "nuisance_summary": {
            "ranking_preservation_rate": ranking_rate,
            "naurc_gain_median": float(np.median(gains)),
            "naurc_gain_q025": float(np.quantile(gains, 0.025)),
            "naurc_gain_q975": float(np.quantile(gains, 0.975)),
            "worst_nested_minus_registered_wrong_all": wrong_margin,
            "baseline_nested_clean_coverage": baseline_clean,
            "nested_clean_coverage_min": min(float(row["nested_clean_coverage"]) for row in nuisance_rows),
            "worst_clean_coverage_loss": clean_loss,
        },
        "evidence_response_summary": dict(response_diagnostics),
        "promotion_criteria": all_checks,
        "claim_boundary": (
            "The nuisance sweep tests robustness of the controlled opinion interface; the paired response curve tests "
            "whether observable evidence degradation changes selective admission in the expected direction. Neither "
            "analysis constitutes raw-sensor, physical-execution, or participant validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuisance-settings", type=int, default=256)
    parser.add_argument("--response-pairs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-nuisance", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    preflight = v1924.preflight()
    baseline = v1924.baseline_check()
    if preflight["status"] != "pass" or baseline["status"] != "pass":
        raise RuntimeError({"preflight": preflight, "baseline": baseline})

    nuisance = nuisance_settings(args.nuisance_settings, NUISANCE_SEED)
    response = response_settings(args.response_pairs)
    if args.reuse_nuisance:
        with (OUT / "nuisance_setting_results.csv").open(encoding="utf-8", newline="") as handle:
            nuisance_rows = list(csv.DictReader(handle))
    elif args.workers == 1:
        nuisance_rows = [detailed_evaluate(setting) for setting in nuisance]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            nuisance_rows = list(executor.map(detailed_evaluate, nuisance))

    if args.workers == 1:
        response_rows = [detailed_evaluate(setting) | {"pair_id": setting["pair_id"]} for setting in response]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            response_rows = list(executor.map(detailed_evaluate, response))

    response_by_level, response_diagnostics = response_summary(response_rows)
    payload = build_report(nuisance_rows, response_rows, response_diagnostics, baseline, preflight)
    write_csv(OUT / "nuisance_setting_results.csv", nuisance_rows)
    write_csv(OUT / "evidence_response_results.csv", response_rows)
    write_csv(OUT / "evidence_response_summary.csv", response_by_level)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
