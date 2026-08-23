#!/usr/bin/env python3
"""Independently audit the corrected native-view analysis artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
MODELS = {
    "primary_32b": ("analysis_test32", "qwen3vl_32b.jsonl.gz", "qwen3vl_32b_environment.json"),
    "replication_8b": ("analysis_test8", "qwen3vl_8b.jsonl.gz", "qwen3vl_8b_environment.json"),
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tied_risk(scores: np.ndarray, wrong: np.ndarray, coverage: float) -> float:
    groups: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for score, error in zip(scores, wrong):
        groups[float(score)][0] += 1.0
        groups[float(score)][1] += float(error)
    target = coverage * len(scores)
    retained = errors = 0.0
    for score in sorted(groups, reverse=True):
        count, group_errors = groups[score]
        take = min(count, target - retained)
        errors += group_errors * take / count
        retained += take
        if retained >= target:
            break
    return errors / target


def ncs(rows: list[dict], score_name: str, grid: np.ndarray) -> float:
    scores = np.asarray([float(row[score_name]) for row in rows])
    p = np.asarray([float(row["p_ready"]) for row in rows])
    y = np.asarray([int(row["y"]) for row in rows])
    wrong = (p > 0.5) != y
    risks = [tied_risk(scores, wrong, float(coverage)) for coverage in grid]
    return float(np.trapezoid(risks, grid) / (grid[-1] - grid[0]))


def macro_ncs(rows: list[dict], m: int, arm: str, score_name: str, grid: np.ndarray) -> float:
    selected = [row for row in rows if int(row["m"]) == m and row["arm"] == arm]
    tasks = sorted({row["task_id"] for row in selected})
    return float(np.mean([ncs([row for row in selected if row["task_id"] == task], score_name, grid) for task in tasks]))


def main() -> None:
    lock = json.loads((HERE / "ANALYSIS_LOCK.json").read_text(encoding="utf-8-sig"))
    amendment = json.loads((HERE / "ANALYSIS_AMENDMENT_TIE_V1.json").read_text(encoding="utf-8-sig"))
    grid = np.asarray(lock["primary_estimand"]["coverage_grid"], dtype=float)
    prompt_path = HERE / "protocol" / "test" / "prompt_pack.jsonl.gz"
    prompts = {row["id"]: row for row in read_jsonl(prompt_path)}
    model_audits = {}
    all_ids_unique = all_probabilities_valid = all_raw_rows_complete = all_status_ok = True
    all_traceable = semantic_ties = independent = True

    for model_key, (analysis_dir, output_name, environment_name) in MODELS.items():
        output_path = HERE / "outputs" / output_name
        outputs = read_jsonl(output_path)
        output_ids = [row["id"] for row in outputs]
        all_ids_unique &= len(output_ids) == len(set(output_ids)) and set(output_ids) == set(prompts)
        all_raw_rows_complete &= len(outputs) == len(prompts) == 45120
        for row in outputs:
            p = float(row["p_ready"])
            evidence = [float(value) for value in row["evidence"]]
            all_probabilities_valid &= (
                math.isfinite(p)
                and 0.0 <= p <= 1.0
                and len(evidence) == 2
                and math.isclose(sum(evidence), 2.0, abs_tol=1e-12)
            )
            all_status_ok &= row.get("status") == "ok"
            prompt = prompts[row["id"]]
            all_traceable &= all(
                row.get(field) == prompt.get(field)
                for field in (
                    "episode_id",
                    "event_id",
                    "window",
                    "physical_view_id",
                    "surface_id",
                    "parent_id",
                    "image_sha256",
                    "prompt_record_sha256",
                    "reference_ready",
                )
            )

        result_dir = HERE / analysis_dir
        records = read_csv(result_dir / "records.csv")
        points = read_csv(result_dir / "point_estimates.csv")
        result = json.loads((result_dir / "PRIMARY_RESULT.json").read_text(encoding="utf-8"))
        expected_record_count = 2256 * 3 * 6
        all_raw_rows_complete &= len(records) == expected_record_count

        for m in (1, 2, 4):
            lineage = [row for row in records if int(row["m"]) == m and row["arm"] == "lineage_unaware"]
            native = [row for row in records if int(row["m"]) == m and row["arm"] == "native_one_per_view"]
            semantic_ties &= len({row["score"] for row in lineage}) == 1
            semantic_ties &= len({row["score"] for row in native}) == 1
        for row in records:
            semantic_ties &= math.isclose(
                float(row["common_score"]),
                max(float(row["p_ready"]), 1.0 - float(row["p_ready"])),
                abs_tol=1e-15,
            )
            if row["arm"] == "exact_dedup":
                semantic_ties &= float(row["budget"]) == 2.0 * float(row["retained"])
        m1 = defaultdict(dict)
        for row in records:
            if int(row["m"]) == 1 and row["arm"] in {"acquisition_registered", "lineage_unaware"}:
                m1[(row["event_id"], row["window"])][row["arm"]] = (
                    row["p_ready"], row["score"], row["budget"]
                )
        semantic_ties &= all(pair["acquisition_registered"] == pair["lineage_unaware"] for pair in m1.values())

        recomputed_primary = []
        recomputed_common = []
        for m in (1, 2, 4):
            registered = macro_ncs(records, m, "acquisition_registered", "score", grid)
            unaware = macro_ncs(records, m, "lineage_unaware", "score", grid)
            common_registered = macro_ncs(records, m, "acquisition_registered", "common_score", grid)
            common_unaware = macro_ncs(records, m, "lineage_unaware", "common_score", grid)
            recomputed_primary.append(unaware - registered)
            recomputed_common.append(common_unaware - common_registered)
        reported_primary = [float(row["estimate"]) for row in result["primary_contrasts"]]
        reported_common = [float(row["estimate"]) for row in result["common_score_contrasts"]]
        independent &= np.allclose(recomputed_primary, reported_primary, atol=1e-12, rtol=0)
        independent &= np.allclose(recomputed_common, reported_common, atol=1e-12, rtol=0)

        m4_points = defaultdict(dict)
        for row in points:
            if int(row["m"]) == 4:
                m4_points[row["arm"]][row["metric"]] = float(row["estimate"])
        artifacts = {
            name: file_sha(result_dir / name)
            for name in (
                "PRIMARY_RESULT.json",
                "bootstrap_intervals.csv",
                "paired_contrasts.csv",
                "point_estimates.csv",
                "records.csv",
            )
        }
        model_audits[model_key] = {
            "analysis_artifact_sha256": artifacts,
            "environment_sha256": file_sha(HERE / "outputs" / environment_name),
            "output_content_sha256": content_sha(output_path),
            "primary_contrasts": result["primary_contrasts"],
            "common_score_contrasts": result["common_score_contrasts"],
            "m4_point_estimates": m4_points,
            "independent_primary_recompute": recomputed_primary,
            "independent_common_score_recompute": recomputed_common,
        }

    audit = {
        "experiment": "Native-view foundation-model provenance transfer",
        "generated_utc": amendment["locked_utc"],
        "status": "FROZEN_RESULTS_COMPLETE_WITH_LOCKED_TIE_CORRECTION",
        "analysis_amendment": amendment,
        "locks": {
            "analysis_lock_sha256": file_sha(HERE / "ANALYSIS_LOCK.json"),
            "analysis_amendment_sha256": file_sha(HERE / "ANALYSIS_AMENDMENT_TIE_V1.json"),
            "protocol_lock_sha256": file_sha(HERE / "protocol" / "test" / "PROTOCOL_LOCK.json"),
            "prompt_pack_content_sha256": content_sha(prompt_path),
            "test_analysis_lock_sha256": file_sha(HERE / "TEST_ANALYSIS_LOCK.json"),
        },
        "verification": {
            "all_ids_unique": bool(all_ids_unique),
            "all_probabilities_valid": bool(all_probabilities_valid),
            "all_raw_rows_complete": bool(all_raw_rows_complete),
            "all_status_ok": bool(all_status_ok),
            "all_parent_records_traceable": bool(all_traceable),
            "semantic_tie_checks_pass": bool(semantic_ties),
            "independent_primary_recompute_pass": bool(independent),
        },
        "models": model_audits,
        "interpretation_constraints": [
            "The original native non-vacuity score is constant for lineage-unaware and one-output-per-view arms at fixed multiplicity; fractional tie handling therefore returns their task error rate.",
            "The primary 32B native-score contrast at m=4 includes zero, whereas the 8B replication contrast is positive.",
            "The post-hoc posterior-confidence contrast is positive for 32B and includes zero for 8B; selective effects depend on checkpoint and score readout.",
            "The equal-cardinality shuffled grouping and posterior-confidence readout are labeled post-hoc diagnostics, not preregistered primary endpoints.",
            "Task-level signs remain heterogeneous, and the study does not support checkpoint-universal selective dominance.",
        ],
    }
    if not all(audit["verification"].values()):
        raise RuntimeError(json.dumps(audit["verification"], sort_keys=True))
    output = HERE / "gates" / "FINAL_RESULT_AUDIT.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit["verification"], sort_keys=True))


if __name__ == "__main__":
    main()
