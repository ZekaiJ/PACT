"""Verify the released controlled-study artifact from one command."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "controlled_study.json"
SOURCE = ROOT / "data" / "controlled" / "source_records.jsonl.gz"
LABELS = ROOT / "data" / "controlled" / "evaluation_labels.jsonl.gz"
REFERENCE = ROOT / "results" / "reference"
SCALABILITY = ROOT / "outputs" / "scalability"
PUBLIC_OUTCOME = ROOT / "results" / "public_outcome_closure"
NATIVE_VIEW = ROOT / "results" / "native_view_fm_provenance_transfer"
COMMON_SCORE = ROOT / "results" / "common_score_parity"
HABIT_CHECKPOINT = ROOT / "results" / "habit_checkpoint_admission"
CLAIM_MAP = ROOT / "docs" / "CLAIM_ARTIFACT_MAP.csv"
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
FORBIDDEN_SOURCE_FIELDS = {
    "preferred_contract",
    "acceptable_contracts",
    "cue_family",
    "accepted",
    "correct",
    "wrong",
    "execution_log",
}


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def uncompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in collect_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in collect_keys(child)}
    return set()


def run(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )


def csv_matches(
    observed: Path,
    reference: Path,
    tolerance: float = 1e-12,
) -> bool:
    with observed.open(encoding="utf-8", newline="") as left, reference.open(
        encoding="utf-8",
        newline="",
    ) as right:
        observed_rows = list(csv.DictReader(left))
        reference_rows = list(csv.DictReader(right))
    if len(observed_rows) != len(reference_rows):
        return False
    for observed_row, reference_row in zip(observed_rows, reference_rows):
        if observed_row.keys() != reference_row.keys():
            return False
        for field in observed_row:
            observed_value = observed_row[field]
            reference_value = reference_row[field]
            if observed_value == reference_value:
                continue
            try:
                if math.isclose(
                    float(observed_value),
                    float(reference_value),
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    continue
            except (TypeError, ValueError):
                pass
            return False
    return True



def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release_manifest() -> dict[str, int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if set(manifest) != {"scope", "file_count", "files"}:
        raise AssertionError("Unexpected release-manifest schema")
    entries = manifest["files"]
    if manifest["file_count"] != len(entries):
        raise AssertionError("Release-manifest file count mismatch")
    paths = [entry["path"] for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise AssertionError("Release-manifest paths must be sorted and unique")
    required = {
        "docs/RELEASE_SCOPE.md",
        "requirements-release.txt",
        "analysis/verify_p0_estimand_closure.py",
        "analysis/minimal_attribution_2x2.py",
        "analysis/score_verifier_factorial.py",
        "analysis/verify_secondary_results.py",
        "analysis/verify_recent_mechanism_results.py",
        "analysis/common_score_parity.py",
        "analysis/build_claim_artifact_map.py",
        "analysis/candidate_score_gap_decomposition.py",
        "analysis/joint_scene_configuration_holdout.py",
        "analysis/habit_checkpoint_admission.py",
        "analysis/partition_coarsening_surface.py",
        "docs/CLAIM_ARTIFACT_MAP.csv",
        "docs/CLAIM_ARTIFACT_MAP.md",
        "results/topology_multiplicity/README.md",
        "results/operator_characterization/README.md",
        "results/equal_cardinality_topology/README.md",
        "results/frozen_score_transport/README.md",
        "results/common_score_parity/README.md",
        "results/common_score_parity/PROTOCOL.json",
        "results/common_score_parity/table_panel_b_rows.csv",
        "results/minimal_attribution_2x2/README.md",
        "results/public_outcome_closure/README.md",
        "results/native_view_fm_provenance_transfer/README.md",
        "results/native_view_fm_provenance_transfer/gates/FINAL_RESULT_AUDIT.json",
        "results/score_verifier_factorial/README.md",
        "results/evidence_response_sensitivity/README.md",
        "results/habit_fixed_image_admission/README.md",
        "results/habit_checkpoint_admission/README.md",
        "results/habit_checkpoint_admission/gate.json",
        "results/joint_scene_configuration_holdout/README.md",
        "results/learned_set_fusion_comparison/README.md",
        "results/opinion_interface_sensitivity/README.md",
        "results/partition_coarsening_surface/README.md",
        "results/provenance_corruption_sensitivity/README.md",
        "results/h2o_stage3/verify_released_aggregates.py",
    }
    if not required.issubset(paths):
        raise AssertionError("Release manifest omits a required closure artifact")
    for entry in entries:
        if set(entry) != {"path", "bytes", "sha256"}:
            raise AssertionError("Unexpected release-manifest entry schema")
        path = ROOT / entry["path"]
        if not path.is_file():
            raise AssertionError(f"Manifest path is missing: {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            raise AssertionError(f"Manifest size mismatch: {entry['path']}")
        if file_sha256(path) != entry["sha256"]:
            raise AssertionError(f"Manifest checksum mismatch: {entry['path']}")
    return {"files": len(entries)}


def verify_claim_artifact_map() -> dict[str, Any]:
    expected_fields = [
        "claim_id",
        "manuscript_location",
        "claim_summary",
        "reported_values",
        "artifact_path",
        "artifact_locator",
        "artifact_sha256",
        "expected_artifact_tokens",
        "analysis_script",
        "script_sha256",
        "denominator",
        "statistical_unit",
        "claim_boundary",
    ]
    with CLAIM_MAP.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise AssertionError("Unexpected claim-artifact map schema")
        rows = list(reader)
    if len(rows) < 20:
        raise AssertionError("Claim-artifact map is unexpectedly short")
    claim_ids = [row["claim_id"] for row in rows]
    if any(not claim_id for claim_id in claim_ids) or len(claim_ids) != len(set(claim_ids)):
        raise AssertionError("Claim-artifact identifiers must be nonempty and unique")

    checked_artifacts: set[str] = set()
    checked_scripts: set[str] = set()
    for row in rows:
        for field in ("artifact_path", "analysis_script"):
            relative = row[field]
            if not relative:
                if field == "analysis_script":
                    continue
                raise AssertionError(f"Missing artifact path: {row['claim_id']}")
            if "\\" in relative or relative.startswith("/") or ".." in relative.split("/"):
                raise AssertionError(
                    f"Claim-artifact paths must be safe POSIX-relative paths: {relative}"
                )

        artifact = ROOT / row["artifact_path"]
        if not artifact.is_file():
            raise AssertionError(f"Claim artifact is missing: {row['artifact_path']}")
        if file_sha256(artifact) != row["artifact_sha256"]:
            raise AssertionError(f"Claim artifact changed: {row['artifact_path']}")
        artifact_text = artifact.read_text(encoding="utf-8-sig")
        for token in filter(None, row["expected_artifact_tokens"].split("|")):
            if token not in artifact_text:
                raise AssertionError(
                    f"Claim token is missing for {row['claim_id']}: {token}"
                )
        checked_artifacts.add(row["artifact_path"])

        if row["analysis_script"]:
            script = ROOT / row["analysis_script"]
            if not script.is_file():
                raise AssertionError(f"Claim analysis script is missing: {row['analysis_script']}")
            if file_sha256(script) != row["script_sha256"]:
                raise AssertionError(f"Claim analysis script changed: {row['analysis_script']}")
            checked_scripts.add(row["analysis_script"])

    return {
        "claims": len(rows),
        "artifacts": len(checked_artifacts),
        "analysis_scripts": len(checked_scripts),
    }


def verify_common_score_parity() -> dict[str, Any]:
    protocol = json.loads((COMMON_SCORE / "PROTOCOL.json").read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "PASS"
        or protocol.get("score_training_excludes_target_comparators") is not True
        or protocol.get("main_factorial_max_abs_error", 1.0) > 1e-12
        or protocol.get("target_native_max_abs_error", 1.0) > 1e-12
    ):
        raise AssertionError("Common-score parity protocol did not pass")
    for relative, expected_hash in protocol.get("inputs", {}).items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise AssertionError(f"Common-score parity input changed: {relative}")

    expected = {
        "registered_lineage_pooling": [
            0.8125327373072897,
            0.8125327373072897,
            0.8064734907462827,
            0.8256215582470559,
            0.37106748832286374,
        ],
        "hierarchical_cautious_cumulative": [
            0.8597383184173139,
            0.7869593172175243,
            0.7839829921849003,
            0.7943467549067271,
            0.46589201848292194,
        ],
    }
    with (COMMON_SCORE / "table_panel_b_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["method"] for row in rows} != set(expected) or len(rows) != len(expected):
        raise AssertionError("Unexpected common-score comparator rows")
    fields = ("native", "peak", "margin", "entropy", "shared")
    for row in rows:
        if any(
            not math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-12)
            for field, value in zip(fields, expected[row["method"]])
        ):
            raise AssertionError(f"Common-score values changed: {row['method']}")

    hashes = json.loads((COMMON_SCORE / "sha256.json").read_text(encoding="utf-8"))
    for name, expected_hash in hashes.items():
        if file_sha256(COMMON_SCORE / name) != expected_hash:
            raise AssertionError(f"Common-score output changed: {name}")
    return {
        "status": protocol["status"],
        "comparators": sorted(expected),
        "main_factorial_max_abs_error": protocol["main_factorial_max_abs_error"],
        "target_native_max_abs_error": protocol["target_native_max_abs_error"],
    }


def verify_public_outcome_snapshot() -> dict[str, Any]:
    audit = json.loads((PUBLIC_OUTCOME / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS" or not all(audit.get("checks", {}).values()):
        raise AssertionError("Public-outcome audit did not pass")

    invariance = json.loads(
        (PUBLIC_OUTCOME / "INVARIANCE_GATE.json").read_text(encoding="utf-8")
    )
    if set(invariance) != {"HandWritten-Mfeat__TMC", "HandWritten-Mfeat__RCML"}:
        raise AssertionError("Unexpected public-outcome emitter pairs")
    if not all(result.get("pass") for result in invariance.values()):
        raise AssertionError("Registered-copy invariance failed")

    support = json.loads((PUBLIC_OUTCOME / "PAIR_SUPPORT.json").read_text(encoding="utf-8"))
    if any(
        result.get("gamma_min") != 0.1
        or result.get("gamma_max") != 0.9
        or result.get("selection_uses_labels") is not False
        for result in support.values()
    ):
        raise AssertionError("Unexpected public-outcome support")

    with (PUBLIC_OUTCOME / "PAIRED_CONTRASTS.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    primary = [
        row
        for row in rows
        if row["contrast"] == "provenance_error"
        and row["metric"] == "ncsAURC"
        and row["aggregation"] == "donor_macro"
    ]
    if len(primary) != 2 or any(float(row["ci_low"]) <= 0 for row in primary):
        raise AssertionError("Public-outcome primary contrast is incomplete")
    return {
        "pairs": len(primary),
        "bootstrap_draws": sorted({int(row["draws"]) for row in primary}),
    }


def verify_native_view_snapshot() -> dict[str, Any]:
    audit = json.loads(
        (NATIVE_VIEW / "gates" / "FINAL_RESULT_AUDIT.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "FROZEN_RESULTS_COMPLETE_WITH_LOCKED_TIE_CORRECTION":
        raise AssertionError("Native-view corrected result audit is not complete")
    verification = audit.get("verification", {})
    required_checks = (
        "all_ids_unique",
        "all_probabilities_valid",
        "all_raw_rows_complete",
        "all_status_ok",
        "all_parent_records_traceable",
        "semantic_tie_checks_pass",
        "independent_primary_recompute_pass",
    )
    if not all(verification.get(name) is True for name in required_checks):
        raise AssertionError("Native-view corrected result checks did not all pass")

    prompt_pack = NATIVE_VIEW / "protocol" / "test" / "prompt_pack.jsonl.gz"
    models = {
        "primary_32b": ("analysis_test32", "qwen3vl_32b.jsonl.gz"),
        "replication_8b": ("analysis_test8", "qwen3vl_8b.jsonl.gz"),
    }
    contrasts: dict[str, dict[str, list[float]]] = {}
    for audit_key, (analysis_dir, output_name) in models.items():
        result = json.loads(
            (NATIVE_VIEW / analysis_dir / "PRIMARY_RESULT.json").read_text(encoding="utf-8")
        )
        if (
            result.get("status") != "COMPLETE_WITH_LOCKED_TIE_CORRECTION"
            or result.get("episodes") != 696
            or result.get("tasks") != 6
            or result.get("units") != 2256
        ):
            raise AssertionError(f"Unexpected native-view result dimensions: {analysis_dir}")
        primary = result.get("primary_contrasts", [])
        common = result.get("common_score_contrasts", [])
        if [row.get("m") for row in primary] != [1, 2, 4] or [row.get("m") for row in common] != [1, 2, 4]:
            raise AssertionError(f"Unexpected multiplicity series: {analysis_dir}")
        if primary[0].get("estimate") != 0.0 or common[0].get("estimate") != 0.0:
            raise AssertionError(f"Native-view m=1 identity failed: {analysis_dir}")
        if uncompressed_sha256(NATIVE_VIEW / "outputs" / output_name) != result["hashes"]["outputs"]:
            raise AssertionError(f"Native-view output content hash mismatch: {output_name}")
        if uncompressed_sha256(prompt_pack) != result["hashes"]["prompt_pack"]:
            raise AssertionError("Native-view test prompt-pack content hash mismatch")
        expected_hashes = audit["models"][audit_key]["analysis_artifact_sha256"]
        for name, expected in expected_hashes.items():
            if file_sha256(NATIVE_VIEW / analysis_dir / name) != expected:
                raise AssertionError(f"Native-view analysis hash mismatch: {analysis_dir}/{name}")
        contrasts[analysis_dir] = {
            "native": [float(row["estimate"]) for row in primary],
            "posterior_confidence": [float(row["estimate"]) for row in common],
        }
    return {
        "episodes": 696,
        "events": 1128,
        "tasks": 6,
        "multiplicities": [1, 2, 4],
        "contrasts": contrasts,
    }


def _read_scalability_csv(path: Path) -> list[dict[str, Any]]:
    fields = [
        "source_count",
        "method",
        "median_us",
        "p95_us",
        "throughput_calls_s",
        "throughput_repeat_sd",
        "repeat_median_sd_us",
        "peak_traced_allocation_kib",
        "timed_calls",
    ]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise AssertionError(f"Unexpected scalability CSV schema: {path.name}")
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if list(raw) != fields or any(value is None for value in raw.values()):
            raise AssertionError(f"Malformed scalability CSV row: {path.name}")
        row = {
            "source_count": int(raw["source_count"]),
            "method": raw["method"],
            "median_us": float(raw["median_us"]),
            "p95_us": float(raw["p95_us"]),
            "throughput_calls_s": float(raw["throughput_calls_s"]),
            "throughput_repeat_sd": float(raw["throughput_repeat_sd"]),
            "repeat_median_sd_us": float(raw["repeat_median_sd_us"]),
            "peak_traced_allocation_kib": float(raw["peak_traced_allocation_kib"]),
            "timed_calls": int(raw["timed_calls"]),
        }
        numeric = [value for key, value in row.items() if key != "method"]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise AssertionError(f"Non-finite scalability value: {path.name}")
        if row["source_count"] <= 0 or row["timed_calls"] <= 0:
            raise AssertionError(f"Non-positive scalability count: {path.name}")
        if row["median_us"] < 0 or row["p95_us"] < row["median_us"]:
            raise AssertionError(f"Invalid scalability latency order: {path.name}")
        if row["throughput_calls_s"] <= 0 or row["peak_traced_allocation_kib"] < 0:
            raise AssertionError(f"Invalid scalability resource value: {path.name}")
        rows.append(row)
    return rows


def _assert_scalability_rows_equal(
    csv_rows: list[dict[str, Any]],
    summary_rows: Any,
    label: str,
) -> None:
    if not isinstance(summary_rows, list) or len(csv_rows) != len(summary_rows):
        raise AssertionError(f"Scalability JSON/CSV row-count mismatch: {label}")
    expected_fields = set(csv_rows[0]) if csv_rows else set()
    for index, (csv_row, summary_row) in enumerate(zip(csv_rows, summary_rows)):
        if not isinstance(summary_row, dict) or set(summary_row) != expected_fields:
            raise AssertionError(f"Unexpected scalability JSON row schema: {label}[{index}]")
        for field, csv_value in csv_row.items():
            summary_value = summary_row[field]
            if field == "method":
                equal = summary_value == csv_value
            elif field in {"source_count", "timed_calls"}:
                equal = isinstance(summary_value, int) and summary_value == csv_value
            else:
                equal = isinstance(summary_value, (int, float)) and math.isclose(
                    float(summary_value), float(csv_value), rel_tol=0.0, abs_tol=1e-12
                )
            if not equal:
                raise AssertionError(
                    f"Scalability JSON/CSV mismatch: {label}[{index}].{field}"
                )


def verify_scalability_outputs() -> dict[str, Any]:
    summary_path = SCALABILITY / "summary.json"
    canonical_path = SCALABILITY / "canonical_path.csv"
    scaling_path = SCALABILITY / "source_count_scaling.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary) != {
        "protocol",
        "environment",
        "source_count_scaling",
        "canonical_path",
        "provenance",
    }:
        raise AssertionError("Unexpected scalability summary schema")

    protocol = summary["protocol"]
    expected_protocol = {
        "clock": "time.perf_counter_ns",
        "warmup_calls": 200,
        "repeat_blocks": 5,
        "timed_calls_per_block": 1000,
        "memory_metric": "incremental peak traced Python allocation",
        "memory_calls": 100,
        "throughput_metric": "median over dedicated repeat blocks",
        "source_counts": [3, 8, 16, 32, 64],
        "registered_graph_topology": "sparse disjoint pairs",
        "execution": "single Python process; CPU affinity not set",
    }
    if protocol != expected_protocol:
        raise AssertionError("Unexpected scalability protocol")

    environment = summary["environment"]
    if set(environment) != {"platform", "python", "processor", "logical_cpu_count"}:
        raise AssertionError("Unexpected scalability environment schema")
    if not all(environment.get(key) for key in ("platform", "python", "processor")):
        raise AssertionError("Incomplete scalability environment identity")
    if not isinstance(environment.get("logical_cpu_count"), int) or environment["logical_cpu_count"] <= 0:
        raise AssertionError("Invalid scalability CPU count")

    provenance = summary["provenance"]
    if set(provenance) != {"script_sha256", "output_sha256"}:
        raise AssertionError("Unexpected scalability provenance schema")
    if set(provenance["output_sha256"]) != {
        "canonical_path.csv",
        "source_count_scaling.csv",
    }:
        raise AssertionError("Unexpected scalability output-hash schema")
    benchmark = ROOT / "experiments" / "scalability_benchmark.py"
    if provenance["script_sha256"] != file_sha256(benchmark):
        raise AssertionError("Scalability script checksum mismatch")
    for name, path in {
        "canonical_path.csv": canonical_path,
        "source_count_scaling.csv": scaling_path,
    }.items():
        if provenance["output_sha256"][name] != file_sha256(path):
            raise AssertionError(f"Scalability output checksum mismatch: {name}")

    canonical_rows = _read_scalability_csv(canonical_path)
    scaling_rows = _read_scalability_csv(scaling_path)
    if len(canonical_rows) != 5 or len(scaling_rows) != 15:
        raise AssertionError("Unexpected scalability output row count")
    _assert_scalability_rows_equal(canonical_rows, summary["canonical_path"], "canonical_path")
    _assert_scalability_rows_equal(
        scaling_rows, summary["source_count_scaling"], "source_count_scaling"
    )

    source_counts = {int(row["source_count"]) for row in scaling_rows}
    if source_counts != {3, 8, 16, 32, 64}:
        raise AssertionError("Unexpected source-count sweep")
    scaling_methods = {
        "quality_weighted_vote",
        "evidential_composition",
        "registered_graph_components",
    }
    observed_pairs = {(row["source_count"], row["method"]) for row in scaling_rows}
    expected_pairs = {(count, method) for count in source_counts for method in scaling_methods}
    if observed_pairs != expected_pairs:
        raise AssertionError("Incomplete source-count-by-method scalability grid")
    if {row["method"] for row in canonical_rows} != {
        "quality_weighted_vote",
        "evidential_composition",
        "registered_lineage_analysis",
        "shared_verifier",
        "typed_record_end_to_end",
    }:
        raise AssertionError("Unexpected canonical scalability methods")
    if {row["source_count"] for row in canonical_rows} != {3}:
        raise AssertionError("Canonical path must use exactly three sources")

    return {
        "canonical_rows": len(canonical_rows),
        "source_count_rows": len(scaling_rows),
        "script_sha256": provenance["script_sha256"],
        "output_sha256": provenance["output_sha256"],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "release_verification.json",
    )
    arguments = parser.parse_args()

    manifest = verify_release_manifest()
    claim_artifact_map = verify_claim_artifact_map()
    common_score = verify_common_score_parity()
    public_outcome = verify_public_outcome_snapshot()
    native_view = verify_native_view_snapshot()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_hash = uncompressed_sha256(SOURCE)
    label_hash = uncompressed_sha256(LABELS)
    if source_hash != config["source_records_sha256"]:
        raise AssertionError("Source-record checksum mismatch")
    if label_hash != config["evaluation_labels_sha256"]:
        raise AssertionError("Evaluation-label checksum mismatch")

    records = read_gzip_jsonl(SOURCE)
    labels = read_gzip_jsonl(LABELS)
    record_ids = [str(row["record_id"]) for row in records]
    label_ids = [str(row["record_id"]) for row in labels]
    if len(records) != 31_200 or len(labels) != 31_200:
        raise AssertionError("Expected 31,200 source and evaluation rows")
    if len(set(record_ids)) != len(record_ids):
        raise AssertionError("Duplicate source-record identifier")
    if len(set(label_ids)) != len(label_ids):
        raise AssertionError("Duplicate evaluation-label identifier")
    if set(record_ids) != set(label_ids):
        raise AssertionError("Source and evaluation identifiers differ")

    source_keys = collect_keys(records)
    leaked = sorted(source_keys & FORBIDDEN_SOURCE_FIELDS)
    if leaked:
        raise AssertionError(f"Evaluation fields leaked into source records: {leaked}")

    scene_by_cluster: dict[str, set[str]] = {}
    scenes = set()
    for record in records:
        scene = str(record["metadata"]["scene_id"])
        cluster = str(record["record_id"]).split("__", 1)[0]
        scenes.add(scene)
        scene_by_cluster.setdefault(cluster, set()).add(scene)
    if len(scenes) != 48:
        raise AssertionError(f"Expected 48 scene units, found {len(scenes)}")
    crossing = [cluster for cluster, values in scene_by_cluster.items() if len(values) != 1]
    if crossing:
        raise AssertionError("A cluster crosses scene units")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="action-admission-release-") as temp:
        temp_root = Path(temp)
        controlled_output = temp_root / "controlled"
        nested_output = temp_root / "nested"
        pcecf_output = temp_root / "pcecf"
        habit_checkpoint_output = temp_root / "habit_checkpoint_admission"
        run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            env,
        )
        run([sys.executable, "analysis/check_reported_metrics.py"], env)
        run([sys.executable, "analysis/verify_secondary_results.py"], env)
        run([sys.executable, "analysis/verify_recent_mechanism_results.py"], env)
        run(
            [
                sys.executable,
                "analysis/habit_checkpoint_admission.py",
                "--output",
                str(habit_checkpoint_output),
            ],
            env,
        )
        run(
            [
                sys.executable,
                "results/h2o_stage3/verify_released_aggregates.py",
            ],
            env,
        )
        run(
            [
                sys.executable,
                "experiments/nested_selection.py",
                "--output",
                str(nested_output),
            ],
            env,
        )
        run(
            [
                sys.executable,
                "experiments/controlled_study.py",
                "--output",
                str(controlled_output),
            ],
            env,
        )
        run(
            [
                sys.executable,
                "experiments/pcecf_study.py",
                "--output",
                str(pcecf_output),
            ],
            env,
        )
        run([sys.executable, "analysis/verify_p0_estimand_closure.py"], env)
        matched = []
        for name in (
            "risk_coverage.csv",
            "aurc_summary.csv",
            "fixed_target_summary.csv",
        ):
            if not csv_matches(
                controlled_output / name,
                REFERENCE / name,
            ):
                raise AssertionError(f"Regenerated output differs: {name}")
            matched.append(name)
        nested = json.loads(
            (nested_output / "summary.json").read_text(encoding="utf-8")
        )
        pcecf_references = {
            "table1_no_verifier_summary.csv": "pcecf_fusion_only.csv",
            "table2_shared_verifier_summary.csv": "pcecf_shared_verifier.csv",
        }
        pcecf_matched = []
        for observed_name, reference_name in pcecf_references.items():
            if not csv_matches(
                pcecf_output / observed_name,
                REFERENCE / reference_name,
            ):
                raise AssertionError(
                    f"Regenerated PC-ECF output differs: {observed_name}"
                )
            pcecf_matched.append(reference_name)
        pcecf_summary = json.loads(
            (pcecf_output / "summary.json").read_text(encoding="utf-8")
        )
        if pcecf_summary.get("status") != "PASS":
            raise AssertionError("PC-ECF study validation did not pass")
        observed_deltas = json.loads(
            (pcecf_output / "paired_scene_bootstrap_deltas.json").read_text(encoding="utf-8")
        )
        reference_deltas = json.loads(
            (REFERENCE / "pcecf_paired_deltas.json").read_text(encoding="utf-8")
        )
        if observed_deltas != reference_deltas:
            raise AssertionError("Regenerated PC-ECF paired deltas differ")
        pcecf_matched.append("pcecf_paired_deltas.json")
        habit_checkpoint_matched = []
        for name in (
            "checkpoint_admission_summary.csv",
            "checkpoint_admission_by_task.csv",
            "checkpoint_admission_bootstrap.csv",
        ):
            if not csv_matches(habit_checkpoint_output / name, HABIT_CHECKPOINT / name):
                raise AssertionError(f"Regenerated HABIT checkpoint output differs: {name}")
            habit_checkpoint_matched.append(name)
        if json.loads((habit_checkpoint_output / "gate.json").read_text(encoding="utf-8")) != json.loads(
            (HABIT_CHECKPOINT / "gate.json").read_text(encoding="utf-8")
        ):
            raise AssertionError("Regenerated HABIT checkpoint gate differs")
        habit_checkpoint_matched.append("gate.json")
        scalability = verify_scalability_outputs()

    report = {
        "status": "pass",
        "release_manifest": manifest,
        "claim_artifact_map": claim_artifact_map,
        "common_score_parity": common_score,
        "public_outcome_snapshot": public_outcome,
        "native_view_snapshot": native_view,
        "source_records": len(records),
        "evaluation_labels": len(labels),
        "scene_units": len(scenes),
        "clusters": len(scene_by_cluster),
        "source_records_sha256": source_hash,
        "evaluation_labels_sha256": label_hash,
        "selected_concentrations": nested["selected_concentrations"],
        "tests": "pass",
        "reported_metric_check": "pass",
        "secondary_result_check": "pass",
        "recent_mechanism_result_check": "pass",
        "h2o_aggregate_check": "pass",
        "reference_outputs_within_1e-12": matched,
        "pcecf_reference_outputs_within_1e-12": pcecf_matched,
        "pcecf_validation_status": pcecf_summary["status"],
        "habit_checkpoint_outputs_within_1e-12": habit_checkpoint_matched,
        "scalability_outputs": scalability,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print("Release verification completed successfully.")


if __name__ == "__main__":
    main()
