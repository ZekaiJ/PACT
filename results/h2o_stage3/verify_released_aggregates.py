#!/usr/bin/env python3
"""Verify the integrity and internal consistency of released H2O aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
EXPECTED_FILES = {
    "h2o_egocentric_allocentric_summary.csv",
    "h2o_unified_quality_shift_gate.json",
    "h2o_unified_quality_shift_summary.csv",
    "h2o_view_removal_monotonicity.csv",
    "stage3_gate.json",
    "stage3_integrity_audit.json",
    "stage3_threshold_sensitivity.csv",
    "stage3_threshold_sensitivity_gate.json",
}
EXPECTED_SETTINGS = {"ego_plus_world", "ego_only", "world_only", "neither"}
EXPECTED_REMOVALS = {"ego_only", "world_only", "neither"}
EXPECTED_METHODS = {"raw_blind", "raw_registered", "pact_sv"}
EXPECTED_VARIANTS = {
    "baseline",
    "missing_depth",
    "missing_object_pose",
    "depth_quality_shift",
    "object_pose_noise",
    "stale_world_geometry",
    "high_source_conflict",
    "low_quality_all",
}


class VerificationError(RuntimeError):
    """Raised when a released aggregate fails verification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_csv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == expected_header, f"unexpected header: {path.name}")
        return list(reader)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def as_int(value: str | int, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"invalid integer for {field}: {value!r}") from exc


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest = BASE / "stage3_hash_manifest.csv"
    rows = read_csv(manifest, ("path", "sha256", "bytes"))
    paths = [row["path"] for row in rows]
    require(len(paths) == len(set(paths)), "duplicate path in stage3_hash_manifest.csv")
    require(set(paths) == EXPECTED_FILES, "stage3_hash_manifest.csv has an unexpected file set")

    for row in rows:
        path = BASE / row["path"]
        require(path.is_file(), f"missing released aggregate: {row['path']}")
        require(path.stat().st_size == as_int(row["bytes"], f"{row['path']}.bytes"),
                f"byte count mismatch: {row['path']}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch: {row['path']}")
    return len(rows)


def verify_snapshot() -> dict[str, int | str]:
    manifest_files = verify_manifest()

    stage_gate = read_json(BASE / "stage3_gate.json")
    integrity = read_json(BASE / "stage3_integrity_audit.json")
    threshold_gate = read_json(BASE / "stage3_threshold_sensitivity_gate.json")
    quality_gate = read_json(BASE / "h2o_unified_quality_shift_gate.json")
    for name, gate in (
        ("stage3_gate.json", stage_gate),
        ("stage3_integrity_audit.json", integrity),
        ("stage3_threshold_sensitivity_gate.json", threshold_gate),
        ("h2o_unified_quality_shift_gate.json", quality_gate),
    ):
        require(gate.get("status") == "pass", f"non-pass status: {name}")

    denominator = stage_gate.get("denominator")
    require(
        denominator == {"h2o_records": 60, "commands": 5, "settings": 4, "decisions": 1200},
        "unexpected Stage 3 denominator",
    )

    summary = read_csv(
        BASE / "h2o_egocentric_allocentric_summary.csv",
        (
            "setting", "setting_label", "n", "raw_blind_admission_rate",
            "raw_registered_admission_rate", "pact_sv_admission_rate",
            "pact_sv_release_authority_rate", "pact_sv_admit_rate",
            "pact_sv_hold_rate", "pact_sv_confirm_rate", "pact_sv_retreat_rate",
            "pact_sv_fallback_rate", "mean_registered_confidence",
        ),
    )
    require({row["setting"] for row in summary} == EXPECTED_SETTINGS, "unexpected H2O settings")
    require(len(summary) == 4, "expected one row per H2O setting")
    require(sum(as_int(row["n"], "summary.n") for row in summary) == denominator["decisions"],
            "summary denominator mismatch")

    removal = read_csv(
        BASE / "h2o_view_removal_monotonicity.csv",
        (
            "method", "removed_to_setting", "comparisons",
            "admission_nonincrease_violations", "release_authority_nonincrease_violations",
        ),
    )
    observed_pairs = {(row["method"], row["removed_to_setting"]) for row in removal}
    expected_pairs = {(method, setting) for method in EXPECTED_METHODS for setting in EXPECTED_REMOVALS}
    require(observed_pairs == expected_pairs and len(removal) == 9, "unexpected view-removal rows")
    pact_rows = [row for row in removal if row["method"] == "pact_sv"]
    require(all(as_int(row["admission_nonincrease_violations"], "removal.pact_admission") == 0
                and as_int(row["release_authority_nonincrease_violations"], "removal.pact_authority") == 0
                for row in pact_rows), "PACT view-removal violation is nonzero")
    raw_registered = [row for row in removal if row["method"] == "raw_registered"]
    require(sum(as_int(row["admission_nonincrease_violations"], "removal.raw_registered")
                for row in raw_registered) == stage_gate["raw_registered_admission_nonincrease_violations"],
            "Stage 3 raw-registered violation count mismatch")
    require(sum(as_int(row["admission_nonincrease_violations"], "removal.pact")
                for row in pact_rows) == stage_gate["pact_sv_admission_nonincrease_violations"],
            "Stage 3 PACT violation count mismatch")

    thresholds = read_csv(
        BASE / "stage3_threshold_sensitivity.csv",
        (
            "local_low_m", "local_high_m", "world_low_mps", "world_high_mps",
            "ego_plus_world_pact_sv_admission", "ego_plus_world_release_authority",
            "ego_only_raw_registered_admission", "world_only_raw_registered_admission",
            "neither_raw_registered_admission",
            "raw_registered_admission_nonincrease_violations",
            "pact_sv_admission_nonincrease_violations",
        ),
    )
    require(len(thresholds) == as_int(threshold_gate["configurations"], "threshold.configurations"),
            "threshold configuration count mismatch")
    pact_range = [as_int(row["pact_sv_admission_nonincrease_violations"], "threshold.pact")
                  for row in thresholds]
    raw_range = [as_int(row["raw_registered_admission_nonincrease_violations"], "threshold.raw")
                 for row in thresholds]
    admission_range = [float(row["ego_plus_world_pact_sv_admission"]) for row in thresholds]
    require([min(pact_range), max(pact_range)] == threshold_gate["pact_sv_violation_range"],
            "threshold PACT violation range mismatch")
    require([min(raw_range), max(raw_range)] == threshold_gate["raw_registered_violation_range"],
            "threshold raw violation range mismatch")
    released_admission_range = threshold_gate["pact_sv_full_evidence_admission_range"]
    require(close(min(admission_range), released_admission_range[0])
            and close(max(admission_range), released_admission_range[1]),
            "threshold admission range mismatch")

    quality = read_csv(
        BASE / "h2o_unified_quality_shift_summary.csv",
        (
            "variant", "n", "raw_admission_rate", "pact_sv_admission_rate",
            "raw_admission_nonincrease_violations",
            "raw_release_authority_nonincrease_violations",
            "pact_sv_admission_nonincrease_violations",
            "pact_sv_release_authority_nonincrease_violations", "pact_sv_admit",
            "pact_sv_hold", "pact_sv_confirm", "pact_sv_retreat", "pact_sv_fallback",
        ),
    )
    require({row["variant"] for row in quality} == EXPECTED_VARIANTS and len(quality) == 8,
            "unexpected quality-shift variants")
    baseline = next(row for row in quality if row["variant"] == "baseline")
    stress = [row for row in quality if row["variant"] != "baseline"]
    require(sum(as_int(row["n"], "quality.n") for row in quality) == quality_gate["decisions"],
            "quality-shift decision count mismatch")
    require(as_int(baseline["n"], "quality.baseline.n") == quality_gate["baseline_decisions"],
            "quality-shift baseline count mismatch")
    require(sum(as_int(row["n"], "quality.stress.n") for row in stress) == quality_gate["stress_decisions"],
            "quality-shift stress count mismatch")
    for field in (
        "raw_admission_nonincrease_violations",
        "raw_release_authority_nonincrease_violations",
        "pact_sv_admission_nonincrease_violations",
        "pact_sv_release_authority_nonincrease_violations",
    ):
        require(sum(as_int(row[field], f"quality.{field}") for row in quality) == quality_gate[field],
                f"quality-shift gate mismatch: {field}")

    checks = integrity.get("checks")
    require(isinstance(checks, list) and checks, "integrity check list is missing")
    require(all(check.get("pass") is True for check in checks), "integrity audit contains a failed check")
    require(integrity.get("passed") == integrity.get("total") == len(checks),
            "integrity audit count mismatch")

    return {
        "status": "pass",
        "manifest_files": manifest_files,
        "settings": len(summary),
        "view_removal_rows": len(removal),
        "threshold_configurations": len(thresholds),
        "quality_variants": len(quality),
        "integrity_checks": len(checks),
    }


def main() -> int:
    try:
        result = verify_snapshot()
    except (OSError, ValueError, KeyError, TypeError, VerificationError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
