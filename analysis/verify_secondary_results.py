"""Verify the released policy-factorial and 2x2 attribution snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTION = ROOT / "results" / "minimal_attribution_2x2"
FACTORIAL = ROOT / "results" / "score_verifier_factorial"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def verify_manifest(directory: Path, name: str) -> int:
    manifest = json.loads((directory / name).read_text(encoding="utf-8"))
    for filename, metadata in manifest.items():
        path = directory / filename
        if not path.is_file():
            raise AssertionError(f"missing released file: {path}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise AssertionError(f"size mismatch: {path}")
        if sha256(path) != metadata["sha256"]:
            raise AssertionError(f"checksum mismatch: {path}")
    return len(manifest)


def verify_attribution() -> dict[str, Any]:
    files = verify_manifest(ATTRIBUTION, "OUTPUT_MANIFEST.json")
    inputs = json.loads((ATTRIBUTION / "INPUT_HASHES.json").read_text(encoding="utf-8"))
    for metadata in inputs.values():
        path = ROOT / metadata["path"]
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise AssertionError(f"attribution input mismatch: {metadata['path']}")

    protocol = json.loads((ATTRIBUTION / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if protocol["directional_acceptance_rule"] is not None:
        raise AssertionError("2x2 analysis must retain results regardless of direction")
    expected_partition = (
        "current registered partition in both fusion arms; "
        "F changes only operator aggregation"
    )
    if protocol["verifier_partition_fixed"] != expected_partition:
        raise AssertionError("verifier partition is not fixed across fusion arms")

    cells = {
        (row["stratum"], row["support"], row["cell"]): row
        for row in read_csv(ATTRIBUTION / "ncsaurc_cells.csv")
    }
    key = ("all_records", "primary_0.10_0.39")
    expected = {
        "F0_singleton:V0_no_provenance": 0.41346387685055297,
        "F0_singleton:V1_full": 0.1521145648367512,
        "F1_registered:V0_no_provenance": 0.3891164502870042,
        "F1_registered:V1_full": 0.08612194505919865,
    }
    for cell, value in expected.items():
        if not close(float(cells[(*key, cell)]["ncsAURC"]), value):
            raise AssertionError(f"2x2 cell changed: {cell}")

    effects = {
        row["contrast"]: row
        for row in read_csv(ATTRIBUTION / "ncsaurc_factorial_effects.csv")
        if row["stratum"] == key[0] and row["support"] == key[1]
    }
    bootstrap = {
        cell: float(cells[(*key, cell)]["bootstrap_mean"])
        for cell in expected
    }
    derived = {
        "fusion_effect_at_V0": (
            bootstrap["F1_registered:V0_no_provenance"]
            - bootstrap["F0_singleton:V0_no_provenance"]
        ),
        "fusion_effect_at_V1": (
            bootstrap["F1_registered:V1_full"]
            - bootstrap["F0_singleton:V1_full"]
        ),
        "verifier_effect_at_F0": (
            bootstrap["F0_singleton:V1_full"]
            - bootstrap["F0_singleton:V0_no_provenance"]
        ),
        "verifier_effect_at_F1": (
            bootstrap["F1_registered:V1_full"]
            - bootstrap["F1_registered:V0_no_provenance"]
        ),
    }
    derived["difference_in_differences"] = (
        derived["fusion_effect_at_V1"] - derived["fusion_effect_at_V0"]
    )
    for contrast, value in derived.items():
        if not close(float(effects[contrast]["mean"]), value):
            raise AssertionError(f"2x2 contrast arithmetic changed: {contrast}")

    transitions = read_csv(ATTRIBUTION / "route_transitions.csv")
    rerouted = {
        row["fusion"]: int(row["records"])
        for row in transitions
        if row["stratum"] == "all_records"
        and row["from_V0"] == "admit"
        and row["to_V1"] == "confirm"
    }
    if rerouted != {"F0_singleton": 2400, "F1_registered": 2400}:
        raise AssertionError("unexpected admit-to-confirm transition count")

    anchor = json.loads(
        (ATTRIBUTION / "CANONICAL_ANCHOR_CHECK.json").read_text(encoding="utf-8")
    )
    if anchor.get("pass") is not True:
        raise AssertionError("registered PACT anchor failed")
    return {"files": files, "cells": len(cells), "rerouted": rerouted}


def verify_factorial() -> dict[str, Any]:
    hashes = json.loads((FACTORIAL / "sha256.json").read_text(encoding="utf-8"))
    for filename, digest in hashes.items():
        path = FACTORIAL / filename
        if not path.is_file() or sha256(path) != digest:
            raise AssertionError(f"policy-factorial checksum mismatch: {filename}")

    summary = json.loads((FACTORIAL / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise AssertionError("policy-factorial snapshot did not pass")

    policies = {
        (row["method"], row["score"]): float(row["csaurc_0p10_0p39"])
        for row in read_csv(FACTORIAL / "verifier_policy_summary.csv")
    }
    expected = {
        ("nested_evidential_composition", "A_unanimous_three_source"): 0.1479,
        ("nested_evidential_composition", "B_multi_source_three_component"): 0.1490,
        ("nested_evidential_composition", "C_required_role_two_component"): 0.4116,
        ("pcecf", "A_unanimous_three_source"): 0.0861,
        ("pcecf", "B_multi_source_three_component"): 0.0868,
        ("pcecf", "C_required_role_two_component"): 0.3891,
    }
    for key, value in expected.items():
        if not close(policies[key], value, tolerance=5e-5):
            raise AssertionError(f"policy result changed: {key}")

    transitions = read_csv(FACTORIAL / "verifier_policy_transitions.csv")
    lookup = {
        (
            row["method"],
            row["alternative_policy"],
            row["current_pass"],
            row["alternative_pass"],
            row["candidate_correct"],
        ): int(row["records"])
        for row in transitions
    }
    if lookup[
        (
            "pcecf",
            "B_multi_source_three_component",
            "True",
            "False",
            "True",
        )
    ] != 18:
        raise AssertionError("policy B transition count changed")
    if lookup[
        (
            "pcecf",
            "C_required_role_two_component",
            "False",
            "True",
            "False",
        )
    ] != 2400:
        raise AssertionError("policy C transition count changed")
    return {"files": len(hashes), "policies": len(policies)}


def main() -> None:
    report = {
        "status": "pass",
        "minimal_attribution_2x2": verify_attribution(),
        "score_verifier_factorial": verify_factorial(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
