"""Verify the released topology, operator, and frozen-score analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DIRECTORIES = (
    RESULTS / "topology_multiplicity",
    RESULTS / "operator_characterization",
    RESULTS / "equal_cardinality_topology",
    RESULTS / "frozen_score_transport" / "full",
    RESULTS / "frozen_score_transport" / "no_count",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(observed: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, observed {observed}")


def verify_manifest(directory: Path) -> int:
    manifest = directory / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    observed = {
        path.name: sha256(path)
        for path in directory.iterdir()
        if path.is_file() and path.name != manifest.name
    }
    if observed != expected:
        raise AssertionError(f"manifest mismatch: {directory.relative_to(ROOT)}")
    return len(observed)


def contrast(path: Path, multiplicity: int, name: str, support: str | None = None) -> dict[str, str]:
    matches = [
        row
        for row in rows(path)
        if int(row["multiplicity"]) == multiplicity
        and row["contrast"] == name
        and (support is None or row.get("support") == support)
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing contrast {name} at m={multiplicity}: {path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifested = {str(path.relative_to(ROOT)): verify_manifest(path) for path in DIRECTORIES}

    topology = RESULTS / "topology_multiplicity"
    gate = json.loads((topology / "gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate["registered_ncsaurc_max_residual"] > 1e-12:
        raise AssertionError("topology multiplicity gate failed")
    expected_topology = {1: 0.02434742656354877, 8: 0.36761138670710997, 32: 0.4337567003573904}
    for multiplicity, expected in expected_topology.items():
        row = contrast(
            topology / "matched_contrast.csv",
            multiplicity,
            "F0_singleton_minus_F1_registered_at_V0",
            "primary_0.10_0.39",
        )
        close(float(row["point"]), expected)

    operator = RESULTS / "operator_characterization"
    structure = json.loads((operator / "structural_checks.json").read_text(encoding="utf-8"))
    close(float(structure["meet_common_evidence_cap_max_residual"]), 0.0)
    close(float(structure["join_exact_duplicate_max_residual"]), 0.0)
    close(float(structure["join_common_evidence_cap_violating_complete_record_fraction"]), 1.0)
    close(float(structure["join_same_component_insertion_increase_complete_record_fraction"]), 0.963625)
    operator_rows = {row["method"]: row for row in rows(operator / "operator_summary.csv")}
    close(float(operator_rows["pact_meet"]["ncsaurc_0p10_0p39"]), 0.6294374684440601)
    close(float(operator_rows["pact_join"]["ncsaurc_0p10_0p39"]), 0.7917538710731552)


    equal = RESULTS / "equal_cardinality_topology"
    equal_gate = json.loads((equal / "gate.json").read_text(encoding="utf-8"))
    if equal_gate.get("status") != "PASS" or equal_gate["formula_check"]["maximum_absolute_residual"] > 1e-12:
        raise AssertionError("equal-cardinality topology gate failed")
    equal_rows = {row["contrast"]: row for row in rows(equal / "paired_contrasts.csv")}
    expected_equal = {
        "wrong_LG_R_minus_registered_L_GR": 0.19880837906032967,
        "wrong_LR_G_minus_registered_L_GR": 0.20127650787455498,
    }
    for name, expected in expected_equal.items():
        close(float(equal_rows[name]["point"]), expected)
        close(float(equal_rows[name]["fraction_above_zero"]), 1.0)
    native_rows = {row["method"]: row for row in rows(equal / "native_random_references.csv")}
    pact_random = native_rows["pcecf"]
    close(float(pact_random["eligible_random_reference"]), 0.6169791666666666)
    close(float(pact_random["native_ncsaurc_0p10_0p39"]), 0.6294374684440601)
    close(float(pact_random["native_minus_random"]), 0.012458301777393488)

    score_results = {}
    for variant, anchor, crossover in (
        ("full", 0.6488988790621325, 2),
        ("no_count", 0.776297187923261, 4),
    ):
        directory = RESULTS / "frozen_score_transport" / variant
        summary = [row for row in rows(directory / "stress_summary.csv") if row["method"] == "pact_registered"]
        registered = [float(row["ncsaurc_0p10_0p39"]) for row in summary]
        if len(registered) != 6 or max(abs(value - anchor) for value in registered) > 1e-12:
            raise AssertionError(f"registered PACT changed in {variant} score transport")
        gap_rows = [
            row
            for row in rows(directory / "matched_contrast.csv")
            if row["contrast"] == "pact_registered_minus_nested_unaware"
        ]
        first_nonpositive = min(int(row["multiplicity"]) for row in gap_rows if float(row["point"]) <= 0.0)
        if first_nonpositive != crossover:
            raise AssertionError(f"unexpected {variant} score crossover: {first_nonpositive}")
        score_results[variant] = {"registered_anchor": anchor, "crossover_multiplicity": crossover}

    report = {
        "status": "pass",
        "manifested_files": manifested,
        "topology_contrast": expected_topology,
        "operator": {"meet_ncsaurc": 0.6294374684440601, "join_ncsaurc": 0.7917538710731552},
        "equal_cardinality_topology": {"contrasts": expected_equal, "pact_random_reference": 0.6169791666666666},
        "frozen_score_transport": score_results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
