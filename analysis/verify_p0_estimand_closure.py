"""Verify hashes, denominators, estimands, and paired contrast arithmetic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "p0_estimand_closure" / "v1"
P0_INPUTS = OUTPUT / "inputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(name: str) -> dict[str, Any]:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def verify_manifest() -> None:
    manifest = OUTPUT / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in expected:
            raise AssertionError(f"duplicate manifest entry: {name}")
        expected[name] = digest
    observed = {
        path.name
        for path in OUTPUT.iterdir()
        if path.is_file() and path.name != manifest.name
    }
    if observed != set(expected):
        raise AssertionError(
            f"manifest file set differs: missing={set(expected) - observed}, "
            f"extra={observed - set(expected)}"
        )
    for name, digest in expected.items():
        if sha256(OUTPUT / name) != digest:
            raise AssertionError(f"manifest hash mismatch: {name}")


def verify_input_hashes() -> None:
    protocol = load_json("PROTOCOL_LOCK.json")
    for relative, digest in protocol["input_sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"input is missing: {relative}")
        if sha256(path) != digest:
            raise AssertionError(f"input hash mismatch: {relative}")


def verify_controlled() -> None:
    gate = load_json("CONTROLLED_GATE.json")
    if gate["status"] != "EXECUTED_COMPLETE" or gate["fixed_support"] != [0.1, 0.35]:
        raise AssertionError("controlled gate or support is invalid")
    if gate["denominator"] != {"outer_folds": 5, "records": 31200, "scenes": 48}:
        raise AssertionError("controlled denominator changed")
    counts = read_csv("CONTROLLED_ACCEPTED_COUNTS.csv")
    if len(counts) != 52 or not all(row["all_methods_exact"] == "True" for row in counts):
        raise AssertionError("accepted-count check is incomplete")
    estimates = read_csv("CONTROLLED_ESTIMATES.csv")
    lookup = {
        (row["table"], row["estimator"], row["method"]): float(
            row["ncsAURC_point"]
        )
        for row in estimates
    }
    support = np.linspace(0.10, 0.35, 36)
    for table, source in (
        ("fusion_no_verifier", "table1_no_verifier_risk_coverage.csv"),
        ("shared_verifier", "table2_shared_verifier_risk_coverage.csv"),
    ):
        grouped: dict[str, list[dict[str, str]]] = {}
        with (P0_INPUTS / source).open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault(row["method"], []).append(row)
        for method, rows in grouped.items():
            by_coverage: dict[float, list[float]] = {}
            for row in rows:
                by_coverage.setdefault(float(row["coverage"]), []).append(
                    float(row["wrong_admitted"])
                )
            coverage = np.asarray(sorted(by_coverage))
            risk = np.asarray(
                [np.mean(by_coverage[value]) for value in coverage]
            )
            expected_point = float(
                np.trapezoid(np.interp(support, coverage, risk), support)
                / (support[-1] - support[0])
            )
            observed_point = lookup[(table, "outer_train_threshold", method)]
            if abs(expected_point - observed_point) > 1e-12:
                raise AssertionError(
                    f"controlled frozen-curve reproduction changed: {table}/{method}"
                )
    for row in read_csv("CONTROLLED_PAIRED_CONTRASTS.csv"):
        left = lookup[(row["table"], row["estimator"], row["left_method"])]
        right = lookup[(row["table"], row["estimator"], row["right_method"])]
        if abs((left - right) - float(row["point_difference"])) > 1e-12:
            raise AssertionError(f"controlled contrast arithmetic changed: {row['contrast']}")


def verify_repeated_splits() -> None:
    gate = load_json("REPEATED_SPLIT_GATE.json")
    if gate["status"] == "NOT_ELIGIBLE":
        if not gate.get("missing_inputs"):
            raise AssertionError("NOT_ELIGIBLE gate does not name missing inputs")
        return
    if gate["status"] != "EXECUTED_COMPLETE" or gate["repeats"] != 50:
        raise AssertionError("repeated-split gate is invalid")
    repeats = read_csv("REPEATED_SPLIT_ESTIMATES.csv")
    selections = read_csv("REPEATED_SPLIT_OUTER_SELECTION.csv")
    if len(repeats) != 50 or len(selections) != 250:
        raise AssertionError("repeated-split output denominator changed")
    if not gate["original_split_self_check"]["passed"]:
        raise AssertionError("original selector self-check failed")


def verify_balanced_panel() -> None:
    gate = load_json("BALANCED_PANEL_GATE.json")
    expected = {"checkpoints": 8, "episodes": 696, "events": 1128, "rows": 2256, "tasks": 6}
    if gate["status"] != "EXECUTED_COMPLETE" or gate["denominator"] != expected:
        raise AssertionError("balanced-panel denominator changed")
    if gate["estimands"]["registered"]["support"] != [0.1, 0.9]:
        raise AssertionError("registered balanced-panel support changed")
    if gate["estimands"]["controlled_comparability_sensitivity"]["support"] != [0.1, 0.39]:
        raise AssertionError("comparability support changed")
    if gate["full_support_release_reproduction_max_abs_residual"] > 1e-12:
        raise AssertionError("released balanced-panel points were not reproduced")
    estimates = read_csv("BALANCED_PANEL_ESTIMATES.csv")
    lookup = {
        (row["estimand"], row["method"]): float(row["ncsAURC_point"])
        for row in estimates
    }
    with (
        ROOT / "results" / "balanced_fm_panel" / "analysis" / "FUSION_METRICS.csv"
    ).open(encoding="utf-8", newline="") as handle:
        published = {
            row["method"]: float(row["ncsAURC"]) for row in csv.DictReader(handle)
        }
    for method, expected_point in published.items():
        observed_point = lookup[("registered_full_fractional_tie", method)]
        if abs(expected_point - observed_point) > 1e-12:
            raise AssertionError(
                f"balanced full-support release reproduction changed: {method}"
            )
    for row in read_csv("BALANCED_PANEL_PAIRED_CONTRASTS.csv"):
        left = lookup[(row["estimand"], row["left_method"])]
        right = lookup[(row["estimand"], row["right_method"])]
        if abs((left - right) - float(row["point_difference"])) > 1e-12:
            raise AssertionError(f"balanced contrast arithmetic changed: {row['contrast']}")


def main() -> None:
    final = load_json("FINAL_VERDICT.json")
    if final["status"] not in {
        "COMPLETE",
        "COMPLETE_WITH_REPEATED_SPLIT_NOT_ELIGIBLE",
    }:
        raise AssertionError("final verdict is incomplete")
    if not final["no_sign_filter"] or not final["no_fabricated_results"]:
        raise AssertionError("integrity flags are not set")
    verify_manifest()
    verify_input_hashes()
    verify_controlled()
    verify_repeated_splits()
    verify_balanced_panel()
    print("P0_ESTIMAND_CLOSURE_VERIFY_PASS")


if __name__ == "__main__":
    main()
