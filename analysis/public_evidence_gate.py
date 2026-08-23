#!/usr/bin/env python3
"""Emit the twelve pairwise public-benchmark eligibility verdicts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("HandWritten-Mfeat", "PIE", "Scene15", "CUB", "Caltech101", "HMDB")
EMITTERS = ("TMC", "RCML")
EXPECTED_HASHES = {
    "tmc_handwritten": "d731ec02aaa97cf9a5826df68b0dd1e73ba61a82ba3c8f64606ba402926c885b",
    "rcml_handwritten": "8a9ad81f11c05fd7f07f1369bcbbaefc3d391bad10d3aa32366bd47d608eb05d",
    "rcml_pie": "7b7f5b73b140c9e00d6d9a4ed78e0292b298bd12fbcfb2c96125795c7f8150f8",
    "rcml_scene15": "52a71c6c675d5b0c1e07189778716bc52b972b17b58d8156c01422bcce2e4442",
}
EXPECTED_COMMITS = {
    "TMC": "a3272b8746861c76a3461943b5eee51df5b5a8fe",
    "RCML": "c9c5ab41e6fe62a85e5f6441a4dc7b568e1fa421",
}


def export_ready(inventory: dict, backbone: str, views: int, classes: int) -> bool:
    rows = [row for row in inventory["exports"] if row["backbone"] == backbone]
    seeds = {int(row["seed"]) for row in rows if row["views"] == views and row["classes"] == classes}
    return len(seeds) >= 5 and all(row["rows"] > 0 for row in rows if int(row["seed"]) in seeds)


def verdict(dataset: str, emitter: str, inventory: dict) -> dict:
    common = {
        "dataset": dataset,
        "emitter": emitter,
        "scope": "eligibility only; no scientific claim",
    }
    if inventory["commits"].get(emitter) != EXPECTED_COMMITS[emitter]:
        return common | {"verdict": "MISMATCH", "blocking_field": "code_commit"}

    if dataset == "HandWritten-Mfeat":
        key = f"{emitter.lower()}_handwritten"
        asset = inventory["datasets"].get(key, {})
        if not asset.get("exists"):
            return common | {"verdict": "UNAVAILABLE", "blocking_field": "processed_payload"}
        if asset.get("sha256") != EXPECTED_HASHES[key]:
            return common | {"verdict": "MISMATCH", "blocking_field": "payload_sha256"}
        if not export_ready(inventory, emitter, 6, 10):
            return common | {"verdict": "MISMATCH", "blocking_field": "per_view_evidence_export"}
        reproduction = "within_2pp_gate" if emitter == "TMC" else "goalh_60_20_20_within_2pp"
        return common | {
            "verdict": "PASS",
            "blocking_field": None,
            "payload_sha256": asset["sha256"],
            "native_reproduction": reproduction,
            "views": 6,
            "classes": 10,
        }

    if dataset == "PIE" and emitter == "RCML":
        asset = inventory["datasets"].get("rcml_pie", {})
        if not asset.get("exists"):
            return common | {"verdict": "UNAVAILABLE", "blocking_field": "processed_payload"}
        if asset.get("sha256") != EXPECTED_HASHES["rcml_pie"]:
            return common | {"verdict": "MISMATCH", "blocking_field": "payload_sha256"}
        if not export_ready(inventory, "RCML-PIE", 3, 68):
            return common | {"verdict": "MISMATCH", "blocking_field": "per_view_evidence_export"}
        return common | {
            "verdict": "PASS",
            "blocking_field": None,
            "payload_sha256": asset["sha256"],
            "native_reproduction": "official_80_20_exact_published_0.9471",
            "views": 3,
            "classes": 68,
        }

    if dataset == "Scene15" and emitter == "RCML":
        asset = inventory["datasets"].get("rcml_scene15", {})
        return common | {
            "verdict": "MISMATCH",
            "blocking_field": "native_performance_reproduction",
            "payload_sha256": asset.get("sha256"),
            "observed_accuracy": 0.6972,
            "published_accuracy": 0.7619,
        }

    return common | {
        "verdict": "UNAVAILABLE",
        "blocking_field": "verified_processed_payload_and_per_view_export",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    results = [verdict(dataset, emitter, inventory) for dataset in DATASETS for emitter in EMITTERS]
    assert len(results) == 12 and {(r["dataset"], r["emitter"]) for r in results} == {
        (dataset, emitter) for dataset in DATASETS for emitter in EMITTERS
    }
    for row in results:
        name = f"{row['dataset']}__{row['emitter']}.json".replace("/", "-")
        (args.output / name).write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    counts = {status: sum(row["verdict"] == status for row in results) for status in ("PASS", "MISMATCH", "UNAVAILABLE")}
    lines = ["# B0 pairwise eligibility", "", f"PASS: {counts['PASS']}; MISMATCH: {counts['MISMATCH']}; UNAVAILABLE: {counts['UNAVAILABLE']}.", ""]
    lines += [f"- {row['dataset']} × {row['emitter']}: **{row['verdict']}** ({row.get('blocking_field') or 'eligible'})" for row in results]
    (args.output.parent / "B0_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
