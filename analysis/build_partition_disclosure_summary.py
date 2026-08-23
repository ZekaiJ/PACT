#!/usr/bin/env python3
"""Build the compact disclosure summary used by the coarsening claims."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "partition_coarsening_surface"
PROTOCOL = ROOT / "configs" / "partition_coarsening_protocol.json"
OUTPUT = RESULT_DIR / "DISCLOSURE_SUMMARY.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads((RESULT_DIR / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    metrics = read_csv(RESULT_DIR / "PARTITION_METRICS.csv")
    outcomes = read_csv(RESULT_DIR / "COVER_EDGE_OUTCOMES.csv")
    summary_rows = read_csv(RESULT_DIR / "COVER_EDGE_SUMMARY.csv")
    structural = json.loads((RESULT_DIR / "STRUCTURAL_GATE.json").read_text(encoding="utf-8"))

    pairs = tuple(manifest["pairs"])
    scores = tuple(manifest["scores"])
    expected_partitions = int(manifest["partitions"])
    expected_edges = int(manifest["cover_edges"])

    zero_floor_counts: dict[str, dict[str, int]] = {}
    direction_summary: dict[str, dict[str, dict[str, Any]]] = {}
    native_median_abs_delta: dict[str, dict[str, float]] = {}

    for pair_id in pairs:
        zero_floor_counts[pair_id] = {}
        direction_summary[pair_id] = {}
        for score in scores:
            metric_rows = [
                row for row in metrics
                if row["pair_id"] == pair_id and row["score"] == score
            ]
            if len(metric_rows) != expected_partitions:
                raise AssertionError(f"Unexpected partition count for {pair_id}/{score}")
            zero_floor_counts[pair_id][score] = sum(
                float(row["ncsAURC_0p10_0p90"]) == 0.0 for row in metric_rows
            )

            edge_rows = [
                row for row in outcomes
                if row["pair_id"] == pair_id and row["score"] == score
            ]
            if len(edge_rows) != expected_edges:
                raise AssertionError(f"Unexpected edge count for {pair_id}/{score}")
            counts = Counter(row["direction"] for row in edge_rows)
            estimates: dict[str, Any] = {}
            for direction in (
                "coarsening_improves",
                "coarsening_worsens",
                "tie",
            ):
                estimand = f"cover_edge_fraction_{direction}"
                matched = [
                    row for row in summary_rows
                    if row["pair_id"] == pair_id
                    and row["score"] == score
                    and row["estimand"] == estimand
                ]
                if len(matched) != 1:
                    raise AssertionError(f"Missing summary row for {pair_id}/{score}/{direction}")
                row = matched[0]
                estimates[direction] = {
                    "count": counts[direction],
                    "fraction": float(row["point"]),
                    "ci_95": [float(row["ci_low"]), float(row["ci_high"])],
                }
            direction_summary[pair_id][score] = estimates

        native_rows = [
            row for row in outcomes
            if row["pair_id"] == pair_id and row["score"] == "native_nonvacuity"
        ]
        native_median_abs_delta[pair_id] = {
            direction: statistics.median(
                abs(float(row["ncsAURC_delta"]))
                for row in native_rows
                if row["direction"] == direction
            )
            for direction in ("coarsening_improves", "coarsening_worsens")
        }

    pair_protocol = {
        row["pair_id"]: {
            "records_per_realization": int(row["records_per_seed"]),
            "frozen_emitter_realizations": int(row["frozen_emitter_realizations"]),
            "views": int(row["views"]),
        }
        for row in protocol["pairs"]
    }
    budget_pass = {
        row["pair_id"]: bool(row["budget_nonincrease_on_every_record_seed_edge"])
        for row in structural
    }
    if set(pair_protocol) != set(pairs) or set(budget_pass) != set(pairs):
        raise AssertionError("Pair identifiers do not agree across artifacts")
    if not all(budget_pass.values()):
        raise AssertionError("The budget nonincrease gate did not pass")

    source_paths = (
        "configs/partition_coarsening_protocol.json",
        "results/partition_coarsening_surface/RUN_MANIFEST.json",
        "results/partition_coarsening_surface/PARTITION_METRICS.csv",
        "results/partition_coarsening_surface/COVER_EDGE_OUTCOMES.csv",
        "results/partition_coarsening_surface/COVER_EDGE_SUMMARY.csv",
        "results/partition_coarsening_surface/STRUCTURAL_GATE.json",
    )
    payload = {
        "analysis": "partition coarsening disclosure summary",
        "budget_nonincrease_on_every_record_realization_edge": budget_pass,
        "cover_edge_directions": direction_summary,
        "native_non_tied_median_absolute_ncsAURC_delta": native_median_abs_delta,
        "partition_zero_floor_counts": zero_floor_counts,
        "protocol": {
            "bootstrap_draws": int(manifest["bootstrap_draws"]),
            "bootstrap_unit": manifest["bootstrap_unit"],
            "cover_edges": expected_edges,
            "pairs": pair_protocol,
            "partitions": expected_partitions,
            "support": manifest["support"],
        },
        "source_artifacts": {
            path: sha256(ROOT / path) for path in source_paths
        },
        "status": "PASS",
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = RESULT_DIR / "MANIFEST.sha256"
    manifest_lines = [
        f"{sha256(path).lower()}  {path.name}"
        for path in sorted(RESULT_DIR.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
