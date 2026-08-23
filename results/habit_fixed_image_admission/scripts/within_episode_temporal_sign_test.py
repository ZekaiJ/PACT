#!/usr/bin/env python3
"""Recompute the released within-episode temporal sign test."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[1]
EVENTS = BUNDLE / "task_heldout_delta_admission" / "task_heldout_delta_events.csv"
DECISIONS = (
    BUNDLE
    / "task_heldout_delta_admission"
    / "task_heldout_delta_admission_decisions.csv"
)
OUTPUT = (
    BUNDLE
    / "task_heldout_temporal_model"
    / "within_episode_temporal_sign_test.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with EVENTS.open(encoding="utf-8-sig", newline="") as handle:
        all_events = list(csv.DictReader(handle))
    with DECISIONS.open(encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    selected_ids = {
        row["event_id"]
        for row in decisions
        if row["condition"] == "factual" and row["window"] == "ready"
    }
    rows = [row for row in all_events if row["event_id"] in selected_ids]
    if len(rows) != 60 or len(selected_ids) != 60:
        raise AssertionError("Expected 60 unique paired endpoint events")

    task_counts = Counter(row["sid"] for row in rows)
    if len(task_counts) != 6 or set(task_counts.values()) != {10}:
        raise AssertionError("Expected six tasks with ten events each")

    differences = [float(row["delta"]) for row in rows]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    non_ties = positive + negative
    one_sided_p = sum(
        math.comb(non_ties, count) for count in range(positive, non_ties + 1)
    ) / (2**non_ties)
    event_hash = hashlib.sha256(
        "\n".join(sorted(row["event_id"] for row in rows)).encode()
    ).hexdigest()

    result = {
        "analysis": "one-sided exact paired sign test",
        "null": "early and event-proximal window labels are exchangeable within each selected episode",
        "selection_rule": "condition=factual and window=ready in the released 720-case decision table",
        "endpoint_episodes": len(rows),
        "tasks": len(task_counts),
        "episodes_per_task": dict(sorted(task_counts.items())),
        "positive_differences": positive,
        "negative_differences": negative,
        "ties": ties,
        "mean_proximal_minus_early": sum(differences) / len(differences),
        "one_sided_exact_p": one_sided_p,
        "selected_event_id_sha256": event_hash,
        "inputs": {
            "task_heldout_delta_admission/task_heldout_delta_events.csv": sha256(EVENTS),
            "task_heldout_delta_admission/task_heldout_delta_admission_decisions.csv": sha256(DECISIONS),
            "scripts/within_episode_temporal_sign_test.py": sha256(Path(__file__)),
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
