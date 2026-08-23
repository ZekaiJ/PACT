"""Evaluate one fixed target-plus-event admission rule across three checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "habit_checkpoint_admission"
SCORED = RESULTS / "scored_rows_three_checkpoint.csv"
PROXIMITY = RESULTS / "event_proximity_predictions.csv"
PROXIMITY_GATE = RESULTS / "event_proximity_source_gate.json"
MODEL_GATE = RESULTS / "foundation_model_source_gate.json"
EXPECTED_HASHES = {
    SCORED: "4d573b7578dcc331b449113dd81812e14e28e1c15eb63a9cc8dd913e79227f0a",
    PROXIMITY: "f225a340e9f0bd697afcef4f40dc17117963503abc060855c53221c84b30624e",
    PROXIMITY_GATE: "7b3b5ccaf2a4507fe9b2e3ba7ff442393bd65a60586290536f91345fc113f2a9",
    MODEL_GATE: "e1aed6a2506ca3a73277bc49d3f5ddd1cb99889192a9abe3194cb44e1bcbf0e6",
}
MODELS = ("qwen3vl_8b", "qwen3vl_32b", "internvl3_8b")
METHODS = ("candidate_only", "target_identity", "event_proximity", "combined")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260811
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    admitted = [bool(row[field]) for row in rows]
    reference = [bool(row["reference_admit"]) for row in rows]
    correct = sum(a and y for a, y in zip(admitted, reference))
    wrong = sum(a and not y for a, y in zip(admitted, reference))
    positives = sum(reference)
    return {
        "records": len(rows),
        "admitted": sum(admitted),
        "correct_admitted": correct,
        "wrong_admitted": wrong,
        "coverage": sum(admitted) / len(rows),
        "correct_all": correct / len(rows),
        "wrong_all": wrong / len(rows),
        "ready_factual_recall": correct / positives,
    }


def task_macro(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sid"]].append(row)
    if len(grouped) != 6 or {len(value) for value in grouped.values()} != {120}:
        raise AssertionError("Expected six balanced 120-row tasks")
    per_task = [metrics(value, field) for value in grouped.values()]
    return {
        name: sum(float(row[name]) for row in per_task) / len(per_task)
        for name in ("coverage", "correct_all", "wrong_all", "ready_factual_recall")
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def bootstrap_deltas(rows_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    episode_rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        model: defaultdict(dict) for model in MODELS
    }
    task_episodes: dict[str, set[str]] = defaultdict(set)
    for model, rows in rows_by_model.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["sid"], row["episode_id"])].append(row)
        for (sid, episode_id), values in grouped.items():
            if len(values) != 12:
                raise AssertionError(f"{model}/{episode_id}: expected 12 rows")
            episode_rows[model][sid][episode_id] = values
            task_episodes[sid].add(episode_id)
    if len(task_episodes) != 6 or {len(value) for value in task_episodes.values()} != {10}:
        raise AssertionError("Expected ten episodes in each of six tasks")

    observed = {
        model: {
            metric: task_macro(rows, "combined")[metric]
            - task_macro(rows, "candidate_only")[metric]
            for metric in ("wrong_all", "correct_all", "ready_factual_recall")
        }
        for model, rows in rows_by_model.items()
    }
    draws = {
        (model, metric): []
        for model in MODELS
        for metric in ("wrong_all", "correct_all", "ready_factual_recall")
    }
    rng = random.Random(BOOTSTRAP_SEED)
    ordered_tasks = sorted(task_episodes)
    for _ in range(BOOTSTRAPS):
        sampled_ids = {
            sid: [rng.choice(sorted(task_episodes[sid])) for _ in range(10)]
            for sid in ordered_tasks
        }
        for model in MODELS:
            sampled = [
                row
                for sid in ordered_tasks
                for episode_id in sampled_ids[sid]
                for row in episode_rows[model][sid][episode_id]
            ]
            candidate = task_macro(sampled, "candidate_only")
            combined = task_macro(sampled, "combined")
            for metric in ("wrong_all", "correct_all", "ready_factual_recall"):
                draws[(model, metric)].append(combined[metric] - candidate[metric])

    return [
        {
            "model": model,
            "contrast": "combined_minus_candidate",
            "metric": metric,
            "point": observed[model][metric],
            "ci95_low": percentile(draws[(model, metric)], 0.025),
            "ci95_high": percentile(draws[(model, metric)], 0.975),
            "bootstrap_draws": BOOTSTRAPS,
            "bootstrap_unit": "episode within task",
        }
        for model in MODELS
        for metric in ("wrong_all", "correct_all", "ready_factual_recall")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    for path, expected in EXPECTED_HASHES.items():
        if sha256(path) != expected:
            raise AssertionError(f"frozen input changed: {path.name}")
    proximity_gate = json.loads(PROXIMITY_GATE.read_text(encoding="utf-8"))
    separation = proximity_gate["development_test_separation"]
    if any(
        int(separation[name]) != 0
        for name in ("episode_overlap", "identifier_overlap", "exact_image_sha256_overlap")
    ):
        raise AssertionError("Event-proximity development/test separation failed")
    model_gate = json.loads(MODEL_GATE.read_text(encoding="utf-8"))
    if model_gate.get("status") != "PASS_720_THREE_CHECKPOINT_COMPLETION":
        raise AssertionError("Foundation-model source gate did not pass")

    scored = read_csv(SCORED)
    if len(scored) != 2_160 or Counter(row["model"] for row in scored) != Counter(
        {model: 720 for model in MODELS}
    ):
        raise AssertionError("Expected 720 frozen rows for each checkpoint")
    proximity_rows = read_csv(PROXIMITY)
    proximity = {
        (row["event_id"], row["window"]): float(
            row["resnet50_tfidf_ready_probability"]
        )
        for row in proximity_rows
        if row["setting"] == "all_five"
    }

    decisions: list[dict[str, Any]] = []
    for row in scored:
        key = (row["event_id"], row["window"])
        if key not in proximity:
            raise AssertionError(f"Missing event-proximity output: {key}")
        candidate = row["target_match"] == "matched" and row["release_readiness"] == "ready"
        target = candidate and row["target_sid"] == row["sid"]
        temporal = candidate and proximity[key] >= THRESHOLD
        decisions.append(
            {
                "model": row["model"],
                "id": row["id"],
                "event_id": row["event_id"],
                "episode_id": row["episode_id"],
                "sid": row["sid"],
                "window": row["window"],
                "condition": row["condition"],
                "query_target_sid": row["target_sid"],
                "candidate_only": candidate,
                "target_identity": target,
                "event_proximity": temporal,
                "combined": target and temporal,
                "event_proximity_probability": proximity[key],
                "reference_admit": row["condition"] == "factual" and row["window"] == "ready",
            }
        )

    rows_by_model = {
        model: [row for row in decisions if row["model"] == model] for model in MODELS
    }
    summaries = []
    task_rows = []
    for model, rows in rows_by_model.items():
        for method in METHODS:
            summaries.append({"model": model, "method": method, **metrics(rows, method)})
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[row["sid"]].append(row)
            for sid in sorted(grouped):
                task_rows.append(
                    {"model": model, "method": method, "sid": sid, **metrics(grouped[sid], method)}
                )

    expected_counts = {
        ("qwen3vl_8b", "candidate_only"): (144, 53, 91),
        ("qwen3vl_8b", "combined"): (52, 51, 1),
        ("qwen3vl_32b", "candidate_only"): (143, 57, 86),
        ("qwen3vl_32b", "combined"): (55, 54, 1),
        ("internvl3_8b", "candidate_only"): (98, 27, 71),
        ("internvl3_8b", "combined"): (25, 24, 1),
    }
    by_key = {(row["model"], row["method"]): row for row in summaries}
    for key, expected in expected_counts.items():
        observed = by_key[key]
        values = (
            int(observed["admitted"]),
            int(observed["correct_admitted"]),
            int(observed["wrong_admitted"]),
        )
        if values != expected:
            raise AssertionError(f"checkpoint admission count changed: {key}: {values}")

    bootstrap = bootstrap_deltas(rows_by_model)
    write_csv(output / "checkpoint_admission_decisions.csv", decisions)
    write_csv(output / "checkpoint_admission_summary.csv", summaries)
    write_csv(output / "checkpoint_admission_by_task.csv", task_rows)
    write_csv(output / "checkpoint_admission_bootstrap.csv", bootstrap)
    gate = {
        "status": "PASS",
        "analysis": "fixed target-plus-event admission across three frozen checkpoints",
        "models": list(MODELS),
        "rows_per_model": 720,
        "episode_clusters": 60,
        "tasks": 6,
        "threshold": THRESHOLD,
        "bootstrap": {
            "draws": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "unit": "episode within task",
        },
        "expected_counts": {
            f"{model}/{method}": {
                "admitted": counts[0],
                "correct_admitted": counts[1],
                "wrong_admitted": counts[2],
            }
            for (model, method), counts in expected_counts.items()
        },
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): digest for path, digest in EXPECTED_HASHES.items()},
        "analysis_script_sha256": sha256(Path(__file__)),
        "claim_boundary": (
            "Frozen-output checkpoint robustness of one fixed admission rule; not a "
            "claim of checkpoint independence, closed-loop execution, or physical safety."
        ),
    }
    (output / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
