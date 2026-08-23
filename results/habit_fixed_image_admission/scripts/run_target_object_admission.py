#!/usr/bin/env python3
"""Compose HABIT semantic candidates with target and temporal evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
COUNTERFACTUAL = (
    ROOT / "results" / "habit_level2" / "counterfactual_semantics" / "scored_rows.csv"
)
NON_FM = ROOT / "results" / "habit_level2" / "nonfm_baseline" / "nonfm_predictions.csv"
NON_FM_REPORT = (
    ROOT / "results" / "habit_level2" / "nonfm_baseline" / "HABIT_NONFM_BASELINE.json"
)
PROTOCOL = ROOT / "results" / "habit_level2" / "HABIT_RELEASE_PROTOCOL_LOCK.json"
OUT = ROOT / "results" / "habit_level2" / "task_object_admission"

PHASE_THRESHOLD = 0.5
BOOTSTRAPS = 10_000
SEED = 20260713
TASK_NAMES = {
    "S6759": "Power Cable",
    "S7025": "Organizing box",
    "S7414": "Paper to-go box",
    "S7788": "iced cup",
    "S8940": "Airtight Container lid",
    "S8975": "100 cm Hotel Towel",
}
EVALUATION_ONLY = {"window", "condition", "expected_target_match"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def candidate_admission(source: Mapping[str, Any]) -> bool:
    assert not (set(source) & EVALUATION_ONLY)
    return (
        bool(source["parsed"])
        and bool(source["schema_valid"])
        and source["target_match"] == "matched"
        and source["release_readiness"] == "ready"
    )


def verified_decision(source: Mapping[str, Any]) -> tuple[str, str]:
    """Return a release or a typed non-release decision from observable fields."""
    assert not (set(source) & EVALUATION_ONLY)
    if not source["parsed"] or not source["schema_valid"]:
        return "retreat_fallback", "invalid_candidate_schema"
    if source["query_target_sid"] != source["native_target_sid"]:
        return "hold_confirm", "target_state_mismatch"
    if source["target_match"] != "matched":
        return "hold_confirm", "target_not_visually_supported"
    if source["release_readiness"] != "ready":
        return "hold_confirm", "candidate_not_ready"
    if source["phase_probability"] < PHASE_THRESHOLD:
        return "hold_confirm", "temporal_evidence_not_ready"
    return "release", "supported_release"


def metrics(rows: list[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    admitted = [bool(row[key]) for row in rows]
    reference = [bool(row["reference_admit"]) for row in rows]
    tp = sum(a and y for a, y in zip(admitted, reference))
    fp = sum(a and not y for a, y in zip(admitted, reference))
    positives = sum(reference)
    n = len(rows)
    return {
        "n": n,
        "admitted": sum(admitted),
        "coverage": sum(admitted) / n,
        "correct_authorization_all": tp / n,
        "wrong_authorization_all": fp / n,
        "wrong_authorization_admitted": fp / sum(admitted) if sum(admitted) else 0.0,
        "authorization_accuracy": sum(a == y for a, y in zip(admitted, reference)) / n,
        "ready_factual_recall": tp / positives if positives else 0.0,
    }


def task_metrics(rows: list[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sid"])].append(row)
    result = []
    for sid in sorted(grouped):
        entry = metrics(grouped[sid], key)
        result.append({"sid": sid, "task": TASK_NAMES[sid], **entry})
    return result


def task_macro(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    per_task = task_metrics(rows, key)
    metric_names = (
        "coverage",
        "correct_authorization_all",
        "wrong_authorization_all",
        "wrong_authorization_admitted",
        "authorization_accuracy",
        "ready_factual_recall",
    )
    result = {
        name: sum(float(row[name]) for row in per_task) / len(per_task)
        for name in metric_names
    }
    result["worst_task_ready_factual_recall"] = min(
        float(row["ready_factual_recall"]) for row in per_task
    )
    result["worst_task_sid"] = min(
        per_task, key=lambda row: (float(row["ready_factual_recall"]), str(row["sid"]))
    )["sid"]
    return result


def bootstrap(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task_event: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_task_event[str(row["sid"])][str(row["event_id"])].append(row)
    rng = random.Random(SEED)
    observed_candidate = task_macro(rows, "candidate_admit")
    observed_verified = task_macro(rows, "verified_admit")
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sample: list[Mapping[str, Any]] = []
        for sid in sorted(by_task_event):
            event_ids = sorted(by_task_event[sid])
            for _ in event_ids:
                sample.extend(by_task_event[sid][rng.choice(event_ids)])
        candidate = task_macro(sample, "candidate_admit")
        verified = task_macro(sample, "verified_admit")
        for name in (
            "wrong_authorization_all",
            "correct_authorization_all",
            "ready_factual_recall",
        ):
            draws[name].append(float(verified[name]) - float(candidate[name]))
    result = []
    for name, values in draws.items():
        values.sort()
        result.append(
            {
                "metric": name,
                "verified_minus_candidate": float(observed_verified[name]) - float(observed_candidate[name]),
                "ci95_low": values[int(0.025 * len(values))],
                "ci95_high": values[int(0.975 * len(values)) - 1],
                "bootstrap_iterations": BOOTSTRAPS,
                "sampling_unit": "episode within each of six tasks",
            }
        )
    return result


def main() -> None:
    source_rows = read_csv(COUNTERFACTUAL)
    phase_rows = read_csv(NON_FM)
    phase_report = json.loads(NON_FM_REPORT.read_text(encoding="utf-8"))
    if len(source_rows) != 720:
        raise RuntimeError(f"Expected 720 counterfactual records, found {len(source_rows)}")
    separation = phase_report["development_test_separation"]
    if any(
        int(separation[key]) != 0
        for key in ("episode_overlap", "identifier_overlap", "exact_image_sha256_overlap")
    ):
        raise RuntimeError("The conventional phase source is not development/test disjoint")

    phase_lookup: dict[tuple[str, str], float] = {}
    for row in phase_rows:
        if row["setting"] == "all_five":
            phase_lookup[(row["event_id"], row["window"])] = float(
                row["resnet50_tfidf_ready_probability"]
            )

    decisions: list[dict[str, Any]] = []
    policy_fields: set[str] = set()
    for row in source_rows:
        phase_key = (row["event_id"], row["window"])
        if phase_key not in phase_lookup:
            raise RuntimeError(f"No phase estimate for {phase_key}")
        source = {
            "parsed": row["parsed"] == "1",
            "schema_valid": row["schema_valid"] == "1",
            "target_match": row["target_match"],
            "release_readiness": row["release_readiness"],
            "query_target_sid": row["target_sid"],
            "native_target_sid": row["sid"],
            "phase_probability": phase_lookup[phase_key],
        }
        policy_fields.update(source)
        candidate = candidate_admission(source)
        target_verified = candidate and source["query_target_sid"] == source["native_target_sid"]
        temporal_verified = candidate and source["phase_probability"] >= PHASE_THRESHOLD
        decision, reason = verified_decision(source)
        decisions.append(
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "episode_id": row["episode_id"],
                "sid": row["sid"],
                "task": TASK_NAMES[row["sid"]],
                "window": row["window"],
                "condition": row["condition"],
                "query_target_sid": row["target_sid"],
                "candidate_admit": candidate,
                "target_verified_admit": target_verified,
                "temporal_verified_admit": temporal_verified,
                "verified_admit": decision == "release",
                "verified_decision": decision,
                "decision_reason": reason,
                "phase_probability": source["phase_probability"],
                "reference_admit": row["condition"] == "factual" and row["window"] == "ready",
            }
        )

    if policy_fields & EVALUATION_ONLY:
        raise RuntimeError("Evaluation-only fields reached the admission policy")
    if len({row["event_id"] for row in decisions}) != 60:
        raise RuntimeError("The decision set must contain 60 fixed events")
    if Counter(row["sid"] for row in decisions) != Counter({sid: 120 for sid in TASK_NAMES}):
        raise RuntimeError("Task denominator is not balanced")

    candidate = metrics(decisions, "candidate_admit")
    verified = metrics(decisions, "verified_admit")
    ablation_keys = (
        ("semantic_candidate", "candidate_admit"),
        ("current_target_only", "target_verified_admit"),
        ("temporal_source_only", "temporal_verified_admit"),
        ("combined_verification", "verified_admit"),
    )
    ablation = []
    for method, key in ablation_keys:
        entry = metrics(decisions, key)
        entry["off_target_wrong_authorizations"] = sum(
            bool(row[key]) and row["query_target_sid"] != row["sid"] for row in decisions
        )
        entry["premature_factual_authorizations"] = sum(
            bool(row[key])
            and row["query_target_sid"] == row["sid"]
            and row["window"] == "early"
            for row in decisions
        )
        assert round(entry["wrong_authorization_all"] * entry["n"]) == (
            entry["off_target_wrong_authorizations"]
            + entry["premature_factual_authorizations"]
        )
        ablation.append({"method": method, **entry})
    assert all(
        row["verified_admit"]
        == (row["target_verified_admit"] and row["temporal_verified_admit"])
        for row in decisions
    )
    blocked = [row for row in decisions if row["candidate_admit"] and not row["verified_admit"]]
    wrong_blocked = sum(not row["reference_admit"] for row in blocked)
    correct_blocked = sum(row["reference_admit"] for row in blocked)

    slice_counts = []
    for method, key in ablation_keys:
        for window in ("early", "ready"):
            for condition in ("factual", "counterfactual"):
                subset = [
                    row
                    for row in decisions
                    if row["window"] == window and row["condition"] == condition
                ]
                slice_counts.append(
                    {
                        "method": method,
                        "window": window,
                        "condition": condition,
                        "n": len(subset),
                        "admitted": sum(bool(row[key]) for row in subset),
                        "admission_rate": sum(bool(row[key]) for row in subset) / len(subset),
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "habit_task_object_admission_decisions.csv",
        decisions,
        [
            "id",
            "event_id",
            "episode_id",
            "sid",
            "task",
            "window",
            "condition",
            "query_target_sid",
            "candidate_admit",
            "target_verified_admit",
            "temporal_verified_admit",
            "verified_admit",
            "verified_decision",
            "decision_reason",
            "phase_probability",
            "reference_admit",
        ],
    )
    write_csv(
        OUT / "habit_task_object_admission_slices.csv",
        slice_counts,
        ["method", "window", "condition", "n", "admitted", "admission_rate"],
    )
    write_csv(
        OUT / "habit_task_object_admission_by_task.csv",
        [
            {"method": method, **row}
            for method, key in ablation_keys
            for row in task_metrics(decisions, key)
        ],
        [
            "method",
            "sid",
            "task",
            "n",
            "admitted",
            "coverage",
            "correct_authorization_all",
            "wrong_authorization_all",
            "wrong_authorization_admitted",
            "authorization_accuracy",
            "ready_factual_recall",
        ],
    )

    report = {
        "status": "pass",
        "analysis_role": "task- and object-conditioned admission on fixed HABIT real-image events",
        "denominator": {
            "rows": len(decisions),
            "events": len({row["event_id"] for row in decisions}),
            "episodes": len({row["episode_id"] for row in decisions}),
            "tasks": len(TASK_NAMES),
            "rows_per_event": 12,
            "composition": "2 temporal frames x 1 factual plus 5 counterfactual targets",
        },
        "admission_sources": {
            "semantic_candidate": "schema-valid frozen VLM target match and release-readiness fields",
            "target_state": "task/object identity derived from HABIT task metadata and compared with the queried target",
            "temporal_state": "development-trained frozen ResNet-50 plus TF-IDF phase probability at its default 0.5 decision threshold",
            "lineage": "the VLM and conventional encoder share visual parents and are combined by verification, not counted as independent votes",
        },
        "phase_source_validation": {
            "selected_c": phase_report["development_selection"]["resnet50_tfidf"]["selected_c"],
            "development_rows": phase_report["development_selection"]["resnet50_tfidf"]["development_rows"],
            "development_episodes": separation["development_episodes"],
            "test_episodes": separation["test_episodes"],
            "episode_overlap": separation["episode_overlap"],
            "identifier_overlap": separation["identifier_overlap"],
            "exact_image_sha256_overlap": separation["exact_image_sha256_overlap"],
        },
        "candidate_only": candidate,
        "evidence_verified": verified,
        "factorial_evidence_role_ablation": ablation,
        "candidate_task_macro": task_macro(decisions, "candidate_admit"),
        "verified_task_macro": task_macro(decisions, "verified_admit"),
        "transition": {
            "candidate_admissions": candidate["admitted"],
            "candidate_to_nonadmission": len(blocked),
            "wrong_authorizations_blocked": wrong_blocked,
            "correct_authorizations_blocked": correct_blocked,
            "blocked_wrong_fraction": wrong_blocked / len(blocked),
            "candidate_correct_authorization_retention": (
                verified["correct_authorization_all"] / candidate["correct_authorization_all"]
            ),
        },
        "slice_counts": slice_counts,
        "task_balanced_episode_bootstrap": bootstrap(decisions),
        "typed_decision_counts": Counter(row["verified_decision"] for row in decisions),
        "nonadmission_reason_counts": Counter(row["decision_reason"] for row in decisions),
        "policy_input_fields": sorted(policy_fields),
        "forbidden_field_check": {
            "evaluation_only_fields": sorted(EVALUATION_ONLY),
            "violations": sorted(policy_fields & EVALUATION_ONLY),
        },
        "claim_boundary": [
            "This analysis evaluates evidence composition for action admission on fixed real multi-view observations.",
            "The target identity is derived from HABIT task metadata and is used as source-state evidence, not as a learned object recognizer.",
            "The early/event-proximal labels evaluate release timing; they are not available to the admission policy.",
            "The result is not physical-safety certification, participant validation, or validation of the complete five-contract ontology.",
        ],
        "input_sha256": {
            str(COUNTERFACTUAL.relative_to(ROOT)): sha256(COUNTERFACTUAL),
            str(NON_FM.relative_to(ROOT)): sha256(NON_FM),
            str(NON_FM_REPORT.relative_to(ROOT)): sha256(NON_FM_REPORT),
            str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
        },
    }
    with (OUT / "habit_task_object_admission_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
