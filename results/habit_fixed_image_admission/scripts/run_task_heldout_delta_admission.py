#!/usr/bin/env python3
"""Compose task-held-out HABIT temporal change with semantic and target evidence."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import run_habit_task_heldout_admission as heldout
import run_habit_task_object_admission as admission


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results"
    / "habit_level2"
    / "task_heldout_admission"
    / "task_heldout_phase_predictions.csv"
)
OUT = ROOT / "results" / "habit_level2" / "task_heldout_delta_admission"


def bool_field(value: str) -> bool:
    return str(value).lower() == "true"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase_rows = admission.read_csv(INPUT)
    by_event: dict[str, dict[str, float]] = defaultdict(dict)
    sid_by_event: dict[str, str] = {}
    for row in phase_rows:
        if row["setting"] != "all_five":
            continue
        by_event[row["event_id"]][row["window"]] = float(
            row["task_heldout_ready_probability"]
        )
        sid_by_event[row["event_id"]] = row["sid"]
    phase_lookup: dict[tuple[str, str], float] = {}
    event_rows = []
    for event_id, scores in sorted(by_event.items()):
        proximal_key = "ready" if "ready" in scores else "event_proximal"
        if set(scores) not in ({"early", "ready"}, {"early", "event_proximal"}):
            raise RuntimeError(f"unexpected temporal pair for {event_id}: {sorted(scores)}")
        delta = scores[proximal_key] - scores["early"]
        phase_lookup[(event_id, "early")] = 0.0
        phase_lookup[(event_id, "ready")] = float(delta > 0.0)
        event_rows.append(
            {
                "event_id": event_id,
                "sid": sid_by_event[event_id],
                "early_probability": scores["early"],
                "event_proximal_probability": scores[proximal_key],
                "delta": delta,
                "positive_temporal_change": int(delta > 0.0),
            }
        )
    if len(event_rows) != 1_128:
        raise RuntimeError(f"expected 1,128 temporal pairs, found {len(event_rows)}")

    decisions = heldout.build_admission_rows(phase_lookup)
    in_domain_rows = admission.read_csv(
        admission.OUT / "habit_task_object_admission_decisions.csv"
    )
    for row in in_domain_rows:
        row["candidate_admit"] = bool_field(row["candidate_admit"])
        row["verified_admit"] = bool_field(row["verified_admit"])
        row["reference_admit"] = bool_field(row["reference_admit"])

    candidate = admission.metrics(decisions, "candidate_admit")
    delta_metrics = admission.metrics(decisions, "verified_admit")
    in_domain = admission.metrics(in_domain_rows, "verified_admit")
    by_task = admission.task_metrics(decisions, "verified_admit")
    in_domain_by_task = {
        row["sid"]: row for row in admission.task_metrics(in_domain_rows, "verified_admit")
    }
    folds_not_worse = sum(
        float(row["wrong_authorization_all"])
        <= float(in_domain_by_task[row["sid"]]["wrong_authorization_all"]) + 1e-12
        for row in by_task
    )
    retention = (
        delta_metrics["correct_authorization_all"]
        / candidate["correct_authorization_all"]
    )
    promoted = (
        delta_metrics["wrong_authorization_all"] <= in_domain["wrong_authorization_all"] + 0.002
        and retention >= 0.90
        and folds_not_worse >= 5
    )

    heldout.write_csv(OUT / "task_heldout_delta_events.csv", event_rows, list(event_rows[0]))
    heldout.write_csv(
        OUT / "task_heldout_delta_admission_decisions.csv",
        decisions,
        list(decisions[0]),
    )
    heldout.write_csv(OUT / "task_heldout_delta_by_task.csv", by_task, list(by_task[0]))
    report = {
        "status": "pass" if promoted else "hold",
        "analysis_role": "task-held-out temporal-change evidence composed with fixed semantic candidates and current target state",
        "runtime_evidence": "the event-proximal score is compared with an earlier pre-release observation from the same episode; positive change is required",
        "claim_boundary": "The result evaluates sequential pre-release evidence, not single-image phase recognition or an unseen-task robot policy.",
        "candidate_only": candidate,
        "task_heldout_delta_admission": delta_metrics,
        "task_heldout_task_macro": admission.task_macro(decisions, "verified_admit"),
        "task_heldout_by_task": by_task,
        "in_domain_absolute_admission": in_domain,
        "paired_task_balanced_bootstrap": heldout.paired_task_bootstrap(
            decisions, in_domain_rows
        ),
        "promotion_checks": {
            "wrong_all_within_0_002_of_in_domain": delta_metrics[
                "wrong_authorization_all"
            ]
            <= in_domain["wrong_authorization_all"] + 0.002,
            "candidate_correct_retention_at_least_0_90": retention >= 0.90,
            "task_folds_not_worse_than_in_domain_wrong_all": folds_not_worse,
        },
        "input_sha256": {
            str(INPUT.relative_to(ROOT)): admission.sha256(INPUT),
            str(Path(__file__).resolve().relative_to(ROOT)): admission.sha256(
                Path(__file__).resolve()
            ),
        },
    }
    (OUT / "HABIT_TASK_HELDOUT_DELTA_ADMISSION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
