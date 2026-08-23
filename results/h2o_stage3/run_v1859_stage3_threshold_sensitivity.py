#!/usr/bin/env python3
"""ARCHIVAL, NON-STANDALONE H2O threshold-sensitivity record.

The provider inputs, internal study module, and original output tree are absent
from the public release. Use ``verify_released_aggregates.py`` instead.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
study = importlib.import_module("run_v1857_h2o_egocentric_allocentric_admission")
OUT = ROOT / "results" / "stage3_real_dual_frame_admission"


def distribution(clearance, low, high):
    if clearance <= low:
        return dict(normal=.05, slow_clearance=.22, hold_confirm=.58, retreat_fallback=.13, bounded_urgent=.02)
    if clearance <= high:
        return dict(normal=.12, slow_clearance=.48, hold_confirm=.30, retreat_fallback=.08, bounded_urgent=.02)
    return dict(normal=.48, slow_clearance=.30, hold_confirm=.14, retreat_fallback=.06, bounded_urgent=.02)


def motion_distribution(speed, low, high):
    if speed >= high:
        return dict(normal=.08, slow_clearance=.42, hold_confirm=.36, retreat_fallback=.12, bounded_urgent=.02)
    if speed >= low:
        return dict(normal=.25, slow_clearance=.42, hold_confirm=.23, retreat_fallback=.08, bounded_urgent=.02)
    return dict(normal=.45, slow_clearance=.30, hold_confirm=.17, retreat_fallback=.06, bounded_urgent=.02)


def main():
    cards = study.read_csv(study.IN / "h2o_real_source_cards_60_diverse.csv")
    manifest = study.read_csv(study.IN / "h2o_real_source_manifest_60_diverse.csv")
    evidence = study.build_evidence(cards, manifest)
    rows = []
    for local_low, local_high in ((.03, .08), (.04, .10), (.05, .12)):
        for world_low, world_high in ((.03, .15), (.05, .25), (.08, .35)):
            study.local_belief = lambda value, a=local_low, b=local_high: distribution(value, a, b)
            study.world_belief = lambda value, a=world_low, b=world_high: motion_distribution(value, a, b)
            decisions = study.build_decisions(evidence)
            summary = {row["setting"]: row for row in study.summarize(decisions)}
            violations = study.monotonicity(decisions)
            raw = sum(int(row["admission_nonincrease_violations"]) for row in violations if row["method"] == "raw_registered")
            pact = sum(int(row["admission_nonincrease_violations"]) for row in violations if row["method"] == "pact_sv")
            rows.append({
                "local_low_m": local_low,
                "local_high_m": local_high,
                "world_low_mps": world_low,
                "world_high_mps": world_high,
                "ego_plus_world_pact_sv_admission": summary["ego_plus_world"]["pact_sv_admission_rate"],
                "ego_plus_world_release_authority": summary["ego_plus_world"]["pact_sv_release_authority_rate"],
                "ego_only_raw_registered_admission": summary["ego_only"]["raw_registered_admission_rate"],
                "world_only_raw_registered_admission": summary["world_only"]["raw_registered_admission_rate"],
                "neither_raw_registered_admission": summary["neither"]["raw_registered_admission_rate"],
                "raw_registered_admission_nonincrease_violations": raw,
                "pact_sv_admission_nonincrease_violations": pact,
            })
    with (OUT / "stage3_threshold_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gate = {
        "status": "pass" if all(row["raw_registered_admission_nonincrease_violations"] > 0 and row["pact_sv_admission_nonincrease_violations"] == 0 for row in rows) else "fail",
        "configurations": len(rows),
        "pact_sv_full_evidence_admission_range": [
            min(row["ego_plus_world_pact_sv_admission"] for row in rows),
            max(row["ego_plus_world_pact_sv_admission"] for row in rows),
        ],
        "raw_registered_violation_range": [
            min(row["raw_registered_admission_nonincrease_violations"] for row in rows),
            max(row["raw_registered_admission_nonincrease_violations"] for row in rows),
        ],
        "pact_sv_violation_range": [
            min(row["pact_sv_admission_nonincrease_violations"] for row in rows),
            max(row["pact_sv_admission_nonincrease_violations"] for row in rows),
        ],
    }
    (OUT / "stage3_threshold_sensitivity_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    if gate["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
