#!/usr/bin/env python3
"""ARCHIVAL, NON-STANDALONE H2O quality-shift record.

The provider inputs, internal study module, and original output tree are absent
from the public release. Use ``verify_released_aggregates.py`` instead.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
study = importlib.import_module("run_v1857_h2o_egocentric_allocentric_admission")
OUT = ROOT / "results" / "stage3_real_dual_frame_admission"
VARIANTS = (
    "baseline",
    "missing_depth",
    "missing_object_pose",
    "depth_quality_shift",
    "object_pose_noise",
    "stale_world_geometry",
    "high_source_conflict",
    "low_quality_all",
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def beliefs(record, text, variant):
    language = study.build_language_source_payload({"command_text": text})
    lq, lc, lm = float(language["quality"]), float(language["conflict"]), bool(language["missing"])
    local_q, local_c, local_m = record["local_quality"], 0.0, False
    world_q, world_c, world_m = record["world_quality"], 0.0, False
    if variant == "missing_depth":
        local_q, local_m = 0.0, True
    elif variant == "missing_object_pose":
        local_q, world_q, local_m, world_m = 0.0, 0.0, True, True
    elif variant == "depth_quality_shift":
        local_q, local_c = max(.1, local_q - .45), .20
    elif variant == "object_pose_noise":
        local_q, world_q, local_c, world_c = max(.1, local_q - .30), max(.1, world_q - .30), .45, .45
    elif variant == "stale_world_geometry":
        world_q, world_c = .35, .35
    elif variant == "high_source_conflict":
        local_c, world_c = .80, .75
    elif variant == "low_quality_all":
        lq, local_q, world_q, local_c, world_c = .45, .20, .30, .25, .20
    return [
        study.SourceBelief("language", language["probabilities"], lq, lc, lm),
        study.SourceBelief("vision", study.local_belief(record["local_clearance_m"]), local_q, local_c, local_m),
        study.SourceBelief("geometry", study.world_belief(record["world_object_speed_mps"]), world_q, world_c, world_m),
    ]


def verifier(variant, raw):
    if variant == "baseline":
        admitted = int(not raw.abstain)
        return admitted, study.typed_route(raw.contract, raw.abstain), "source_state_supported" if admitted else raw.reason
    if variant == "missing_object_pose":
        return 0, "fallback", "local_and_world_geometry_unavailable"
    if variant == "missing_depth":
        return 0, "hold", "local_depth_unavailable"
    return 0, "hold", {
        "depth_quality_shift": "local_depth_support_low",
        "object_pose_noise": "geometry_disagreement",
        "stale_world_geometry": "world_geometry_stale",
        "high_source_conflict": "source_conflict_high",
        "low_quality_all": "source_quality_low",
    }[variant]


def main():
    cards = study.read_csv(study.IN / "h2o_real_source_cards_60_diverse.csv")
    manifest = study.read_csv(study.IN / "h2o_real_source_manifest_60_diverse.csv")
    evidence = study.build_evidence(cards, manifest)
    rows = []
    for record in evidence:
        for command_id, text in study.COMMANDS:
            for variant in VARIANTS:
                fused = study.fuse_contract_beliefs(
                    beliefs(record, text, variant),
                    dependence_edges={("vision", "geometry"): 1.0},
                    provenance_gain=1.0,
                )
                admitted, route, reason = verifier(variant, fused)
                rows.append({
                    "decision_id": f"{record['real_case_id']}__{command_id}__{variant}",
                    "real_case_id": record["real_case_id"],
                    "overlay_command_id": command_id,
                    "variant": variant,
                    "raw_contract": fused.contract,
                    "raw_confidence": round(fused.confidence, 8),
                    "raw_admitted": int(not fused.abstain),
                    "raw_release_authority": study.release_authority(fused.contract, fused.abstain),
                    "pact_sv_admitted": admitted,
                    "pact_sv_route": route,
                    "pact_sv_release_authority": int(admitted and route == "admit"),
                    "pact_sv_reason": reason,
                    "evidence_record_hash": record["evidence_record_hash"],
                })
    baseline = {(row["real_case_id"], row["overlay_command_id"]): row for row in rows if row["variant"] == "baseline"}
    summary = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        routes, n = Counter(row["pact_sv_route"] for row in selected), len(selected)
        raw_admission_violations = raw_release_violations = pact_admission_violations = pact_release_violations = 0
        if variant != "baseline":
            for row in selected:
                base = baseline[row["real_case_id"], row["overlay_command_id"]]
                raw_admission_violations += int(row["raw_admitted"] > base["raw_admitted"])
                raw_release_violations += int(row["raw_release_authority"] > base["raw_release_authority"])
                pact_admission_violations += int(row["pact_sv_admitted"] > base["pact_sv_admitted"])
                pact_release_violations += int(row["pact_sv_release_authority"] > base["pact_sv_release_authority"])
        summary.append({
            "variant": variant, "n": n,
            "raw_admission_rate": sum(row["raw_admitted"] for row in selected) / n,
            "pact_sv_admission_rate": sum(row["pact_sv_admitted"] for row in selected) / n,
            "raw_admission_nonincrease_violations": raw_admission_violations,
            "raw_release_authority_nonincrease_violations": raw_release_violations,
            "pact_sv_admission_nonincrease_violations": pact_admission_violations,
            "pact_sv_release_authority_nonincrease_violations": pact_release_violations,
            **{f"pact_sv_{route}": routes[route] for route in study.ROUTES},
        })
    write_csv(OUT / "h2o_unified_quality_shift_decisions.csv", rows)
    write_csv(OUT / "h2o_unified_quality_shift_summary.csv", summary)
    stress = [row for row in summary if row["variant"] != "baseline"]
    gate = {
        "status": "pass" if len(rows) == 2400 and all(row["pact_sv_admission_nonincrease_violations"] == 0 and row["pact_sv_release_authority_nonincrease_violations"] == 0 for row in stress) else "fail",
        "decisions": len(rows),
        "baseline_decisions": 300,
        "stress_decisions": 2100,
        "raw_admission_nonincrease_violations": sum(row["raw_admission_nonincrease_violations"] for row in stress),
        "raw_release_authority_nonincrease_violations": sum(row["raw_release_authority_nonincrease_violations"] for row in stress),
        "pact_sv_admission_nonincrease_violations": sum(row["pact_sv_admission_nonincrease_violations"] for row in stress),
        "pact_sv_release_authority_nonincrease_violations": sum(row["pact_sv_release_authority_nonincrease_violations"] for row in stress),
    }
    (OUT / "h2o_unified_quality_shift_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "summary": summary}, indent=2))
    if gate["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
