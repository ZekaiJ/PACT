#!/usr/bin/env python3
"""ARCHIVAL, NON-STANDALONE Stage 3 integrity-audit record.

The decision-level inputs and original output tree are not included in the
public release. Use ``verify_released_aggregates.py`` for the released snapshot.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage3_real_dual_frame_admission"
FIG = ROOT / "figures" / "stage3_real_dual_frame_admission"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, tolerance=1e-12):
    return abs(float(left) - float(right)) <= tolerance


def main():
    evidence = read_csv(OUT / "h2o_egocentric_allocentric_evidence.csv")
    decisions = read_csv(OUT / "h2o_egocentric_allocentric_decisions.csv")
    summary = read_csv(OUT / "h2o_egocentric_allocentric_summary.csv")
    violations = read_csv(OUT / "h2o_view_removal_monotonicity.csv")
    gate = json.loads((OUT / "stage3_gate.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, condition, detail=""):
        checks.append({"check": name, "pass": bool(condition), "detail": str(detail)})

    check("evidence_count", len(evidence) == 60, len(evidence))
    check("decision_count", len(decisions) == 1200, len(decisions))
    check("unique_evidence_ids", len({row["real_case_id"] for row in evidence}) == 60)
    check("unique_decision_ids", len({row["decision_id"] for row in decisions}) == 1200)
    check("sequence_count", len({row["sequence_id"] for row in evidence}) == 4)
    check("action_label_count", len({row["action_label"] for row in evidence}) == 19)
    check("object_count", len({row["object_id"] for row in evidence}) == 4)
    check("distance_invariance", max(float(row["distance_invariance_error_m"]) for row in evidence) < 1e-6)
    check("positive_depth", min(float(row["depth_valid_fraction"]) for row in evidence) > 0)
    check("world_temporal_support", min(int(row["world_temporal_support"]) for row in evidence) >= 1)
    check("all_lineage_registered", all(row["lineage_edge"] == "vision--geometry:same_h2o_frame_annotation" for row in decisions))

    counts = Counter(row["setting"] for row in decisions)
    for setting in ("ego_plus_world", "ego_only", "world_only", "neither"):
        check(f"setting_{setting}_count", counts[setting] == 300, counts[setting])
    per_identity = defaultdict(set)
    for row in decisions:
        per_identity[(row["real_case_id"], row["overlay_command_id"])].add(row["setting"])
    check("paired_view_roles", len(per_identity) == 300 and all(len(value) == 4 for value in per_identity.values()))

    check("ego_only_typed_hold", all(row["pact_sv_admitted"] == "0" and row["pact_sv_route"] == "hold" for row in decisions if row["setting"] == "ego_only"))
    check("world_only_typed_confirm", all(row["pact_sv_admitted"] == "0" and row["pact_sv_route"] == "confirm" for row in decisions if row["setting"] == "world_only"))
    check("neither_typed_fallback", all(row["pact_sv_admitted"] == "0" and row["pact_sv_route"] == "fallback" for row in decisions if row["setting"] == "neither"))
    check("no_pact_sv_view_removal_violations", all(int(row["admission_nonincrease_violations"]) == 0 and int(row["release_authority_nonincrease_violations"]) == 0 for row in violations if row["method"] == "pact_sv"))

    summary_map = {row["setting"]: row for row in summary}
    for setting, row in summary_map.items():
        subset = [item for item in decisions if item["setting"] == setting]
        n = len(subset)
        recomputed = {
            "raw_blind_admission_rate": sum(int(item["raw_blind_admitted"]) for item in subset) / n,
            "raw_registered_admission_rate": sum(int(item["raw_registered_admitted"]) for item in subset) / n,
            "pact_sv_admission_rate": sum(int(item["pact_sv_admitted"]) for item in subset) / n,
            "pact_sv_release_authority_rate": sum(int(item["pact_sv_release_authority"]) for item in subset) / n,
        }
        for metric, value in recomputed.items():
            check(f"summary_{setting}_{metric}", close(row[metric], value), f"{row[metric]} vs {value}")
        route_sum = sum(float(row[f"pact_sv_{route}_rate"]) for route in ("admit", "hold", "confirm", "retreat", "fallback"))
        check(f"summary_{setting}_route_sum", close(route_sum, 1.0), route_sum)

    indexed = {(row["real_case_id"], row["overlay_command_id"], row["setting"]): row for row in decisions}
    for audit_row in violations:
        method, setting = audit_row["method"], audit_row["removed_to_setting"]
        admission = release = 0
        for case, command in per_identity:
            base, candidate = indexed[case, command, "ego_plus_world"], indexed[case, command, setting]
            admission += int(int(candidate[f"{method}_admitted"]) > int(base[f"{method}_admitted"]))
            release += int(int(candidate[f"{method}_release_authority"]) > int(base[f"{method}_release_authority"]))
        check(f"violations_{method}_{setting}_admission", admission == int(audit_row["admission_nonincrease_violations"]))
        check(f"violations_{method}_{setting}_release", release == int(audit_row["release_authority_nonincrease_violations"]))

    forbidden = {"reference_contract", "wrong_release", "pact_prediction_correctness", "manual_audit_label", "post_execution_outcome"}
    check("forbidden_fields_absent", not any(forbidden & set(row) for row in evidence + decisions))
    check("figure_pdf_present", (FIG / "h2o_egocentric_allocentric_admission.pdf").exists())
    check("figure_png_present", (FIG / "h2o_egocentric_allocentric_admission.png").exists())
    check("gate_pass", gate["status"] == "pass")

    status = "pass" if all(row["pass"] for row in checks) else "fail"
    result = {"status": status, "passed": sum(row["pass"] for row in checks), "total": len(checks), "checks": checks}
    (OUT / "stage3_integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (OUT / "stage3_hash_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        files = sorted([path for path in OUT.iterdir() if path.is_file()] + [path for path in FIG.iterdir() if path.is_file()])
        for path in files:
            if path.name == "stage3_hash_manifest.csv":
                continue
            writer.writerow({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    print(json.dumps(result, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
