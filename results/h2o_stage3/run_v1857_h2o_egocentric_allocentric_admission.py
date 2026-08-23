#!/usr/bin/env python3
"""ARCHIVAL, NON-STANDALONE H2O generation record.

This file cannot run from the public release because provider-controlled inputs
and internal fusion modules are not included. It is retained only to document
the original aggregate-generation path.

H2O egocentric--allocentric evidence-role admission study.

The local H2O payload contains one head-mounted RGB-D stream.  The
allocentric role is the same record transformed by the dataset camera pose;
it is not misrepresented as a second-camera image.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "E3_HRC" / "E3_HRC-main" / "src"))

from safe_fuse import SourceBelief, fuse_contract_beliefs  # noqa: E402
from safe_fuse_language_adapters import build_language_source_payload  # noqa: E402

IN = ROOT / "results" / "goal_i" / "h2o_v1835_diverse_gate"
ASSET = IN / "assets"
OUT = ROOT / "results" / "stage3_real_dual_frame_admission"
FIG = ROOT / "figures" / "stage3_real_dual_frame_admission"
COMMANDS = [
    ("normal", "Pass me the object."),
    ("cautious", "Careful, my hand is close. Pass it slowly."),
    ("uncertain", "Wait, I am not ready."),
    ("corrective", "Stop, my hand is there."),
    ("bounded_urgent", "Move it now safely, but do not come close to my hand."),
]
SETTINGS = [
    ("ego_plus_world", "Ego + world", True, True),
    ("ego_only", "Ego only", True, False),
    ("world_only", "World only", False, True),
    ("neither", "Neither", False, False),
]
ROUTES = ("admit", "hold", "confirm", "retreat", "fallback")
FORBIDDEN = {"reference_contract", "wrong_release", "pact_prediction_correctness", "manual_audit_label", "post_execution_outcome"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(path: Path):
    return [float(item) for item in path.read_text(encoding="utf-8").split()]


def wrists(path: Path):
    row = values(path)
    if len(row) != 124:
        raise ValueError(f"unexpected MANO row: {path}")
    points = [row[start + 1 : start + 4] for start in (0, 62) if row[start] > 0.5]
    return np.asarray(points, dtype=float)


def object_points(path: Path):
    row = values(path)
    if len(row) != 64:
        raise ValueError(f"unexpected object row: {path}")
    return int(row[0]), np.asarray(row[1:], dtype=float).reshape(21, 3)


def transform(path: Path, prefixed=False):
    row = values(path)[1 if prefixed else 0 :]
    if len(row) != 16:
        raise ValueError(f"unexpected transform row: {path}")
    return np.asarray(row, dtype=float).reshape(4, 4)


def apply_transform(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))]
    return (matrix @ homogeneous.T).T[:, :3]


def rotation_error(matrix):
    rotation = matrix[:3, :3]
    return float(np.linalg.norm(rotation.T @ rotation - np.eye(3)) + abs(np.linalg.det(rotation) - 1.0))


def local_belief(clearance):
    if clearance <= 0.04:
        return dict(normal=.05, slow_clearance=.22, hold_confirm=.58, retreat_fallback=.13, bounded_urgent=.02)
    if clearance <= 0.10:
        return dict(normal=.12, slow_clearance=.48, hold_confirm=.30, retreat_fallback=.08, bounded_urgent=.02)
    return dict(normal=.48, slow_clearance=.30, hold_confirm=.14, retreat_fallback=.06, bounded_urgent=.02)


def world_belief(speed):
    if speed >= 0.25:
        return dict(normal=.08, slow_clearance=.42, hold_confirm=.36, retreat_fallback=.12, bounded_urgent=.02)
    if speed >= 0.05:
        return dict(normal=.25, slow_clearance=.42, hold_confirm=.23, retreat_fallback=.08, bounded_urgent=.02)
    return dict(normal=.45, slow_clearance=.30, hold_confirm=.17, retreat_fallback=.06, bounded_urgent=.02)


def typed_route(contract, abstain):
    if abstain:
        return "hold"
    return {"hold_confirm": "confirm", "retreat_fallback": "retreat"}.get(contract, "admit")


def release_authority(contract, abstain):
    return int(not abstain and contract in {"normal", "slow_clearance", "bounded_urgent"})


def build_evidence(cards, manifest):
    meta = {row["real_case_id"]: row for row in manifest}
    output = []
    for card in cards:
        case = card["real_case_id"]
        hand = wrists(ASSET / "hand_pose_mano" / f"{case}.txt")
        object_id, obj = object_points(ASSET / "obj_pose" / f"{case}.txt")
        obj_rt = transform(ASSET / "obj_pose_rt" / f"{case}.txt", True)
        cam = transform(ASSET / "cam_pose" / f"{case}.txt")
        world_hand, world_obj = apply_transform(cam, hand), apply_transform(cam, obj)
        local_distance = float(np.linalg.norm(hand[:, None] - obj[None, :], axis=2).min())
        world_distance = float(np.linalg.norm(world_hand[:, None] - world_obj[None, :], axis=2).min())
        depth = np.asarray(Image.open(ASSET / "depth" / f"{case}.png"))
        depth_quality = float((depth > 0).mean())
        rt_residual = float(np.linalg.norm(obj[0] - obj_rt[:3, 3]))
        transform_error = rotation_error(cam) + rotation_error(obj_rt) + rt_residual
        world_quality = float(np.clip(math.exp(-25 * transform_error), 0, 1))
        row = {
            "real_case_id": case,
            "sequence_id": meta[case]["sequence_id"],
            "frame_id": int(meta[case]["frame_id"]),
            "action_label": int(meta[case]["action_label"]),
            "object_id": object_id,
            "depth_valid_fraction": depth_quality,
            "local_clearance_m": local_distance,
            "world_clearance_m": world_distance,
            "distance_invariance_error_m": abs(local_distance - world_distance),
            "object_rt_residual_m": rt_residual,
            "local_quality": depth_quality,
            "world_quality": world_quality,
            "world_object_x": float(world_obj[0, 0]),
            "world_object_y": float(world_obj[0, 1]),
            "world_object_z": float(world_obj[0, 2]),
            "source_card_hash": card["source_card_hash"],
        }
        output.append(row)

    groups = defaultdict(list)
    for row in output:
        groups[row["sequence_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["frame_id"])
        for index, row in enumerate(rows):
            point = np.asarray([row["world_object_x"], row["world_object_y"], row["world_object_z"]])
            speeds = []
            for other in rows[max(0, index - 1) : index] + rows[index + 1 : index + 2]:
                other_point = np.asarray([other["world_object_x"], other["world_object_y"], other["world_object_z"]])
                seconds = abs(row["frame_id"] - other["frame_id"]) / 30
                if seconds:
                    speeds.append(float(np.linalg.norm(point - other_point) / seconds))
            row["world_object_speed_mps"] = float(np.median(speeds)) if speeds else 0.0
            row["world_temporal_support"] = len(speeds)
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
            row["evidence_record_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return output


def build_decisions(evidence):
    output = []
    for record in evidence:
        local_ok = record["local_quality"] >= .50
        world_ok = record["world_quality"] >= .95 and record["world_temporal_support"] >= 1
        for command_id, text in COMMANDS:
            language = build_language_source_payload({"command_text": text})
            for setting, label, use_local, use_world in SETTINGS:
                local_missing = not (use_local and local_ok)
                world_missing = not (use_world and world_ok)
                beliefs = [
                    SourceBelief("language", language["probabilities"], float(language["quality"]), float(language["conflict"]), bool(language["missing"])),
                    SourceBelief("vision", local_belief(record["local_clearance_m"]), record["local_quality"] if not local_missing else 0, 0, local_missing),
                    SourceBelief("geometry", world_belief(record["world_object_speed_mps"]), record["world_quality"] if not world_missing else 0, 0, world_missing),
                ]
                blind = fuse_contract_beliefs(beliefs)
                registered = fuse_contract_beliefs(beliefs, dependence_edges={("vision", "geometry"): 1.0}, provenance_gain=1.0)
                if not use_local and not use_world:
                    admitted, route, reason = 0, "fallback", "local_and_world_evidence_unavailable"
                elif not use_local:
                    admitted, route, reason = 0, "confirm", "local_interaction_evidence_unavailable"
                elif not use_world:
                    admitted, route, reason = 0, "hold", "world_frame_evidence_unavailable"
                elif local_missing or world_missing:
                    admitted, route, reason = 0, "hold", "source_state_incomplete"
                else:
                    admitted = int(not registered.abstain)
                    route = typed_route(registered.contract, registered.abstain)
                    reason = "source_state_supported" if admitted else registered.reason
                output.append({
                    "decision_id": f"{record['real_case_id']}__{command_id}__{setting}",
                    "real_case_id": record["real_case_id"], "sequence_id": record["sequence_id"],
                    "frame_id": record["frame_id"], "overlay_command_id": command_id,
                    "setting": setting, "setting_label": label,
                    "local_available": int(not local_missing), "world_available": int(not world_missing),
                    "raw_blind_contract": blind.contract, "raw_blind_confidence": round(blind.confidence, 8),
                    "raw_blind_admitted": int(not blind.abstain),
                    "raw_blind_release_authority": release_authority(blind.contract, blind.abstain),
                    "raw_registered_contract": registered.contract, "raw_registered_confidence": round(registered.confidence, 8),
                    "raw_registered_admitted": int(not registered.abstain),
                    "raw_registered_release_authority": release_authority(registered.contract, registered.abstain),
                    "pact_sv_admitted": admitted, "pact_sv_route": route,
                    "pact_sv_release_authority": int(admitted and route == "admit"), "pact_sv_reason": reason,
                    "local_clearance_m": round(record["local_clearance_m"], 8),
                    "world_object_speed_mps": round(record["world_object_speed_mps"], 8),
                    "local_quality": round(record["local_quality"], 8), "world_quality": round(record["world_quality"], 8),
                    "lineage_edge": "vision--geometry:same_h2o_frame_annotation",
                    "evidence_record_hash": record["evidence_record_hash"],
                })
    return output


def summarize(rows):
    output = []
    for setting, label, _, _ in SETTINGS:
        selected = [row for row in rows if row["setting"] == setting]
        routes, n = Counter(row["pact_sv_route"] for row in selected), len(selected)
        output.append({
            "setting": setting, "setting_label": label, "n": n,
            "raw_blind_admission_rate": sum(row["raw_blind_admitted"] for row in selected) / n,
            "raw_registered_admission_rate": sum(row["raw_registered_admitted"] for row in selected) / n,
            "pact_sv_admission_rate": sum(row["pact_sv_admitted"] for row in selected) / n,
            "pact_sv_release_authority_rate": sum(row["pact_sv_release_authority"] for row in selected) / n,
            **{f"pact_sv_{route}_rate": routes[route] / n for route in ROUTES},
            "mean_registered_confidence": sum(row["raw_registered_confidence"] for row in selected) / n,
        })
    return output


def monotonicity(rows):
    indexed = {(row["real_case_id"], row["overlay_command_id"], row["setting"]): row for row in rows}
    identities = sorted({(row["real_case_id"], row["overlay_command_id"]) for row in rows})
    output = []
    for method in ("raw_blind", "raw_registered", "pact_sv"):
        for setting, _, _, _ in SETTINGS[1:]:
            pairs = [(indexed[case, command, "ego_plus_world"], indexed[case, command, setting]) for case, command in identities]
            output.append({
                "method": method, "removed_to_setting": setting, "comparisons": len(pairs),
                "admission_nonincrease_violations": sum(int(candidate[f"{method}_admitted"] > base[f"{method}_admitted"]) for base, candidate in pairs),
                "release_authority_nonincrease_violations": sum(int(candidate[f"{method}_release_authority"] > base[f"{method}_release_authority"]) for base, candidate in pairs),
            })
    return output


def draw(summary):
    labels, x, width = [row["setting_label"] for row in summary], np.arange(4), .24
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].bar(x - width, [row["raw_blind_admission_rate"] for row in summary], width, label="Dependence blind", color="#A8ADB4")
    axes[0].bar(x, [row["raw_registered_admission_rate"] for row in summary], width, label="Registered lineage", color="#4C78A8")
    axes[0].bar(x + width, [row["pact_sv_admission_rate"] for row in summary], width, label="Evidence-verified", color="#2E8B57")
    axes[0].set_title("a  Admission under evidence-role removal", loc="left", fontweight="bold")
    axes[0].set_ylabel("Contract-admission rate")
    axes[0].legend(frameon=False, fontsize=8)
    colors = {"admit": "#2E8B57", "hold": "#E69F00", "confirm": "#56B4E9", "retreat": "#CC79A7", "fallback": "#6B6B6B"}
    bottom = np.zeros(4)
    for route in ROUTES:
        vals = np.asarray([row[f"pact_sv_{route}_rate"] for row in summary])
        axes[1].bar(x, vals, bottom=bottom, label=route.capitalize(), color=colors[route], width=.64)
        bottom += vals
    axes[1].set_title("b  Typed response to missing evidence roles", loc="left", fontweight="bold")
    axes[1].set_ylabel("Evidence-verified decision share")
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", color="#D9D9D9", linewidth=.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "h2o_egocentric_allocentric_admission.pdf", bbox_inches="tight")
    fig.savefig(FIG / "h2o_egocentric_allocentric_admission.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    cards = read_csv(IN / "h2o_real_source_cards_60_diverse.csv")
    manifest = read_csv(IN / "h2o_real_source_manifest_60_diverse.csv")
    if len(cards) != 60 or len(manifest) != 60 or any(FORBIDDEN & set(row) for row in cards):
        raise RuntimeError("invalid or contaminated H2O denominator")
    evidence = build_evidence(cards, manifest)
    decisions = build_decisions(evidence)
    summary, violations = summarize(decisions), monotonicity(decisions)
    write_csv(OUT / "h2o_egocentric_allocentric_evidence.csv", evidence)
    write_csv(OUT / "h2o_egocentric_allocentric_decisions.csv", decisions)
    write_csv(OUT / "h2o_egocentric_allocentric_summary.csv", summary)
    write_csv(OUT / "h2o_view_removal_monotonicity.csv", violations)
    draw(summary)
    invariant_error = max(row["distance_invariance_error_m"] for row in evidence)
    raw_violations = sum(row["admission_nonincrease_violations"] for row in violations if row["method"] == "raw_registered")
    sv_violations = sum(row["admission_nonincrease_violations"] for row in violations if row["method"] == "pact_sv")
    gate = {
        "status": "pass" if len(decisions) == 1200 and invariant_error < 1e-6 and sv_violations == 0 else "fail",
        "denominator": {"h2o_records": 60, "commands": 5, "settings": 4, "decisions": len(decisions)},
        "available_local_payload": "head-mounted RGB-D and camera-frame hand-object geometry",
        "allocentric_payload": "camera-pose-transformed world-frame geometry",
        "not_available": "a second fixed-camera RGB-D image stream",
        "registered_lineage": "both roles descend from the same H2O frame annotation",
        "maximum_distance_invariance_error_m": invariant_error,
        "raw_registered_admission_nonincrease_violations": raw_violations,
        "pact_sv_admission_nonincrease_violations": sv_violations,
        "claim_boundary": "Real egocentric--allocentric evidence-role admission, not a two-camera, contract-accuracy, physical-robot, natural-command, or participant validation.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage3_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (OUT / "stage3_report.md").write_text(
        f"# Stage 3: real egocentric--allocentric evidence-role admission\n\n"
        f"Status: **{gate['status'].upper()}**\n\n"
        "The frozen denominator contains 60 H2O head-mounted RGB-D records, five controlled command overlays, and four evidence-role settings (1,200 decisions). "
        "The world-frame role is derived from dataset-native camera poses and shares the same frame-level parent as the egocentric role; it is therefore registered as dependent evidence. "
        "The local files do not contain a fixed-camera image stream, so the study is a coordinate-role ablation rather than a two-camera experiment.\n\n"
        f"Maximum rigid-transform distance error: {invariant_error:.3e} m. Raw registered admission non-increase violations: {raw_violations}. Evidence-verified violations: {sv_violations}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": gate, "summary": summary, "violations": violations}, indent=2))
    if gate["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
