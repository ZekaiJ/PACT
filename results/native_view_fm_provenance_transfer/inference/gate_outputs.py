#!/usr/bin/env python3
"""Validate and gate the frozen native-view development assay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SURFACES = ("canonical", "evidence_first", "criterion_first", "compact")
VIEWS = ("front_view", "left_wrist_view", "right_wrist_view", "human_front_view", "exo_view")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fuse(rows: list[dict], registered: bool) -> tuple[float, float]:
    components = defaultdict(list)
    for row in rows:
        key = row["physical_view_id"] if registered else row["id"]
        components[key].append(row["evidence"])
    evidence = [0.0, 0.0]
    for vectors in components.values():
        evidence[0] += min(vector[0] for vector in vectors)
        evidence[1] += min(vector[1] for vector in vectors)
    total = sum(evidence)
    return 1.0 - 2.0 / (2.0 + total), (evidence[1] + 1.0) / (total + 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assay", type=Path, default=ROOT / "outputs" / "qwen3vl_32b.jsonl.gz")
    parser.add_argument("--environment", type=Path, default=ROOT / "outputs" / "qwen3vl_32b_environment.json")
    parser.add_argument("--model-role", choices=("primary", "replication"), default="primary")
    parser.add_argument("--output", type=Path, default=ROOT / "gates" / "ASSAY32_GATE_RECHECK.json")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "ANALYSIS_LOCK.json").read_text(encoding="utf-8"))
    prompts = {
        row["id"]: row
        for row in load_jsonl(ROOT / "protocol" / "test" / "prompt_pack.jsonl.gz")
    }
    assay_path = args.assay.resolve()
    rows = load_jsonl(assay_path)
    expected_model = next(model for model in protocol["models"] if model["role"] == args.model_role)

    errors: list[str] = []
    ids = [row.get("id") for row in rows]
    if len(rows) != len(prompts):
        errors.append(f"row count {len(rows)} != expected {len(prompts)}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate output ids")
    if set(ids) != set(prompts):
        errors.append("output id set differs from frozen prompt pack")

    valid = 0
    traceable = 0
    enriched: list[dict] = []
    for row in rows:
        prompt = prompts.get(row.get("id"))
        mappings = {item.get("mapping"): item for item in row.get("mapping_logits", [])}
        finite_logits = set(mappings) == {"map0", "map1"} and all(
            math.isfinite(float(item.get(name, math.nan)))
            for item in mappings.values()
            for name in ("logit_A", "logit_B")
        )
        p = float(row.get("p_ready", math.nan))
        evidence = row.get("evidence", [])
        evidence_ok = (
            len(evidence) == 2
            and all(math.isfinite(float(value)) for value in evidence)
            and math.isclose(float(evidence[0]), 2.0 * (1.0 - p), abs_tol=1e-12)
            and math.isclose(float(evidence[1]), 2.0 * p, abs_tol=1e-12)
        )
        row_valid = (
            row.get("status") == "ok"
            and finite_logits
            and math.isfinite(p)
            and 0.0 <= p <= 1.0
            and evidence_ok
            and row.get("model_id") == expected_model["id"]
            and row.get("revision") == expected_model["revision"]
        )
        valid += int(row_valid)
        if prompt is None:
            continue
        parent = hashlib.sha256(
            f'{row["event_id"]}:{row["window"]}:{row["physical_view_id"]}'.encode()
        ).hexdigest()
        fields = (
            "episode_id",
            "event_id",
            "image_sha256",
            "parent_id",
            "physical_view_id",
            "prompt_record_sha256",
            "reference_ready",
            "surface_id",
            "window",
        )
        row_traceable = all(row.get(field) == prompt.get(field) for field in fields) and row["parent_id"] == parent
        traceable += int(row_traceable)
        enriched.append({**row, "task_id": prompt["task_id"]})

    valid_rate = valid / len(rows) if rows else 0.0
    traceability_rate = traceable / len(rows) if rows else 0.0

    by_task_window: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in enriched:
        by_task_window[(row["task_id"], row["window"])].append(float(row["p_ready"]))
    task_deltas = {}
    for task in sorted({row["task_id"] for row in enriched}):
        early = by_task_window[(task, "early")]
        ready = by_task_window[(task, "ready")]
        task_deltas[task] = sum(ready) / len(ready) - sum(early) / len(early)
    positive_tasks = sum(delta > 0.0 for delta in task_deltas.values())

    units: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in enriched:
        units[(row["event_id"], row["window"])].append(row)
    score_ranges: dict[str, dict[str, float]] = {}
    task_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit_rows in units.values():
        if len(unit_rows) != len(SURFACES) * len(VIEWS):
            errors.append(f'incomplete unit {unit_rows[0]["event_id"]}:{unit_rows[0]["window"]}')
            continue
        task = unit_rows[0]["task_id"]
        task_scores[task]["acquisition_registered"].append(fuse(unit_rows, True)[0])
        task_scores[task]["lineage_unaware"].append(fuse(unit_rows, False)[0])
    nonconstant_tasks = 0
    for task, arm_scores in sorted(task_scores.items()):
        score_ranges[task] = {arm: max(values) - min(values) for arm, values in arm_scores.items()}
        nonconstant_tasks += int(score_ranges[task]["acquisition_registered"] > 1e-12)

    grid = analysis["primary_estimand"]["coverage_grid"]
    support_width = max(grid) - min(grid)
    gates = protocol["assay_gates"]
    checks = {
        "valid_mapping_rate": valid_rate >= gates["valid_mapping_rate_min"],
        "parent_traceability_rate": traceability_rate >= gates["parent_traceability_rate"],
        "positive_early_to_ready_tasks": positive_tasks >= gates["positive_early_to_ready_tasks_min"],
        "primary_nonconstant_tasks": nonconstant_tasks >= gates["primary_nonconstant_tasks_min"],
        "common_support_width": support_width >= gates["common_support_width_min"],
        "integrity": not errors,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "interpretation": {
            "primary_nonconstant": "the locked m=4 acquisition-registered PACT selection score varies within task",
            "common_support": "locked fractional-tie coverage grid",
            "scope": "output-integrity gate only; no comparative test endpoint computed",
        },
        "observed": {
            "rows": len(rows),
            "unique_ids": len(set(ids)),
            "valid_mapping_rate": valid_rate,
            "parent_traceability_rate": traceability_rate,
            "positive_early_to_ready_tasks": positive_tasks,
            "primary_nonconstant_tasks": nonconstant_tasks,
            "common_support_width": support_width,
            "task_mean_p_ready_delta": task_deltas,
            "task_primary_score_ranges": score_ranges,
            "errors": errors,
        },
        "hashes": {
            assay_path.name: sha256(assay_path),
            args.environment.name: sha256(args.environment.resolve()),
            "prompt_pack.jsonl.gz": sha256(ROOT / "protocol" / "test" / "prompt_pack.jsonl.gz"),
            "PROTOCOL_LOCK.json": sha256(ROOT / "PROTOCOL_LOCK.json"),
            "ANALYSIS_LOCK.json": sha256(ROOT / "ANALYSIS_LOCK.json"),
        },
    }
    output = args.output.resolve()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

