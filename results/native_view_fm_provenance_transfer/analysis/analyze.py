#!/usr/bin/env python3
"""Analyze the frozen native-view provenance transfer experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
EVIDENCE_PER_OUTPUT = 2.0
ARMS = (
    "acquisition_registered",
    "lineage_unaware",
    "shuffled_equal_cardinality",
    "exact_dedup",
    "all_view_merge",
    "native_one_per_view",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def posterior(evidence: np.ndarray, budget: float | None = None) -> tuple[float, float, float]:
    semantic_budget = float(evidence.sum()) if budget is None else float(budget)
    return (
        float((evidence[1] + 1.0) / (semantic_budget + 2.0)),
        1.0 - 2.0 / (semantic_budget + 2.0),
        semantic_budget,
    )


def component_sum(groups: list[list[np.ndarray]]) -> np.ndarray:
    return sum((np.min(np.stack(group), axis=0) for group in groups), start=np.zeros(2))


def shuffled_groups(
    selected: list[dict], active_surfaces: list[str]
) -> list[list[np.ndarray]]:
    views = sorted({row["physical_view_id"] for row in selected})
    lookup = {
        (row["physical_view_id"], row["surface_id"]): np.asarray(row["evidence"], dtype=float)
        for row in selected
    }
    return [
        [
            lookup[(views[(group_index + surface_index) % len(views)], surface)]
            for surface_index, surface in enumerate(active_surfaces)
        ]
        for group_index in range(len(views))
    ]


def arm_outputs(rows: list[dict], m: int, surface_order: list[str]) -> dict[str, dict[str, float]]:
    active_surfaces = surface_order[:m]
    allowed = set(active_surfaces)
    surface_rank = {surface: index for index, surface in enumerate(surface_order)}
    selected = sorted(
        (row for row in rows if row["surface_id"] in allowed),
        key=lambda row: (row["physical_view_id"], surface_rank[row["surface_id"]]),
    )
    canonical = [row for row in selected if row["surface_id"] == "canonical"]
    by_view: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in selected:
        by_view[row["physical_view_id"]].append(np.asarray(row["evidence"], dtype=float))
    vectors = [np.asarray(row["evidence"], dtype=float) for row in selected]
    native_vectors = [np.asarray(row["evidence"], dtype=float) for row in canonical]
    unique = list(dict.fromkeys(tuple(float(x) for x in vector) for vector in vectors))

    registered = np.sum(vectors, axis=0) if m == 1 else component_sum(list(by_view.values()))
    shuffled = component_sum(shuffled_groups(selected, active_surfaces))
    evidence = {
        "acquisition_registered": registered,
        "lineage_unaware": np.sum(vectors, axis=0),
        "shuffled_equal_cardinality": shuffled,
        "exact_dedup": np.sum(np.asarray(unique), axis=0),
        "all_view_merge": np.min(np.stack(vectors), axis=0),
        "native_one_per_view": np.sum(native_vectors, axis=0),
    }
    semantic_budgets = {
        "acquisition_registered": EVIDENCE_PER_OUTPUT * len(by_view) if m == 1 else float(registered.sum()),
        "lineage_unaware": EVIDENCE_PER_OUTPUT * len(vectors),
        "shuffled_equal_cardinality": float(shuffled.sum()),
        "exact_dedup": EVIDENCE_PER_OUTPUT * len(unique),
        "all_view_merge": float(evidence["all_view_merge"].sum()),
        "native_one_per_view": EVIDENCE_PER_OUTPUT * len(native_vectors),
    }
    native_p, _, native_budget = posterior(
        evidence["native_one_per_view"], semantic_budgets["native_one_per_view"]
    )
    output = {}
    for arm, value in evidence.items():
        p_ready, score, budget = posterior(value, semantic_budgets[arm])
        output[arm] = {
            "p_ready": p_ready,
            "score": score,
            "common_score": max(p_ready, 1.0 - p_ready),
            "budget": budget,
            "budget_ratio": budget / native_budget,
            "posterior_l1": 2.0 * abs(p_ready - native_p),
            "retained": float(len(unique)) if arm == "exact_dedup" else math.nan,
        }
    return output


def risk_at(scores: np.ndarray, wrong: np.ndarray, coverage: float) -> float:
    order = np.argsort(-scores, kind="stable")
    scores, wrong = scores[order], wrong[order].astype(float)
    boundaries = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(scores)]
    counts = np.diff(boundaries).astype(float)
    errors = np.add.reduceat(wrong, boundaries[:-1])
    cumulative = np.cumsum(counts)
    target = coverage * len(scores)
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(counts) - 1)
    before_n = 0.0 if index == 0 else cumulative[index - 1]
    before_e = 0.0 if index == 0 else float(np.cumsum(errors)[index - 1])
    fraction = (target - before_n) / counts[index]
    return (before_e + fraction * errors[index]) / target


def normalized_aurc(scores: np.ndarray, wrong: np.ndarray, grid: np.ndarray) -> float:
    risks = np.asarray([risk_at(scores, wrong, float(point)) for point in grid])
    return float(np.trapezoid(risks, grid) / (grid[-1] - grid[0]))


def metrics(values: dict[str, np.ndarray], indices: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    p = values["p_ready"][indices]
    y = values["y"][indices]
    scores = values["score"][indices]
    common_scores = values["common_score"][indices]
    pred = p > 0.5
    wrong = pred != y
    confidence = np.maximum(p, 1.0 - p)
    correctness = (~wrong).astype(float)
    order = np.argsort(confidence, kind="stable")
    ece = 0.0
    for bin_indices in np.array_split(order, min(10, len(order))):
        if len(bin_indices):
            ece += len(bin_indices) / len(order) * abs(
                float(correctness[bin_indices].mean() - confidence[bin_indices].mean())
            )
    admission_risk = risk_at(scores, wrong, 0.13)
    return {
        "ncsAURC": normalized_aurc(scores, wrong, grid),
        "ncsAURC_common_confidence": normalized_aurc(common_scores, wrong, grid),
        "ncsAURC_random": normalized_aurc(np.ones(len(scores)), wrong, grid),
        "accuracy": float((~wrong).mean()),
        "nll": float(-np.log(np.clip(np.where(y == 1, p, 1.0 - p), 1e-12, 1.0)).mean()),
        "brier": float((2.0 * np.square(p - y)).mean()),
        "ece10": float(ece),
        "wrong_admission_all_at_0.13": float(0.13 * admission_risk),
        "wrong_admission_conditional_at_0.13": float(admission_risk),
        "budget_ratio": float(values["budget_ratio"][indices].mean()),
        "posterior_l1": float(values["posterior_l1"][indices].mean()),
        "exact_dedup_retained": float(np.nanmean(values["retained"][indices]))
        if np.isfinite(values["retained"][indices]).any()
        else math.nan,
    }


def contrast(
    points: list[dict],
    draws: dict[tuple[tuple[int, str], str], list[float]],
    m: int,
    metric: str,
) -> dict[str, float]:
    point = next(
        row["estimate"]
        for row in points
        if row["m"] == m and row["arm"] == "lineage_unaware" and row["metric"] == metric
    ) - next(
        row["estimate"]
        for row in points
        if row["m"] == m and row["arm"] == "acquisition_registered" and row["metric"] == metric
    )
    values = np.asarray(draws[((m, "lineage_unaware"), metric)]) - np.asarray(
        draws[((m, "acquisition_registered"), metric)]
    )
    return {
        "m": m,
        "estimate": float(point),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_draw_fraction": float(np.mean(values > 0)),
    }


def self_test() -> None:
    assert math.isclose(risk_at(np.ones(4), np.asarray([0, 1, 0, 1]), 0.5), 0.5)
    repeated = [
        {
            "physical_view_id": view,
            "surface_id": surface,
            "evidence": [1.5, 0.5],
        }
        for view in ("v1", "v2")
        for surface in ("canonical", "evidence_first")
    ]
    result = arm_outputs(repeated, 2, ["canonical", "evidence_first"])
    assert result["acquisition_registered"]["budget"] == 4.0
    assert result["lineage_unaware"]["budget"] == 8.0
    assert result["native_one_per_view"]["budget"] == 4.0
    assert result["exact_dedup"]["budget"] == 2.0
    m1 = arm_outputs(repeated, 1, ["canonical", "evidence_first"])
    assert m1["acquisition_registered"] == m1["lineage_unaware"]
    assert arm_outputs(list(reversed(repeated)), 2, ["canonical", "evidence_first"])[
        "lineage_unaware"
    ]["score"] == result["lineage_unaware"]["score"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--model-role", choices=("primary", "replication"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("pack", "outputs", "environment", "model_role", "output_dir") if getattr(args, name) is None]
    if missing:
        parser.error("required outside --self-test: " + ", ".join("--" + name.replace("_", "-") for name in missing))

    analysis = json.loads((EXPERIMENT / "ANALYSIS_LOCK.json").read_text(encoding="utf-8-sig"))
    amendment_path = EXPERIMENT / "ANALYSIS_AMENDMENT_TIE_V1.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8-sig"))
    if amendment.get("status") != "LOCKED_ANALYSIS_CORRECTION":
        raise RuntimeError("analysis correction is not locked")
    test_lock = json.loads((EXPERIMENT / "TEST_ANALYSIS_LOCK.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((args.pack / "PROTOCOL_LOCK.json").read_text(encoding="utf-8-sig"))
    expected_model = next(model for model in protocol["models"] if model["role"] == args.model_role)
    prompt_pack = args.pack / "prompt_pack.jsonl.gz"
    if not prompt_pack.exists():
        prompt_pack = args.pack / "prompt_pack.jsonl"
    prompts = {row["id"]: row for row in jsonl(prompt_pack)}
    outputs = jsonl(args.outputs)
    output_ids = [row.get("id") for row in outputs]
    errors = []
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(prompts):
        errors.append("output IDs are not a one-to-one match to the frozen prompt pack")
    enriched = []
    traceable = 0
    for row in outputs:
        prompt = prompts.get(row.get("id"))
        if prompt is None:
            continue
        parent = hashlib.sha256(
            f'{row["event_id"]}:{row["window"]}:{row["physical_view_id"]}'.encode()
        ).hexdigest()
        fields = (
            "episode_id",
            "event_id",
            "window",
            "physical_view_id",
            "surface_id",
            "parent_id",
            "image_sha256",
            "prompt_record_sha256",
            "reference_ready",
        )
        traceable += int(
            all(row.get(field) == prompt.get(field) for field in fields) and parent == row["parent_id"]
        )
        mappings = {item.get("mapping"): item for item in row.get("mapping_logits", [])}
        logits_ok = set(mappings) == {"map0", "map1"} and all(
            math.isfinite(float(item.get(name, math.nan)))
            for item in mappings.values()
            for name in ("logit_A", "logit_B")
        )
        if logits_ok:
            ready_log_odds = 0.5 * (
                mappings["map0"]["logit_B"]
                - mappings["map0"]["logit_A"]
                + mappings["map1"]["logit_A"]
                - mappings["map1"]["logit_B"]
            )
            expected_p = 1.0 / (1.0 + math.exp(-ready_log_odds))
        else:
            expected_p = math.nan
        p_ready = float(row.get("p_ready", math.nan))
        evidence = row.get("evidence", [])
        valid = (
            row.get("status") == "ok"
            and row.get("model_id") == expected_model["id"]
            and row.get("revision") == expected_model["revision"]
            and logits_ok
            and math.isfinite(p_ready)
            and math.isclose(p_ready, expected_p, abs_tol=1e-12)
            and len(evidence) == 2
            and math.isclose(float(evidence[0]), EVIDENCE_PER_OUTPUT * (1.0 - p_ready), abs_tol=1e-12)
            and math.isclose(float(evidence[1]), EVIDENCE_PER_OUTPUT * p_ready, abs_tol=1e-12)
        )
        if not valid:
            errors.append(f'invalid output {row.get("id")}')
        enriched.append({**row, "task_id": prompt["task_id"]})
    if traceable != len(outputs):
        errors.append("parent or prompt traceability is incomplete")
    if errors:
        raise RuntimeError("; ".join(errors[:10]))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in enriched:
        grouped[(row["event_id"], row["window"])].append(row)
    units = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 20:
            raise RuntimeError(f"incomplete event-window unit {key}: {len(rows)}")
        units.append(
            {
                "event_id": key[0],
                "window": key[1],
                "episode_id": rows[0]["episode_id"],
                "task_id": rows[0]["task_id"],
                "y": int(rows[0]["reference_ready"]),
                "rows": rows,
            }
        )

    surface_order = analysis["multiplicity"]["surface_order"]
    arrays = {}
    record_rows = []
    for m in analysis["multiplicity"]["mechanism_series"]:
        per_arm = {arm: defaultdict(list) for arm in ARMS}
        for unit in units:
            results = arm_outputs(unit["rows"], int(m), surface_order)
            for arm, result in results.items():
                for name, value in result.items():
                    per_arm[arm][name].append(value)
                per_arm[arm]["y"].append(unit["y"])
                record_rows.append(
                    {
                        "event_id": unit["event_id"],
                        "window": unit["window"],
                        "episode_id": unit["episode_id"],
                        "task_id": unit["task_id"],
                        "m": m,
                        "arm": arm,
                        "y": unit["y"],
                        **result,
                    }
                )
        for arm in ARMS:
            arrays[int(m), arm] = {
                name: np.asarray(value) for name, value in per_arm[arm].items()
            }

    tasks = sorted({unit["task_id"] for unit in units})
    task_indices = {
        task: np.asarray([i for i, unit in enumerate(units) if unit["task_id"] == task])
        for task in tasks
    }
    grid = np.asarray(analysis["primary_estimand"]["coverage_grid"], dtype=float)
    points = []
    for m in analysis["multiplicity"]["mechanism_series"]:
        for arm in ARMS:
            task_values = [metrics(arrays[int(m), arm], indices, grid) for indices in task_indices.values()]
            for metric in task_values[0]:
                observed = np.asarray([value[metric] for value in task_values], dtype=float)
                estimate = float(np.nanmean(observed)) if np.isfinite(observed).any() else math.nan
                points.append({"m": m, "arm": arm, "metric": metric, "estimate": estimate})

    episode_indices = {}
    for task, indices in task_indices.items():
        episode_indices[task] = defaultdict(list)
        for index in indices:
            episode_indices[task][units[int(index)]["episode_id"]].append(int(index))
    rng = np.random.default_rng(int(test_lock["bootstrap_seed"]))
    draws: dict[tuple[tuple[int, str], str], list[float]] = defaultdict(list)
    selected = [(4, arm) for arm in ARMS] + [
        (m, arm) for m in (1, 2) for arm in ("acquisition_registered", "lineage_unaware")
    ]
    for _ in range(args.draws):
        task_draw = defaultdict(list)
        for task, episodes in episode_indices.items():
            ids = list(episodes)
            sampled = rng.choice(ids, size=len(ids), replace=True)
            indices = np.asarray(
                [index for episode in sampled for index in episodes[str(episode)]], dtype=int
            )
            for key in selected:
                task_draw[key].append(metrics(arrays[key], indices, grid))
        for key, values in task_draw.items():
            for metric in values[0]:
                observed = np.asarray([value[metric] for value in values], dtype=float)
                if np.isfinite(observed).any():
                    draws[(key, metric)].append(float(np.nanmean(observed)))

    intervals = []
    for (key, metric), values in draws.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals.append(
            {
                "m": key[0],
                "arm": key[1],
                "metric": metric,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    primary_contrasts = [contrast(points, draws, m, "ncsAURC") for m in (1, 2, 4)]
    common_score_contrasts = [
        contrast(points, draws, m, "ncsAURC_common_confidence") for m in (1, 2, 4)
    ]

    paired = []
    lower_is_better = {
        "ncsAURC",
        "ncsAURC_common_confidence",
        "ncsAURC_random",
        "nll",
        "brier",
        "ece10",
        "wrong_admission_all_at_0.13",
        "wrong_admission_conditional_at_0.13",
    }
    paired_metrics = (
        "ncsAURC",
        "ncsAURC_common_confidence",
        "ncsAURC_random",
        "accuracy",
        "nll",
        "brier",
        "ece10",
        "wrong_admission_all_at_0.13",
        "wrong_admission_conditional_at_0.13",
    )
    for comparator in ARMS[1:]:
        for metric in paired_metrics:
            registered = next(
                row["estimate"]
                for row in points
                if row["m"] == 4
                and row["arm"] == "acquisition_registered"
                and row["metric"] == metric
            )
            compared = next(
                row["estimate"]
                for row in points
                if row["m"] == 4 and row["arm"] == comparator and row["metric"] == metric
            )
            if metric in lower_is_better:
                values = np.asarray(draws[((4, comparator), metric)]) - np.asarray(
                    draws[((4, "acquisition_registered"), metric)]
                )
                estimate = compared - registered
                definition = "comparator minus acquisition_registered; positive favors registration"
            else:
                values = np.asarray(draws[((4, "acquisition_registered"), metric)]) - np.asarray(
                    draws[((4, comparator), metric)]
                )
                estimate = registered - compared
                definition = "acquisition_registered minus comparator; positive favors registration"
            paired.append(
                {
                    "m": 4,
                    "comparator": comparator,
                    "metric": metric,
                    "estimate": float(estimate),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "positive_draw_fraction": float(np.mean(values > 0)),
                    "definition": definition,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("point_estimates.csv", points),
        ("bootstrap_intervals.csv", intervals),
        ("paired_contrasts.csv", paired),
        ("records.csv", record_rows),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "model": expected_model,
        "status": "COMPLETE_WITH_LOCKED_TIE_CORRECTION",
        "units": len(units),
        "episodes": len({unit["episode_id"] for unit in units}),
        "tasks": len(task_indices),
        "primary_contrasts": primary_contrasts,
        "common_score_contrasts": common_score_contrasts,
        "task_level_m4_contrasts": {
            task: metrics(arrays[4, "lineage_unaware"], indices, grid)["ncsAURC"]
            - metrics(arrays[4, "acquisition_registered"], indices, grid)["ncsAURC"]
            for task, indices in task_indices.items()
        },
        "task_level_m4_common_score_contrasts": {
            task: metrics(arrays[4, "lineage_unaware"], indices, grid)[
                "ncsAURC_common_confidence"
            ]
            - metrics(arrays[4, "acquisition_registered"], indices, grid)[
                "ncsAURC_common_confidence"
            ]
            for task, indices in task_indices.items()
        },
        "hashes": {
            "outputs": content_sha256(args.outputs),
            "environment": sha256(args.environment),
            "prompt_pack": content_sha256(prompt_pack),
            "protocol_lock": sha256(args.pack / "PROTOCOL_LOCK.json"),
            "analysis_lock": sha256(EXPERIMENT / "ANALYSIS_LOCK.json"),
            "analysis_amendment": sha256(amendment_path),
            "test_analysis_lock": sha256(EXPERIMENT / "TEST_ANALYSIS_LOCK.json"),
        },
    }
    (args.output_dir / "PRIMARY_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
