#!/usr/bin/env python3
"""A2: topology corruption, partition recovery, and admission damage.

This analysis reuses the frozen v1851 full-replay inputs and runtime.  It does
not modify the manuscript or retrain any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_v1851_registered_provenance_robustness as v1851  # noqa: E402


OUT = ROOT / "results" / "v1956_granularity_autoresearch" / "a2_lineage_perturbation"
PREREGISTRATION = OUT / "PREREGISTRATION.md"
CONDITIONS = ("false_split", "false_merge", "hidden_edge_deletion", "forged_independence")
RATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = tuple(range(10))
BOOTSTRAP_DRAWS = 1000
TRUE_GRAPH = v1851.GRAPHS["registered"]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def partition_pairs(graph: Mapping[tuple[str, str], float]) -> set[tuple[str, str]]:
    adjacency = {source: set() for source in v1851.SOURCES}
    for (left, right), strength in graph.items():
        if float(strength) > 0:
            adjacency[left].add(right)
            adjacency[right].add(left)
    pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(v1851.SOURCES):
        reached, stack = {left}, [left]
        while stack:
            for right in adjacency[stack.pop()]:
                if right not in reached:
                    reached.add(right)
                    stack.append(right)
        for right in v1851.SOURCES[index + 1 :]:
            if right in reached:
                pairs.add((left, right))
    return pairs


def pairwise_partition_counts(predicted: Mapping[tuple[str, str], float]) -> dict[str, int]:
    truth, observed = partition_pairs(TRUE_GRAPH), partition_pairs(predicted)
    return {
        "partition_tp": len(truth & observed),
        "partition_fp": len(observed - truth),
        "partition_fn": len(truth - observed),
    }


def pairwise_f1(counts: Mapping[str, int]) -> float:
    tp, fp, fn = (int(counts[key]) for key in ("partition_tp", "partition_fp", "partition_fn"))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def rehash_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    body = {key: manifest[key] for key in ("record_id", "schema", "sources")}
    manifest["manifest_sha256"] = v1851.sha256_text(body)
    assert v1851.manifest_valid(manifest)
    return manifest


def corrupted_state(
    manifest: Mapping[str, Any], condition: str, affected: bool
) -> tuple[str, bool, Mapping[tuple[str, str], float]]:
    if not affected:
        return "registered", True, TRUE_GRAPH

    altered = json.loads(json.dumps(manifest))
    if condition == "hidden_edge_deletion":
        # The declaration loses the edge, but the intact registered manifest recovers it.
        state, complete = v1851.policy_state("registered_lineage", {}, True, altered)
    elif condition == "false_split":
        altered["sources"]["risk"]["parents"] = [
            parent if not str(parent).startswith(("scene:", "occlusion:")) else f"split:{parent}"
            for parent in altered["sources"]["risk"]["parents"]
        ]
        state, complete = v1851.policy_state("registered_lineage", {}, True, rehash_manifest(altered))
    elif condition == "false_merge":
        shared = f"false-merge:{altered['record_id']}"
        for source in v1851.SOURCES:
            altered["sources"][source]["parents"].append(shared)
        state, complete = v1851.policy_state("registered_lineage", {}, True, rehash_manifest(altered))
    elif condition == "forged_independence":
        for source in v1851.SOURCES:
            altered["sources"][source] = {
                "pipeline_id": f"forged_{source}",
                "parents": [f"forged:{source}:{altered['record_id']}"],
            }
        state, complete = v1851.policy_state("registered_lineage", {}, True, rehash_manifest(altered))
    else:
        raise ValueError(condition)
    return state, complete, v1851.GRAPHS[state]


def add_counts(target: dict[str, int], decision: Mapping[str, Any], gold: str) -> None:
    v1851.add_count(target, decision, gold)


def evaluate(
    records: list[dict[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
    cache: Mapping[tuple[str, str, bool], Mapping[str, Any]],
    condition: str,
    rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    total = {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
    scenes = defaultdict(lambda: {"n": 0, "admitted": 0, "wrong": 0, "correct": 0})
    partition = {"partition_tp": 0, "partition_fp": 0, "partition_fn": 0}
    seen_clusters: set[str] = set()
    for record in records:
        record_id = str(record["record_id"])
        cluster = v1851.cluster_id(record_id)
        # False splitting and forged independence induce the same all-distinct
        # observed partition, so use the same affected clusters for a direct
        # observability comparison rather than adding Monte Carlo imbalance.
        mask_key = "all_distinct" if condition in {"false_split", "forged_independence"} else condition
        affected = v1851.deterministic_unit(cluster, mask_key, seed) < rate
        state, complete, graph = corrupted_state(manifests[record_id], condition, affected)
        decision = cache[(record_id, state, complete)]
        gold = str(record["gold_contract"])
        add_counts(total, decision, gold)
        add_counts(scenes[v1851.scene_id(record)], decision, gold)
        if cluster not in seen_clusters:
            for key, value in pairwise_partition_counts(graph).items():
                partition[key] += value
            seen_clusters.add(cluster)
    metrics = v1851.metrics(total)
    return {
        "condition": condition,
        "rate": rate,
        "seed": seed,
        "clusters": len(seen_clusters),
        **metrics,
        **partition,
        "partition_pairwise_f1": pairwise_f1(partition),
    }, dict(scenes)


def no_lineage_baseline(
    records: list[dict[str, Any]], cache: Mapping[tuple[str, str, bool], Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    total = {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
    scenes = defaultdict(lambda: {"n": 0, "admitted": 0, "wrong": 0, "correct": 0})
    for record in records:
        decision = cache[(str(record["record_id"]), "independent", True)]
        gold = str(record["gold_contract"])
        add_counts(total, decision, gold)
        add_counts(scenes[v1851.scene_id(record)], decision, gold)
    return {"condition": "no_lineage", "rate": "", "seed": "", **v1851.metrics(total)}, dict(scenes)


def bootstrap_metrics(
    scene_counts: list[Mapping[str, Mapping[str, int]]], seed: int
) -> dict[str, tuple[float, float]]:
    scene_names = sorted(scene_counts[0])
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(BOOTSTRAP_DRAWS):
        source = rng.choice(scene_counts)
        counts = {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
        for _scene in scene_names:
            sampled = source[rng.choice(scene_names)]
            for key in counts:
                counts[key] += int(sampled[key])
        for key, value in v1851.metrics(counts).items():
            if key not in {"n", "admitted", "wrong", "correct"}:
                draws[key].append(float(value))
    return {key: (percentile(values, 0.025), percentile(values, 0.975)) for key, values in draws.items()}


def summarize(
    seed_rows: list[dict[str, Any]],
    scene_rows: Mapping[tuple[str, float], list[Mapping[str, Mapping[str, int]]]],
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["condition"]), float(row["rate"]))].append(row)
    output = []
    for (condition, rate), rows in sorted(grouped.items()):
        intervals = bootstrap_metrics(scene_rows[(condition, rate)], 195600 + int(100 * rate) + CONDITIONS.index(condition))
        item: dict[str, Any] = {
            "condition": condition,
            "rate": rate,
            "seeds": len(rows),
            "records": int(rows[0]["n"]),
            "scenes": len(scene_rows[(condition, rate)][0]),
            "partition_pairwise_f1_mean": statistics.fmean(float(row["partition_pairwise_f1"]) for row in rows),
        }
        for metric in ("coverage", "wrong_all", "wrong_admitted", "correct_all", "expected_cost"):
            item[f"{metric}_mean"] = statistics.fmean(float(row[metric]) for row in rows)
            item[f"{metric}_ci_low"], item[f"{metric}_ci_high"] = intervals[metric]
            item[f"{metric}_delta_vs_no_lineage"] = item[f"{metric}_mean"] - float(baseline[metric])
        output.append(item)
    return output


def crossover(summary: list[dict[str, Any]], baseline: Mapping[str, Any], metric: str) -> list[dict[str, Any]]:
    output = []
    target = float(baseline[metric])
    for condition in CONDITIONS:
        rows = sorted((row for row in summary if row["condition"] == condition), key=lambda row: float(row["rate"]))
        deltas = [float(row[f"{metric}_mean"]) - target for row in rows]
        crossing: float | None = None
        for index, delta in enumerate(deltas):
            if delta >= -1e-12:
                if index == 0 or abs(delta) <= 1e-12:
                    crossing = float(rows[index]["rate"])
                else:
                    previous = deltas[index - 1]
                    low, high = float(rows[index - 1]["rate"]), float(rows[index]["rate"])
                    crossing = low + (high - low) * (-previous) / (delta - previous)
                break
        output.append(
            {
                "condition": condition,
                "metric": metric,
                "no_lineage_value": target,
                "crossover_rate": crossing,
                "status": "identified_on_grid" if crossing is not None else "not_identified_within_grid",
            }
        )
    return output


def draw(summary: list[dict[str, Any]], baseline: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "false_split": "#D55E00",
        "false_merge": "#009E73",
        "hidden_edge_deletion": "#0072B2",
        "forged_independence": "#CC79A7",
    }
    labels = {
        "false_split": "False split",
        "false_merge": "False merge",
        "hidden_edge_deletion": "Hidden-edge deletion",
        "forged_independence": "Forged independence",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    for condition in CONDITIONS:
        rows = sorted((row for row in summary if row["condition"] == condition), key=lambda row: float(row["rate"]))
        x = [100 * float(row["rate"]) for row in rows]
        for axis, metric in zip(axes, ("wrong_all", "coverage")):
            y = [float(row[f"{metric}_mean"]) for row in rows]
            lo = [float(row[f"{metric}_ci_low"]) for row in rows]
            hi = [float(row[f"{metric}_ci_high"]) for row in rows]
            axis.plot(x, y, marker="o", linewidth=1.8, color=colors[condition], label=labels[condition])
            axis.fill_between(x, lo, hi, color=colors[condition], alpha=0.10)
    axes[0].axhline(float(baseline["wrong_all"]), color="#555555", linestyle="--", linewidth=1.2, label="No lineage")
    axes[1].axhline(float(baseline["coverage"]), color="#555555", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Wrong admission / all")
    axes[1].set_ylabel("Coverage")
    for axis in axes:
        axis.set_xlabel("Corrupted scene-cue clusters (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "a2_lineage_perturbation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "a2_lineage_perturbation.pdf", bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    assert pairwise_partition_counts(v1851.GRAPHS["registered"]) == {
        "partition_tp": 1,
        "partition_fp": 0,
        "partition_fn": 0,
    }
    assert pairwise_f1(pairwise_partition_counts(v1851.GRAPHS["independent"])) == 0.0
    assert abs(pairwise_f1(pairwise_partition_counts(v1851.GRAPHS["fully_linked"])) - 0.5) < 1e-12
    manifest = v1851.lineage_manifest({
        "record_id": "check",
        "metadata": {"scene_id": "s", "relation_label": "r", "risk_band": "x", "occlusion_band": "o"},
        "sources": {"language": {"source_route": {"payload_text_hash": "h"}}},
    })
    states = {condition: corrupted_state(manifest, condition, True)[0] for condition in CONDITIONS}
    assert states == {
        "false_split": "independent",
        "false_merge": "fully_linked",
        "hidden_edge_deletion": "registered",
        "forged_independence": "independent",
    }
    print("self-check: PASS")


def main() -> None:
    if not PREREGISTRATION.exists():
        raise RuntimeError(f"missing preregistration: {PREREGISTRATION}")
    records = v1851.read_jsonl(v1851.INPUT)
    if len(records) != 31_200:
        raise RuntimeError(f"expected 31,200 records, found {len(records)}")
    folds = v1851.scene_fold_map(records)
    config_rows = [row for row in v1851.read_csv(v1851.CONFIGS) if row["method"] == "PACT-SV-risk-native-ec-xfit"]
    configs = {int(row["fold"]): row for row in config_rows}
    selected = {int(key): int(value) for key, value in json.loads(v1851.OUT.joinpath("stage2_gate.json").read_text(encoding="utf-8"))["selected_minimum_independent_groups"].items()}
    manifests = {str(record["record_id"]): v1851.lineage_manifest(record) for record in records}
    cache = v1851.precompute(records, configs, folds, selected)
    baseline, baseline_scenes = no_lineage_baseline(records, cache)

    seed_rows: list[dict[str, Any]] = []
    scene_groups: dict[tuple[str, float], list[Mapping[str, Mapping[str, int]]]] = defaultdict(list)
    scene_export = []
    for condition in CONDITIONS:
        for rate in RATES:
            for seed in SEEDS:
                row, scenes = evaluate(records, manifests, cache, condition, rate, seed)
                seed_rows.append(row)
                scene_groups[(condition, rate)].append(scenes)
                for scene, counts in sorted(scenes.items()):
                    scene_export.append({"condition": condition, "rate": rate, "seed": seed, "scene_id": scene, **counts})

    summary = summarize(seed_rows, scene_groups, baseline)
    crossovers = crossover(summary, baseline, "expected_cost") + crossover(summary, baseline, "wrong_all")
    write_csv(OUT / "seed_results.csv", seed_rows)
    write_csv(OUT / "scene_counts.csv", scene_export)
    write_csv(OUT / "risk_coverage_by_corruption.csv", summary)
    write_csv(OUT / "type_specific_crossovers.csv", crossovers)
    write_json(OUT / "no_lineage_baseline.json", {**baseline, "scene_bootstrap": bootstrap_metrics([baseline_scenes], 195601)})
    draw(summary, baseline)

    by_key = {(row["condition"], float(row["rate"])): row for row in summary}
    forged_equivalent = all(
        abs(float(by_key[("false_split", rate)][f"{metric}_mean"]) - float(by_key[("forged_independence", rate)][f"{metric}_mean"])) < 1e-12
        for rate in RATES
        for metric in ("coverage", "wrong_all", "expected_cost", "partition_pairwise_f1")
    )
    hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            v1851.INPUT,
            v1851.CONFIGS,
            v1851.OUT / "stage2_gate.json",
            Path(v1851.__file__).resolve(),
            ROOT / "E3_HRC" / "E3_HRC-main" / "src" / "safe_fuse.py",
            PREREGISTRATION,
            Path(__file__).resolve(),
        )
    }
    hidden_recovered = all(
        abs(float(row[f"{metric}_mean"]) - float(by_key[("hidden_edge_deletion", 0.0)][f"{metric}_mean"])) < 1e-12
        for row in summary
        if row["condition"] == "hidden_edge_deletion"
        for metric in ("coverage", "wrong_all", "expected_cost", "partition_pairwise_f1")
    )
    validation_checks = {
        "output_grid_complete": len(seed_rows) == len(CONDITIONS) * len(RATES) * len(SEEDS),
        "scene_count_complete": all(
            len(group) == len(SEEDS) and all(len(scenes) == 48 for scenes in group)
            for group in scene_groups.values()
        ),
        "hidden_edge_recovered_by_intact_manifest": hidden_recovered,
        "false_split_and_forgery_observationally_equivalent": forged_equivalent,
        "false_split_reaches_no_lineage_at_full_corruption": abs(
            float(by_key[("false_split", 1.0)]["expected_cost_mean"]) - float(baseline["expected_cost"])
        ) < 1e-12,
    }
    result = {
        "version": "v1956-a2",
        "status": "pass" if all(validation_checks.values()) else "fail",
        "records": len(records),
        "scenes": len({v1851.scene_id(record) for record in records}),
        "scene_cue_clusters": len({v1851.cluster_id(str(record["record_id"])) for record in records}),
        "rates": list(RATES),
        "perturbation_seeds": len(SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "no_lineage_baseline": baseline,
        "crossovers": crossovers,
        "false_split_and_forgery_observationally_equivalent": forged_equivalent,
        "validation_checks": validation_checks,
        "forged_independence_scope": "Internally consistent forged independence is unobservable to this interface; the experiment quantifies damage and does not test detection.",
        "input_hashes": hashes,
    }
    write_json(OUT / "result.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"A2 validation failed: {validation_checks}")
    rate_50 = {condition: by_key[(condition, 0.5)] for condition in CONDITIONS}
    report = f"""# A2 lineage-perturbation analysis

Decision: PASS

This frozen full-replay analysis covers {len(records):,} decision records nested in
{result['scenes']} scenes. Corruption is assigned at the {result['scene_cue_clusters']}
scene-cue clusters; uncertainty intervals use {BOOTSTRAP_DRAWS:,} scene-level
bootstrap draws with perturbation-seed resampling.

At 50% corruption, wrong-admission/all was
{float(rate_50['hidden_edge_deletion']['wrong_all_mean']):.4f} after hidden-edge
deletion with an intact registered manifest,
{float(rate_50['false_split']['wrong_all_mean']):.4f} under false splitting,
{float(rate_50['false_merge']['wrong_all_mean']):.4f} under false merging, and
{float(rate_50['forged_independence']['wrong_all_mean']):.4f} under internally
consistent forged independence. The no-lineage value was
{float(baseline['wrong_all']):.4f}.

False splitting and forged independence are numerically identical in this
three-source binary-lineage interface: both induce the all-distinct partition.
This is an observability result, not a detector. In particular, internally
consistent forged independence cannot be detected from the registered interface;
the reported curve quantifies its damage only.

Pairwise partition F1 and type-specific crossovers are reported in
`risk_coverage_by_corruption.csv` and `type_specific_crossovers.csv`. A crossover
is marked unavailable when the corrupted method does not reach the no-lineage
baseline anywhere on the prespecified rate grid.
"""
    (OUT / "A2_REPORT.md").write_text(report, encoding="utf-8")
    self_check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        main()
