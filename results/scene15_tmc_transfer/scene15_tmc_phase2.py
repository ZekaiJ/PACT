#!/usr/bin/env python3
"""Frozen-output topology interventions for the gated Scene15--TMC pair."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_topology_module(path: Path):
    spec = importlib.util.spec_from_file_location("pact_topology_interventions", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hierarchical_bootstrap(rows: list[dict], draws: int, seed: int) -> list[dict]:
    by_seed_arm = defaultdict(dict)
    for row in rows:
        by_seed_arm[(row["seed"], row["arm"])][row["record_id"]] = row
    seeds = sorted({row["seed"] for row in rows})
    arms = sorted({row["arm"] for row in rows if row["arm"] != "A0_native"})
    rng = np.random.default_rng(seed)
    output = []
    for arm in arms:
        for metric in ("evidence_budget", "component_count"):
            deltas = {}
            for train_seed in seeds:
                arm_rows = by_seed_arm[(train_seed, arm)]
                native_rows = by_seed_arm[(train_seed, "A0_native")]
                ids = sorted(set(arm_rows) & set(native_rows))
                deltas[train_seed] = np.asarray(
                    [arm_rows[record_id][metric] - native_rows[record_id][metric] for record_id in ids],
                    dtype=float,
                )
            estimate = float(np.mean([values.mean() for values in deltas.values()]))
            sampled = []
            for _ in range(draws):
                sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                seed_estimates = []
                for sampled_seed in sampled_seeds:
                    values = deltas[int(sampled_seed)]
                    seed_estimates.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
                sampled.append(float(np.mean(seed_estimates)))
            output.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "outer_seeds": len(seeds),
                    "instances_per_seed": sorted({len(values) for values in deltas.values()}),
                    "delta": estimate,
                    "ci_low": float(np.quantile(sampled, 0.025)),
                    "ci_high": float(np.quantile(sampled, 0.975)),
                    "draws": draws,
                    "bootstrap": "two-stage seed-then-instance paired bootstrap",
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-script", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--multiplicities", nargs="+", type=int, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    topology = load_topology_module(args.topology_script)
    topology.self_test()
    source_rows = []
    for seed in args.seeds:
        export = args.native_root / f"seed_{seed}" / "exports" / f"seed_{seed}.jsonl.gz"
        with gzip.open(export, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["seed"] = seed
                source_rows.append(row)
    if not source_rows:
        raise ValueError("no frozen Scene15--TMC outputs")

    records = []
    for row in source_rows:
        native = [np.asarray(values, dtype=float) for values in row["evidences"]]
        donor = int(row["record_id"]) % len(native)
        distinct = [f"native-view:{index}" for index in range(len(native))]
        generated = [("A0_native", native, distinct)]
        generated.append(("A3_false_merge", native, ["merged-parent"] * len(native)))
        for multiplicity in args.multiplicities:
            exact = native + [native[donor].copy() for _ in range(multiplicity - 1)]
            generated.append(
                (
                    f"A1_conserved_exact_m{multiplicity}",
                    exact,
                    distinct + [distinct[donor]] * (multiplicity - 1),
                )
            )
            generated.append(
                (
                    f"A2_false_split_m{multiplicity}",
                    exact,
                    distinct + [f"false-split:{index}" for index in range(multiplicity - 1)],
                )
            )

        baseline = None
        for arm, evidences, parents in generated:
            output = topology.pact(evidences, parents)
            if arm == "A0_native":
                baseline = output
            assert baseline is not None
            records.append(
                {
                    "pair_id": "Scene15__TMC",
                    "seed": int(row["seed"]),
                    "record_id": int(row["record_id"]),
                    "y": int(row["y"]),
                    "arm": arm,
                    "method": "PACT",
                    "posterior": output["posterior"].tolist(),
                    "score": float(output["score"]),
                    "component_count": int(output["component_count"]),
                    "evidence_budget": float(output["evidence_budget"]),
                    "posterior_l1_from_A0": float(
                        np.abs(output["posterior"] - baseline["posterior"]).sum()
                    ),
                    "score_drift_from_A0": float(output["score"] - baseline["score"]),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = topology.summarize(records)
    bootstrap = hierarchical_bootstrap(records, args.draws, 20260726)
    (args.output / "RESULTS.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap[0]))
        writer.writeheader()
        writer.writerows(bootstrap)

    exact = [row for row in summary if row["arm"].startswith("A1_")]
    split_budget = [
        row for row in bootstrap
        if row["metric"] == "evidence_budget" and row["arm"].startswith("A2_")
    ]
    merge_budget = next(
        row for row in bootstrap
        if row["metric"] == "evidence_budget" and row["arm"] == "A3_false_merge"
    )
    checks = {
        "exact_copy_posterior_invariance": all(
            row["mean_posterior_l1_from_A0"] <= 1e-9 for row in exact
        ),
        "exact_copy_budget_invariance": all(
            abs(row["mean_evidence_budget"] - summary[0]["mean_evidence_budget"]) <= 1e-9
            for row in exact
        ),
        "false_split_budget_increase": all(row["ci_low"] > 0 for row in split_budget),
        "all_view_merge_budget_compression": merge_budget["ci_high"] < 0,
    }
    verdict = "PASS" if all(checks.values()) else "PARTIAL"
    payload = {
        "schema_version": "scene15-tmc-topology-1.0",
        "pair_id": "Scene15__TMC",
        "verdict": verdict,
        "seeds": args.seeds,
        "multiplicities": args.multiplicities,
        "source_instances": len(source_rows),
        "checks": checks,
    }
    (args.output / "verdict.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
