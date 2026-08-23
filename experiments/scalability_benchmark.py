#!/usr/bin/env python3
"""CPU latency, allocation, and source-count scaling benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from action_admission.baselines import quality_weighted_prediction
from action_admission.contracts import CONTRACT_CLASSES
from action_admission.dirichlet import predict, restrict_input
from action_admission.lineage import (
    SourceOpinion,
    connected_components,
    graph_from_parent_sets,
    log_linear_posterior,
)
from action_admission.verifier import verify_source_state


SOURCE_COUNTS = (3, 8, 16, 32, 64)
TARGET = "hold_confirm"


def distribution(index: int) -> dict[str, float]:
    peak = 0.72 + 0.02 * (index % 4)
    rest = (1.0 - peak) / (len(CONTRACT_CLASSES) - 1)
    return {label: peak if label == TARGET else rest for label in CONTRACT_CLASSES}


def synthetic_record(count: int) -> tuple[dict, tuple[str, ...], dict]:
    names = tuple(f"source_{index:03d}" for index in range(count))
    sources = {
        name: {
            "probabilities": distribution(index),
            "quality": 0.72 + 0.03 * (index % 5),
            "conflict": 0.02 * (index % 3),
            "missing": False,
        }
        for index, name in enumerate(names)
    }
    graph = {
        (names[index], names[index + 1]): 1.0
        for index in range(0, count - 1, 2)
    }
    return {"sources": sources}, names, graph


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def timed(
    function: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples: list[float] = []
    repeat_medians: list[float] = []
    throughputs: list[float] = []
    gc.disable()
    try:
        for _ in range(repeats):
            block = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                function()
                block.append((time.perf_counter_ns() - start) / 1_000.0)
            samples.extend(block)
            repeat_medians.append(statistics.median(block))
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                function()
            elapsed_seconds = (time.perf_counter_ns() - start) / 1_000_000_000.0
            throughputs.append(iterations / elapsed_seconds)
    finally:
        gc.enable()

    gc.collect()
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for _ in range(min(iterations, 100)):
        function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_us = statistics.median(samples)
    return {
        "median_us": median_us,
        "p95_us": percentile(samples, 0.95),
        "throughput_calls_s": statistics.median(throughputs),
        "throughput_repeat_sd": statistics.stdev(throughputs),
        "repeat_median_sd_us": statistics.stdev(repeat_medians),
        "peak_traced_allocation_kib": max(0, peak - baseline) / 1024.0,
        "timed_calls": len(samples),
    }


def benchmark_source_scaling(args: argparse.Namespace) -> list[dict]:
    rows = []
    for count in SOURCE_COUNTS:
        record, names, graph = synthetic_record(count)
        restricted = restrict_input(record, sources=names)
        functions = {
            "quality_weighted_vote": lambda r=record: quality_weighted_prediction(r),
            "evidential_composition": lambda r=restricted, n=names: predict(
                r, concentration=4.0, sources=n
            ),
            "registered_graph_components": lambda n=names, g=graph: connected_components(n, g),
        }
        for method, function in functions.items():
            result = function()
            if method != "registered_graph_components":
                predicted = getattr(result, "predicted_contract", None)
                if predicted != TARGET:
                    raise AssertionError(f"{method} predicted {predicted!r}")
            row = {"source_count": count, "method": method}
            row.update(
                timed(
                    function,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    iterations=args.iterations,
                )
            )
            rows.append(row)
    return rows


def benchmark_canonical_path(args: argparse.Namespace) -> list[dict]:
    record = json.loads((ROOT / "data/examples/source_record.json").read_text())
    names = tuple(record["sources"])
    graph = graph_from_parent_sets(record["registered_parents"])
    restricted = restrict_input(record, sources=names)
    opinions = [
        SourceOpinion(
            source=name,
            probabilities=record["sources"][name]["probabilities"],
            quality=record["sources"][name]["quality"],
            conflict=record["sources"][name]["conflict"],
        )
        for name in names
    ]

    def verifier():
        return verify_source_state(record, TARGET, graph, lineage_complete=True)

    def end_to_end():
        local_names = tuple(record["sources"])
        local_graph = graph_from_parent_sets(record["registered_parents"])
        candidate = predict(
            restrict_input(record, sources=local_names),
            concentration=4.0,
            sources=local_names,
        )
        return verify_source_state(
            record,
            candidate.predicted_contract,
            local_graph,
            lineage_complete=True,
        )

    functions = {
        "quality_weighted_vote": lambda: quality_weighted_prediction(record),
        "evidential_composition": lambda: predict(
            restricted, concentration=4.0, sources=names
        ),
        "registered_lineage_analysis": lambda: log_linear_posterior(opinions, graph),
        "shared_verifier": verifier,
        "typed_record_end_to_end": end_to_end,
    }
    rows = []
    for method, function in functions.items():
        result = function()
        if method == "shared_verifier" and result.route != "confirm":
            raise AssertionError("Canonical verifier route must be confirm")
        if method == "typed_record_end_to_end" and result.route != "confirm":
            raise AssertionError("Canonical end-to-end route must be confirm")
        row = {"source_count": len(names), "method": method}
        row.update(
            timed(
                function,
                warmup=args.warmup,
                repeats=args.repeats,
                iterations=args.iterations,
            )
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.iterations) <= 0:
        raise SystemExit("warmup, repeats, and iterations must be positive")

    scaling = benchmark_source_scaling(args)
    canonical = benchmark_canonical_path(args)
    output = ROOT / "outputs/scalability"
    write_csv(output / "source_count_scaling.csv", scaling)
    write_csv(output / "canonical_path.csv", canonical)
    summary = {
        "protocol": {
            "clock": "time.perf_counter_ns",
            "warmup_calls": args.warmup,
            "repeat_blocks": args.repeats,
            "timed_calls_per_block": args.iterations,
            "memory_metric": "incremental peak traced Python allocation",
            "memory_calls": min(args.iterations, 100),
            "throughput_metric": "median over dedicated repeat blocks",
            "source_counts": list(SOURCE_COUNTS),
            "registered_graph_topology": "sparse disjoint pairs",
            "execution": "single Python process; CPU affinity not set",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": os.environ.get(
                "ACTION_ADMISSION_CPU_NAME", platform.processor()
            ),
            "logical_cpu_count": os.cpu_count(),
        },
        "source_count_scaling": scaling,
        "canonical_path": canonical,
    }
    summary["provenance"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {
            "source_count_scaling.csv": hashlib.sha256(
                (output / "source_count_scaling.csv").read_bytes()
            ).hexdigest(),
            "canonical_path.csv": hashlib.sha256(
                (output / "canonical_path.csv").read_bytes()
            ).hexdigest(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "summary.json")


if __name__ == "__main__":
    main()
