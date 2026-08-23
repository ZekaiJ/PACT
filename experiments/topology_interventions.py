#!/usr/bin/env python3
"""Run native-view topology interventions on one frozen per-view evidence pair."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from action_admission.pcecf import SourceEvidence, forward  # noqa: E402


def source(index: int, evidence: np.ndarray, parent: str) -> SourceEvidence:
    alpha = evidence + 1.0
    return SourceEvidence(
        source_id=f"opinion_{index}", probabilities=alpha / alpha.sum(), quality=1.0,
        conflict=0.0, missing=False, parents=(parent,), evidence=evidence,
    )


def pact(evidences: list[np.ndarray], parents: list[str]) -> dict:
    result = forward([source(i, e, parents[i]) for i, e in enumerate(evidences)], concentration=1.0)
    return {
        "posterior": result.posterior,
        "score": result.selection_score,
        "component_count": len(result.group_ids),
        "evidence_budget": float(result.group_evidence.sum()),
    }


def product(evidences: list[np.ndarray]) -> dict:
    probabilities = [(evidence + 1.0) / (evidence.sum() + evidence.size) for evidence in evidences]
    log_values = np.sum(np.log(np.clip(probabilities, 1e-300, 1.0)), axis=0)
    values = np.exp(log_values - log_values.max())
    posterior = values / values.sum()
    return {"posterior": posterior, "score": float(posterior.max())}


def nested(evidences: list[np.ndarray]) -> dict:
    classes = evidences[0].size
    opinions = [(e / (e.sum() + classes), classes / (e.sum() + classes)) for e in evidences]
    belief, uncertainty = opinions[0]
    for right_belief, right_uncertainty in opinions[1:]:
        conflict = max(float(belief.sum() * right_belief.sum() - np.dot(belief, right_belief)), 0.0)
        denominator = max(1.0 - conflict, 1e-12)
        belief = (
            belief * right_belief + belief * right_uncertainty + right_belief * uncertainty
        ) / denominator
        uncertainty = uncertainty * right_uncertainty / denominator
    posterior = belief + uncertainty / classes
    posterior = posterior / posterior.sum()
    return {"posterior": posterior, "score": float(1.0 - uncertainty)}


def arms(row: dict, multiplicities: tuple[int, ...], sigma: float, noise_seed: int):
    native = [np.asarray(values, dtype=float) for values in row["evidences"]]
    donor = int(row["record_id"]) % len(native)
    distinct = [f"native-view:{index}" for index in range(len(native))]
    yield "A0_native", native, distinct
    yield "A3_false_merge", native, ["merged-parent"] * len(native)
    for multiplicity in multiplicities:
        exact = native + [native[donor].copy() for _ in range(multiplicity - 1)]
        yield f"A1_conserved_exact_m{multiplicity}", exact, distinct + [distinct[donor]] * (multiplicity - 1)
        yield f"A2_false_split_m{multiplicity}", exact, distinct + [f"false-split:{index}" for index in range(multiplicity - 1)]
        rng = np.random.default_rng(
            noise_seed + int(row["seed"]) * 100003 + int(row["record_id"]) * 101 + multiplicity
        )
        near = native + [
            np.exp(np.log(native[donor] + 1e-12) + rng.normal(0.0, sigma, native[donor].shape))
            for _ in range(multiplicity - 1)
        ]
        yield f"A4_registered_near_m{multiplicity}", near, distinct + [distinct[donor]] * (multiplicity - 1)


def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    values = []
    for label in range(classes):
        tp = np.sum((y == label) & (pred == label))
        fp = np.sum((y != label) & (pred == label))
        fn = np.sum((y == label) & (pred != label))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def ncs_aurc(scores: np.ndarray, wrong: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    risk = np.cumsum(wrong[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    grid = np.linspace(max(0.01, coverage[0]), 1.0, 101)
    return float(np.trapezoid(np.interp(grid, coverage, risk), grid) / (grid[-1] - grid[0]))


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["arm"])].append(row)
    output = []
    for (method, arm), group in sorted(groups.items()):
        y = np.asarray([row["y"] for row in group], dtype=int)
        posterior = np.asarray([row["posterior"] for row in group], dtype=float)
        pred = posterior.argmax(axis=1)
        scores = np.asarray([row["score"] for row in group], dtype=float)
        one_hot = np.eye(posterior.shape[1])[y]
        confidence = posterior.max(axis=1)
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            mask = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
            if mask.any():
                ece += mask.mean() * abs(np.mean(pred[mask] == y[mask]) - confidence[mask].mean())
        base = {
            "method": method, "arm": arm, "n": len(group),
            "accuracy": float(np.mean(pred == y)),
            "macro_f1": macro_f1(y, pred, posterior.shape[1]),
            "nll": float(-np.log(np.clip(posterior[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
            "brier": float(np.square(posterior - one_hot).sum(axis=1).mean()),
            "ece_10bin": float(ece),
            "ncs_aurc_full_support": ncs_aurc(scores, pred != y),
        }
        if method == "PACT":
            base |= {
                "mean_component_count": float(np.mean([row["component_count"] for row in group])),
                "mean_evidence_budget": float(np.mean([row["evidence_budget"] for row in group])),
                "mean_posterior_l1_from_A0": float(np.mean([row["posterior_l1_from_A0"] for row in group])),
                "mean_score_drift_from_A0": float(np.mean([row["score_drift_from_A0"] for row in group])),
            }
        output.append(base)
    return output


def bootstrap(rows: list[dict], draws: int, seed: int) -> list[dict]:
    pact_rows = [row for row in rows if row["method"] == "PACT"]
    by_key = {(row["seed"], row["record_id"], row["arm"]): row for row in pact_rows}
    record_ids = sorted({row["record_id"] for row in pact_rows})
    seeds = sorted({row["seed"] for row in pact_rows})
    arms_seen = sorted({row["arm"] for row in pact_rows if row["arm"] != "A0_native"})
    rng = np.random.default_rng(seed)
    output = []
    for arm in arms_seen:
        for metric in ("evidence_budget", "component_count"):
            per_record = np.asarray([
                np.mean([by_key[(train_seed, record_id, arm)][metric] - by_key[(train_seed, record_id, "A0_native")][metric] for train_seed in seeds])
                for record_id in record_ids
            ])
            estimates = np.asarray([np.mean(per_record[rng.integers(0, len(per_record), len(per_record))]) for _ in range(draws)])
            output.append({
                "arm": arm, "metric": metric, "clusters": len(record_ids), "training_seeds": len(seeds),
                "delta": float(per_record.mean()), "ci_low": float(np.quantile(estimates, 0.025)),
                "ci_high": float(np.quantile(estimates, 0.975)), "draws": draws,
            })
    return output


def self_test() -> None:
    row = {"seed": 11, "record_id": 0, "y": 0, "evidences": [[8, 0], [2, 1]]}
    values = {}
    for arm, evidence, parents in arms(row, (2, 8), 0.01, 4242):
        values[arm] = pact(evidence, parents)
    assert np.allclose(values["A0_native"]["posterior"], values["A1_conserved_exact_m8"]["posterior"])
    assert values["A2_false_split_m8"]["evidence_budget"] > values["A0_native"]["evidence_budget"]
    assert values["A3_false_merge"]["evidence_budget"] < values["A0_native"]["evidence_budget"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob")
    parser.add_argument("--pair-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.input_glob or not args.pair_id or not args.output:
        parser.error("--input-glob, --pair-id, and --output are required")

    source_rows = []
    for filename in sorted(glob.glob(args.input_glob)):
        with Path(filename).open(encoding="utf-8") as handle:
            source_rows.extend(json.loads(line) for line in handle if line.strip())
    source_rows = [row for row in source_rows if row["split"] == "test"]
    if not source_rows:
        raise ValueError("no test records")

    records = []
    for row in source_rows:
        baseline = None
        generated = list(arms(row, (2, 4, 8), args.sigma, 4242))
        for arm_name, evidences, parents in generated:
            pact_output = pact(evidences, parents)
            if arm_name == "A0_native":
                baseline = pact_output
            assert baseline is not None
            for method, output in (("PACT", pact_output), ("product", product(evidences)), ("nested", nested(evidences))):
                record = {
                    "pair_id": args.pair_id, "seed": int(row["seed"]), "record_id": int(row["record_id"]),
                    "y": int(row["y"]), "arm": arm_name, "method": method,
                    "posterior": output["posterior"].tolist(), "score": float(output["score"]),
                }
                if method == "PACT":
                    record |= {
                        "component_count": int(output["component_count"]),
                        "evidence_budget": float(output["evidence_budget"]),
                        "posterior_l1_from_A0": float(np.abs(output["posterior"] - baseline["posterior"]).sum()),
                        "score_drift_from_A0": float(output["score"] - baseline["score"]),
                    }
                records.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = summarize(records)
    (args.output / "RESULTS.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    boot = bootstrap(records, args.draws, 20260726)
    with (args.output / "bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boot[0]))
        writer.writeheader(); writer.writerows(boot)

    exact = [row for row in summary if row["method"] == "PACT" and row["arm"].startswith("A1_")]
    split_budget = [row for row in boot if row["metric"] == "evidence_budget" and row["arm"].startswith("A2_")]
    merge_budget = next(row for row in boot if row["metric"] == "evidence_budget" and row["arm"] == "A3_false_merge")
    posterior_index = {
        (row["method"], row["arm"], row["seed"], row["record_id"]): row["posterior"]
        for row in records
    }
    reference_equal = all(
        np.allclose(
            posterior_index[(method, arm, seed, record_id)],
            posterior_index[(method, arm.replace("A2_false_split", "A1_conserved_exact"), seed, record_id)],
        )
        for method in ("product", "nested")
        for arm in ("A2_false_split_m2", "A2_false_split_m4", "A2_false_split_m8")
        for seed, record_id in {(row["seed"], row["record_id"]) for row in source_rows}
    )
    checks = {
        "exact_copy_invariance": all(row["mean_posterior_l1_from_A0"] <= 1e-9 for row in exact),
        "false_split_budget_increase": all(row["ci_low"] > 0 for row in split_budget),
        "false_merge_budget_compression": merge_budget["ci_high"] < 0,
        "registration_blind_A1_A2_equal": reference_equal,
    }
    verdict = "PASS" if all(checks.values()) else "PARTIAL" if checks["exact_copy_invariance"] else "NEGATIVE"
    (args.output / "verdict.json").write_text(json.dumps({"pair_id": args.pair_id, "verdict": verdict, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
