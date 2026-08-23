#!/usr/bin/env python3
"""Analyze the frozen 4-family x 2-checkpoint FM evidence panel."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODELS = (
    "qwen3vl_8b",
    "qwen3vl_32b",
    "internvl3_2b",
    "internvl3_8b",
    "llava_onevision_0_5b",
    "llava_onevision_7b",
    "smolvlm2_0_5b",
    "smolvlm2_2_2b",
)
REGISTERED = ((0, 1), (2, 3), (4, 5), (6, 7))
COVERAGES = np.linspace(0.10, 0.90, 36)
METHODS = (
    "product",
    "nested_dirichlet",
    "hierarchy_matched_cautious",
    "pact_registered_family",
    "pact_false_split",
    "pact_false_merge",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairings(items: tuple[int, ...]):
    if not items:
        yield ()
        return
    first = items[0]
    for offset in range(1, len(items)):
        second = items[offset]
        rest = items[1:offset] + items[offset + 1 :]
        for tail in pairings(rest):
            yield tuple(sorted(((first, second),) + tail))


def pact(probabilities: np.ndarray, groups: tuple[tuple[int, ...], ...]):
    evidence = np.stack((2.0 * (1.0 - probabilities), 2.0 * probabilities), axis=2)
    grouped = np.stack([np.min(evidence[:, group, :], axis=1) for group in groups], axis=1)
    fused = np.sum(grouped, axis=1)
    budget = np.sum(fused, axis=1)
    posterior = (fused + 1.0) / (budget[:, None] + 2.0)
    return posterior[:, 1], budget


def product(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    log_odds = np.sum(np.log(clipped) - np.log1p(-clipped), axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -60.0, 60.0)))


def nested_dirichlet(probabilities: np.ndarray) -> np.ndarray:
    belief = np.stack((0.5 * (1.0 - probabilities[:, 0]), 0.5 * probabilities[:, 0]), axis=1)
    uncertainty = np.full(probabilities.shape[0], 0.5)
    for column in range(1, probabilities.shape[1]):
        right = np.stack((0.5 * (1.0 - probabilities[:, column]), 0.5 * probabilities[:, column]), axis=1)
        right_uncertainty = 0.5
        conflict = np.sum(belief, axis=1) * 0.5 - np.sum(belief * right, axis=1)
        denominator = np.maximum(1.0 - conflict, 1e-12)
        belief = (
            belief * right
            + belief * right_uncertainty
            + right * uncertainty[:, None]
        ) / denominator[:, None]
        uncertainty = uncertainty * right_uncertainty / denominator
    return np.clip(belief[:, 1] + 0.5 * uncertainty, 0.0, 1.0)


def hierarchy_matched_cautious(probabilities: np.ndarray) -> np.ndarray:
    q_not = 1.0 - 0.5 * probabilities
    q_ready = 0.5 * (1.0 + probabilities)
    q_frame = 0.5
    weights = np.stack(
        (
            np.log(q_not) + np.log(q_ready) - math.log(q_frame),
            math.log(q_frame) - np.log(q_not),
            math.log(q_frame) - np.log(q_ready),
        ),
        axis=2,
    )
    combined = np.sum(
        np.stack([np.min(weights[:, group, :], axis=1) for group in REGISTERED], axis=1),
        axis=1,
    )
    common_not = np.exp(combined[:, 0] + combined[:, 2])
    common_ready = np.exp(combined[:, 0] + combined[:, 1])
    common_frame = np.exp(np.sum(combined, axis=1))
    conflict = 1.0 - common_not - common_ready + common_frame
    denominator = 1.0 - conflict
    if np.any(denominator <= 1e-12):
        raise ValueError("hierarchy-matched cautious fusion has total conflict")
    mass_ready = common_ready - common_frame
    if np.min(mass_ready) < -1e-8 or np.min(common_frame) < -1e-8:
        raise ValueError("hierarchy-matched cautious fusion produced negative mass")
    return np.clip((mass_ready + 0.5 * common_frame) / denominator, 0.0, 1.0)


def adaptive_ece(probability: np.ndarray, reference: np.ndarray) -> float:
    confidence = np.maximum(probability, 1.0 - probability)
    correct = ((probability >= 0.5) == reference).astype(float)
    edges = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, 11)))
    if len(edges) == 1:
        return float(abs(np.mean(correct) - np.mean(confidence)))
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right)
        if right == edges[-1]:
            mask = (confidence >= left) & (confidence <= right)
        if np.any(mask):
            total += np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(confidence[mask]))
    return float(total)


def classification(probability: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "accuracy": float(np.mean((probability >= 0.5) == reference)),
        "negative_log_likelihood": float(-np.mean(reference * np.log(clipped) + (1 - reference) * np.log1p(-clipped))),
        "unnormalized_multiclass_brier": float(np.mean(2.0 * (probability - reference) ** 2)),
        "adaptive_ece_10": adaptive_ece(probability, reference),
    }


def admission_curve(scores: np.ndarray, reference: np.ndarray) -> list[dict[str, float]]:
    unique = np.unique(scores)[::-1]
    group_total = np.asarray([np.sum(scores == value) for value in unique], dtype=float)
    group_wrong = np.asarray([np.sum((scores == value) & (reference == 0)) for value in unique], dtype=float)
    group_correct = group_total - group_wrong
    cumulative_total = np.cumsum(group_total)
    cumulative_wrong = np.cumsum(group_wrong)
    cumulative_correct = np.cumsum(group_correct)
    rows = []
    for coverage in COVERAGES:
        target = coverage * len(scores)
        index = int(np.searchsorted(cumulative_total, target, side="left"))
        before_total = cumulative_total[index - 1] if index else 0.0
        before_wrong = cumulative_wrong[index - 1] if index else 0.0
        before_correct = cumulative_correct[index - 1] if index else 0.0
        fraction = (target - before_total) / group_total[index]
        wrong = before_wrong + fraction * group_wrong[index]
        correct = before_correct + fraction * group_correct[index]
        rows.append(
            {
                "coverage": float(coverage),
                "wrong_all": float(wrong / len(scores)),
                "correct_all": float(correct / len(scores)),
                "conditional_wrong_admission": float(wrong / max(wrong + correct, 1e-12)),
            }
        )
    return rows


def ncs_aurc(curve: list[dict[str, float]]) -> float:
    coverage = np.asarray([row["coverage"] for row in curve])
    risk = np.asarray([row["conditional_wrong_admission"] for row in curve])
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0]))


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    n11 = int(np.sum((left == 1) & (right == 1)))
    n10 = int(np.sum((left == 1) & (right == 0)))
    n01 = int(np.sum((left == 0) & (right == 1)))
    n00 = int(np.sum((left == 0) & (right == 0)))
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / denominator if denominator else float("nan")
    lift_denominator = (n11 + n10) * (n11 + n01)
    lift = len(left) * n11 / lift_denominator if lift_denominator else float("nan")
    conditionals = []
    if n11 + n10:
        conditionals.append(n11 / (n11 + n10))
    if n11 + n01:
        conditionals.append(n11 / (n11 + n01))
    return {
        "n": len(left),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "checkpoint_pair_phi": float(phi),
        "joint_error_lift": float(lift),
        "conditional_co_error": float(np.mean(conditionals)) if conditionals else float("nan"),
    }


def load_inputs(cases_path: Path, input_dir: Path):
    with cases_path.open(encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    ids = [row["id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case identifiers are not unique")
    probabilities = np.empty((len(cases), len(MODELS)), dtype=float)
    input_hashes = {str(cases_path): sha256(cases_path)}
    for column, model in enumerate(MODELS):
        path = input_dir / f"{model}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {row["id"]: row for row in rows}
        if len(rows) != len(cases) or set(by_id) != set(ids):
            raise ValueError(f"{model} denominator mismatch: {len(rows)} rows")
        for index, case in enumerate(cases):
            row = by_id[case["id"]]
            for field in ("event_id", "episode_id", "task_id", "window", "reference_ready", "image"):
                if str(row[field]) != str(case[field]):
                    raise ValueError(f"{model} metadata mismatch at {case['id']}: {field}")
            probabilities[index, column] = float(row["p_ready"])
        input_hashes[str(path)] = sha256(path)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("invalid checkpoint probabilities")
    return cases, probabilities, input_hashes


def cluster_sample_indices(cases: list[dict[str, str]], rng: np.random.Generator) -> np.ndarray:
    task_episode_rows: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(cases):
        task_episode_rows.setdefault(row["task_id"], {}).setdefault(row["episode_id"], []).append(index)
    sampled = []
    for task in sorted(task_episode_rows):
        episodes = sorted(task_episode_rows[task])
        for episode_index in rng.integers(0, len(episodes), size=len(episodes)):
            sampled.extend(task_episode_rows[task][episodes[int(episode_index)]])
    return np.asarray(sampled, dtype=int)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def make_figure(output_dir: Path, individual: list[dict], curves: list[dict], partition_rows: list[dict], registered_stat: float) -> None:
    import matplotlib.pyplot as plt

    labels = [row["model_key"].replace("llava_onevision_", "LLaVA-").replace("qwen3vl_", "Qwen-").replace("internvl3_", "Intern-").replace("smolvlm2_", "Smol-") for row in individual]
    accuracy = [row["accuracy"] for row in individual]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    axes[0].bar(np.arange(len(labels)), accuracy, color="#4C78A8")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Frozen FM opinions", loc="left", fontweight="bold")
    colors = {
        "product": "#9C755F",
        "nested_dirichlet": "#F28E2B",
        "hierarchy_matched_cautious": "#B07AA1",
        "pact_registered_family": "#2A9D8F",
        "pact_false_split": "#E15759",
        "pact_false_merge": "#7F7F7F",
    }
    for method in METHODS:
        subset = [row for row in curves if row["method"] == method]
        axes[1].plot([row["coverage"] for row in subset], [row["conditional_wrong_admission"] for row in subset], label=method.replace("pact_", "PACT ").replace("_", " "), color=colors[method], lw=1.6)
    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("Conditional wrong admission")
    axes[1].set_title("(b) Shared admission interface", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5.7, frameon=False)
    scores = sorted(row["phi_contrast"] for row in partition_rows)
    axes[2].plot(np.arange(1, len(scores) + 1), scores, color="#9AA0A6", lw=1.2)
    rank = 1 + sum(value > registered_stat + 1e-15 for value in scores)
    axes[2].scatter([len(scores) - rank + 1], [registered_stat], color="#D62728", s=28, zorder=3, label="registered")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_xlabel("Pair partition (sorted)")
    axes[2].set_ylabel(r"Mean $\phi_{within}-\phi_{between}$")
    axes[2].set_title("(c) Exact 105-partition test", loc="left", fontweight="bold")
    axes[2].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18, lw=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "FM_PANEL_MAIN_FIGURE.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "FM_PANEL_MAIN_FIGURE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    all_pairings = list(pairings(tuple(range(8))))
    assert len(all_pairings) == 105 and len(set(all_pairings)) == 105
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.05, 0.95, size=(9, 8))
    ready, budget = pact(probabilities, REGISTERED)
    for copied in range(8):
        extended = np.column_stack((probabilities, probabilities[:, copied]))
        groups = tuple(tuple(group) + ((8,) if copied in group else ()) for group in REGISTERED)
        copied_ready, copied_budget = pact(extended, groups)
        assert np.max(np.abs(copied_ready - ready)) < 1e-12
        assert np.max(np.abs(copied_budget - budget)) < 1e-12
    split_ready, split_budget = pact(probabilities, tuple((index,) for index in range(8)))
    merge_ready, merge_budget = pact(probabilities, (tuple(range(8)),))
    assert np.all(split_budget + 1e-12 >= budget) and np.all(merge_budget <= budget + 1e-12)
    assert np.all(np.isfinite(split_ready)) and np.all(np.isfinite(merge_ready))
    reference = np.asarray([0, 1] * 5)
    tied = admission_curve(np.ones(10) * 0.5, reference)
    assert max(abs(row["coverage"] - (row["wrong_all"] + row["correct_all"])) for row in tied) < 1e-12
    assert np.all((nested_dirichlet(probabilities) >= 0.0) & (nested_dirichlet(probabilities) <= 1.0))
    assert np.all((hierarchy_matched_cautious(probabilities) >= 0.0) & (hierarchy_matched_cautious(probabilities) <= 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("SELF_CHECK_PASS")
        return
    if json.loads(args.pilot_gate.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("pilot gate is not PASS")
    analysis_lock = json.loads(args.analysis_lock.read_text(encoding="utf-8"))
    if analysis_lock["status"] != "LOCKED_BEFORE_FULL_INFERENCE":
        raise RuntimeError("analysis lock is not frozen")

    args.output.mkdir(parents=True, exist_ok=True)
    cases, probabilities, input_hashes = load_inputs(args.cases, args.inputs)
    reference = np.asarray([int(row["reference_ready"]) for row in cases], dtype=int)
    predictions = {
        "product": product(probabilities),
        "nested_dirichlet": nested_dirichlet(probabilities),
        "hierarchy_matched_cautious": hierarchy_matched_cautious(probabilities),
    }
    predictions["pact_registered_family"], registered_budget = pact(probabilities, REGISTERED)
    predictions["pact_false_split"], split_budget = pact(probabilities, tuple((index,) for index in range(8)))
    predictions["pact_false_merge"], merge_budget = pact(probabilities, (tuple(range(8)),))

    metric_rows = []
    curve_rows = []
    task_rows = []
    curves_by_method = {}
    for method in METHODS:
        metrics = classification(predictions[method], reference)
        curve = admission_curve(predictions[method], reference)
        curves_by_method[method] = curve
        metric_rows.append({"method": method, **metrics, "ncsAURC": ncs_aurc(curve)})
        curve_rows.extend({"method": method, **row} for row in curve)
        for task in sorted({row["task_id"] for row in cases}):
            indices = np.asarray([index for index, row in enumerate(cases) if row["task_id"] == task])
            task_curve = admission_curve(predictions[method][indices], reference[indices])
            task_rows.append({"method": method, "task_id": task, **classification(predictions[method][indices], reference[indices]), "ncsAURC": ncs_aurc(task_curve)})

    individual_rows = []
    errors = np.empty_like(probabilities, dtype=int)
    early_mask = np.asarray([row["window"] == "early" for row in cases])
    ready_mask = ~early_mask
    for column, model in enumerate(MODELS):
        errors[:, column] = ((probabilities[:, column] >= 0.5) != reference).astype(int)
        predicted_ready = probabilities[:, column] >= 0.5
        individual_rows.append({
            "model_key": model,
            **classification(probabilities[:, column], reference),
            "predicted_ready_rate": float(np.mean(predicted_ready)),
            "early_predicted_ready_rate": float(np.mean(predicted_ready[early_mask])),
            "event_proximal_predicted_ready_rate": float(np.mean(predicted_ready[ready_mask])),
            "probability_mean": float(np.mean(probabilities[:, column])),
            "probability_std": float(np.std(probabilities[:, column])),
        })

    pair_rows = []
    pair_lookup = {}
    for left in range(8):
        for right in range(left + 1, 8):
            metrics = pair_metrics(errors[:, left], errors[:, right])
            relation = "within_family" if (left, right) in REGISTERED else "between_family"
            row = {"left_model": MODELS[left], "right_model": MODELS[right], "registered_relation": relation, **metrics}
            pair_rows.append(row)
            pair_lookup[(left, right)] = float(metrics["checkpoint_pair_phi"])
    if any(not math.isfinite(value) for value in pair_lookup.values()):
        raise RuntimeError("family partition test has an undefined checkpoint-pair phi")

    all_pairings = list(pairings(tuple(range(8))))
    partition_rows = []
    pairing_metric_rows = []
    registered_key = tuple(sorted(REGISTERED))
    for pairing in all_pairings:
        within = [pair_lookup[tuple(sorted(pair))] for pair in pairing]
        between = [value for pair, value in pair_lookup.items() if pair not in pairing]
        statistic = float(np.mean(within) - np.mean(between))
        partition_rows.append({"pairing": ";".join(f"{MODELS[a]}+{MODELS[b]}" for a, b in pairing), "is_registered": pairing == registered_key, "phi_contrast": statistic})
        ready, budget = pact(probabilities, pairing)
        curve = admission_curve(ready, reference)
        pairing_metric_rows.append({"pairing": partition_rows[-1]["pairing"], "is_registered": pairing == registered_key, **classification(ready, reference), "ncsAURC": ncs_aurc(curve), "mean_evidence_budget": float(np.mean(budget)), "mean_budget_ratio_to_registered": float(np.mean(budget / registered_budget)), "mean_posterior_l1_drift": float(np.mean(2.0 * np.abs(ready - predictions["pact_registered_family"])))})
    registered_stat = next(row["phi_contrast"] for row in partition_rows if row["is_registered"])
    exact_p = sum(row["phi_contrast"] >= registered_stat - 1e-15 for row in partition_rows) / len(partition_rows)
    confirmed = registered_stat > 0.0 and exact_p <= 0.05

    copy_budget_residual = 0.0
    copy_posterior_residual = 0.0
    for copied in range(8):
        extended = np.column_stack((probabilities, probabilities[:, copied]))
        groups = tuple(tuple(group) + ((8,) if copied in group else ()) for group in REGISTERED)
        copied_ready, copied_budget = pact(extended, groups)
        copy_budget_residual = max(copy_budget_residual, float(np.max(np.abs(copied_budget - registered_budget))))
        copy_posterior_residual = max(copy_posterior_residual, float(np.max(2.0 * np.abs(copied_ready - predictions["pact_registered_family"]))))
    topology_rows = [
        {"topology": "registered_family", "mean_total_evidence_budget": float(np.mean(registered_budget)), "mean_budget_ratio_to_registered": 1.0, "mean_posterior_l1_drift": 0.0},
        {"topology": "false_split", "mean_total_evidence_budget": float(np.mean(split_budget)), "mean_budget_ratio_to_registered": float(np.mean(split_budget / registered_budget)), "mean_posterior_l1_drift": float(np.mean(2.0 * np.abs(predictions["pact_false_split"] - predictions["pact_registered_family"])))},
        {"topology": "false_merge", "mean_total_evidence_budget": float(np.mean(merge_budget)), "mean_budget_ratio_to_registered": float(np.mean(merge_budget / registered_budget)), "mean_posterior_l1_drift": float(np.mean(2.0 * np.abs(predictions["pact_false_merge"] - predictions["pact_registered_family"])))},
        {"topology": "registered_exact_copy", "mean_total_evidence_budget": float(np.mean(registered_budget)), "mean_budget_ratio_to_registered": 1.0, "mean_posterior_l1_drift": 0.0, "max_budget_invariance_residual": copy_budget_residual, "max_posterior_l1_invariance_residual": copy_posterior_residual},
    ]

    rng = np.random.default_rng(5401)
    metric_names = ("accuracy", "negative_log_likelihood", "unnormalized_multiclass_brier", "adaptive_ece_10", "ncsAURC")
    bootstrap = np.empty((args.bootstrap_replicates, len(METHODS), len(metric_names)))
    pair_bootstrap = np.empty((args.bootstrap_replicates, len(pair_rows), 3))
    registered_stat_bootstrap = np.empty(args.bootstrap_replicates)
    pair_indices = [(left, right) for left in range(8) for right in range(left + 1, 8)]
    for replicate in range(args.bootstrap_replicates):
        indices = cluster_sample_indices(cases, rng)
        for method_index, method in enumerate(METHODS):
            metrics = classification(predictions[method][indices], reference[indices])
            metrics["ncsAURC"] = ncs_aurc(admission_curve(predictions[method][indices], reference[indices]))
            bootstrap[replicate, method_index, :] = [metrics[name] for name in metric_names]
        replicate_phi = {}
        for pair_index, (left, right) in enumerate(pair_indices):
            metrics = pair_metrics(errors[indices, left], errors[indices, right])
            values = [metrics["checkpoint_pair_phi"], metrics["joint_error_lift"], metrics["conditional_co_error"]]
            pair_bootstrap[replicate, pair_index, :] = values
            replicate_phi[(left, right)] = float(values[0])
        within = [replicate_phi[pair] for pair in REGISTERED]
        between = [value for pair, value in replicate_phi.items() if pair not in REGISTERED]
        registered_stat_bootstrap[replicate] = float(np.mean(within) - np.mean(between))

    bootstrap_rows = []
    point_by_method = {row["method"]: row for row in metric_rows}
    for method_index, method in enumerate(METHODS):
        for metric_index, metric in enumerate(metric_names):
            values = bootstrap[:, method_index, metric_index]
            bootstrap_rows.append({"estimand": metric, "method": method, "point": point_by_method[method][metric], "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "replicates": args.bootstrap_replicates})
    contrast_rows = []
    pact_index = METHODS.index("pact_registered_family")
    for method_index, method in enumerate(METHODS):
        if method == "pact_registered_family":
            continue
        for metric_index, metric in enumerate(metric_names):
            differences = bootstrap[:, pact_index, metric_index] - bootstrap[:, method_index, metric_index]
            point = point_by_method["pact_registered_family"][metric] - point_by_method[method][metric]
            contrast_rows.append({"contrast": f"pact_registered_family-minus-{method}", "estimand": metric, "point": point, "ci95_low": float(np.quantile(differences, 0.025)), "ci95_high": float(np.quantile(differences, 0.975)), "replicates": args.bootstrap_replicates})
    for pair_index, row in enumerate(pair_rows):
        for metric_index, metric in enumerate(("checkpoint_pair_phi", "joint_error_lift", "conditional_co_error")):
            values = pair_bootstrap[:, pair_index, metric_index]
            row[f"{metric}_ci95_low"] = float(np.nanquantile(values, 0.025))
            row[f"{metric}_ci95_high"] = float(np.nanquantile(values, 0.975))

    task_summary_rows = []
    for method in METHODS:
        subset = [row for row in task_rows if row["method"] == method]
        task_summary_rows.append({"method": method, "macro_accuracy": float(np.mean([row["accuracy"] for row in subset])), "worst_task_accuracy": float(np.min([row["accuracy"] for row in subset])), "macro_ncsAURC": float(np.mean([row["ncsAURC"] for row in subset])), "worst_task_ncsAURC": float(np.max([row["ncsAURC"] for row in subset]))})

    write_csv(args.output / "INDIVIDUAL_CHECKPOINT_METRICS.csv", individual_rows)
    write_csv(args.output / "FUSION_METRICS.csv", metric_rows)
    write_csv(args.output / "READY_ADMISSION_CURVES.csv", curve_rows)
    write_csv(args.output / "TASK_METRICS.csv", task_rows)
    write_csv(args.output / "TASK_MACRO_WORST.csv", task_summary_rows)
    write_csv(args.output / "CHECKPOINT_PAIR_DEPENDENCE.csv", pair_rows)
    write_csv(args.output / "PAIR_PARTITION_EXACT_TEST.csv", partition_rows)
    write_csv(args.output / "PAIRING_105_FUSION_METRICS.csv", pairing_metric_rows)
    write_csv(args.output / "TOPOLOGY_METRICS.csv", topology_rows)
    write_csv(args.output / "CLUSTER_BOOTSTRAP_CI.csv", bootstrap_rows)
    write_csv(args.output / "PAIRED_CONTRAST_CI.csv", contrast_rows)

    summary = {
        "status": "PASS_COMPLETE",
        "denominator": {"rows": len(cases), "events": len({row["event_id"] for row in cases}), "episodes": len({row["episode_id"] for row in cases}), "tasks": len({row["task_id"] for row in cases}), "checkpoints": len(MODELS), "families": 4},
        "protocol": {"analysis_lock_sha256": sha256(args.analysis_lock), "pilot_gate_sha256": sha256(args.pilot_gate), "input_sha256": input_hashes, "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": 5401},
        "family_partition_test": {"registered_phi_contrast": registered_stat, "registered_rank_descending": 1 + sum(row["phi_contrast"] > registered_stat + 1e-15 for row in partition_rows), "partitions": 105, "one_sided_exact_p": exact_p, "confirmed": confirmed, "bootstrap_ci95": [float(np.nanquantile(registered_stat_bootstrap, 0.025)), float(np.nanquantile(registered_stat_bootstrap, 0.975))], "interpretation": "family registration captures part of measured co-error structure" if confirmed else "family registration remains a conservative accounting policy, not an empirical dependence certificate"},
        "fusion_metrics": metric_rows,
        "constant_class_checkpoints": [row["model_key"] for row in individual_rows if row["predicted_ready_rate"] in (0.0, 1.0)],
        "topology": topology_rows,
        "claims_excluded": analysis_lock["reporting"]["claims_excluded"],
    }
    (args.output / "FM_PANEL_SUMMARY.json").write_text(json.dumps(clean(summary), indent=2, sort_keys=True), encoding="utf-8")
    report = [
        "# Balanced Foundation-Model Evidence Panel",
        "",
        f"Status: PASS_COMPLETE. The frozen denominator contains {len(cases):,} event-window rows from {summary['denominator']['events']:,} events, {summary['denominator']['episodes']:,} episodes, and {summary['denominator']['tasks']} tasks.",
        "",
        "## Exact family-partition result",
        "",
        f"Registered contrast (mean within-pair phi minus mean between-pair phi): {registered_stat:.4f}; exact one-sided p={exact_p:.4f} over 105 pair partitions; cluster-bootstrap 95% interval [{summary['family_partition_test']['bootstrap_ci95'][0]:.4f}, {summary['family_partition_test']['bootstrap_ci95'][1]:.4f}].",
        f"Authorized interpretation: {summary['family_partition_test']['interpretation']}.",
        f"Constant-class diagnostic: {', '.join(summary['constant_class_checkpoints']) or 'none'}. This diagnostic is retained because shared label-conditioned behavior can raise raw co-error without validating a family partition.",
        "",
        "## Claim boundary",
        "",
        "The panel evaluates frozen foundation-model opinions on a derived early/event-proximal HABIT protocol. It does not establish physical safety, authenticate independence, or make any checkpoint a release authority. All positive and null outcomes are retained.",
    ]
    (args.output / "CLAIM_SAFE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    make_figure(args.output, individual_rows, curve_rows, partition_rows, registered_stat)

    manifest_rows = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "MANIFEST.sha256":
            manifest_rows.append(f"{sha256(path)}  {path.name}")
    (args.output / "MANIFEST.sha256").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_COMPLETE", "rows": len(cases), "exact_p": exact_p, "family_confirmed": confirmed, "output": str(args.output)}))


if __name__ == "__main__":
    main()
