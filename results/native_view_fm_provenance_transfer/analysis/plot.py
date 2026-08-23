from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("analysis_test32", "Qwen3-VL-32B", "#17365D", "o"),
    ("analysis_test8", "Qwen3-VL-8B", "#168C88", "s"),
)


def read_points(path: Path) -> dict[tuple[int, str, str], float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (int(row["m"]), row["arm"], row["metric"]): float(row["estimate"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.results / "gates" / "FINAL_RESULT_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "FROZEN_RESULTS_COMPLETE_WITH_LOCKED_TIE_CORRECTION"
    assert audit["verification"]["independent_primary_recompute_pass"]
    assert audit["verification"]["semantic_tie_checks_pass"]

    loaded = []
    for directory, label, color, marker in MODELS:
        result = json.loads((args.results / directory / "PRIMARY_RESULT.json").read_text(encoding="utf-8"))
        points = read_points(args.results / directory / "point_estimates.csv")
        assert result["status"] == "COMPLETE_WITH_LOCKED_TIE_CORRECTION"
        assert [row["m"] for row in result["primary_contrasts"]] == [1, 2, 4]
        assert result["primary_contrasts"][0]["estimate"] == 0.0
        loaded.append((label, color, marker, result, points))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 5.8), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.08))
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, :])

    x = np.arange(3)
    offsets = (-0.055, 0.055)
    for offset, (label, color, marker, result, _) in zip(offsets, loaded):
        rows = result["primary_contrasts"]
        estimate = np.asarray([row["estimate"] for row in rows])
        low = np.asarray([row["ci_low"] for row in rows])
        high = np.asarray([row["ci_high"] for row in rows])
        ax_a.errorbar(
            x + offset,
            estimate,
            yerr=np.vstack((estimate - low, high - estimate)),
            color=color,
            marker=marker,
            markersize=4.8,
            linewidth=1.35,
            capsize=2.5,
            label=label,
        )
    ax_a.axhline(0, color="#777777", linewidth=0.8)
    ax_a.set_xticks(x, ("1", "2", "4"))
    ax_a.set_xlabel("Prompt surfaces per physical view, $m$")
    ax_a.set_ylabel(r"$\Delta$ ncsAURC (unaware $-$ registered)")
    ax_a.set_title("Native non-vacuity readout")
    ax_a.set_ylim(-0.035, 0.22)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    score_specs = (
        ("primary_contrasts", "Native non-vacuity", "o", "#17365D"),
        ("common_score_contrasts", "Posterior confidence", "D", "#D9781C"),
    )
    model_x = np.arange(len(loaded))
    for offset, (result_key, label, marker, color) in zip((-0.08, 0.08), score_specs):
        estimate, low, high = [], [], []
        for _, _, _, result, _ in loaded:
            row = next(item for item in result[result_key] if item["m"] == 4)
            estimate.append(row["estimate"])
            low.append(row["ci_low"])
            high.append(row["ci_high"])
        estimate, low, high = map(np.asarray, (estimate, low, high))
        ax_b.errorbar(
            model_x + offset,
            estimate,
            yerr=np.vstack((estimate - low, high - estimate)),
            linestyle="none",
            marker=marker,
            color=color,
            markersize=5.2,
            capsize=2.5,
            label=label,
        )
    ax_b.axhline(0, color="#777777", linewidth=0.8)
    ax_b.set_xticks(model_x, ("32B", "8B"))
    ax_b.set_xlabel("Frozen checkpoint")
    ax_b.set_ylabel(r"$\Delta$ ncsAURC at $m=4$")
    ax_b.set_title("The contrast depends on score readout")
    ax_b.set_ylim(-0.055, 0.17)
    ax_b.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax_b.legend(frameon=False, loc="upper left")

    arms = (
        ("acquisition_registered", "Acquisition registered"),
        ("lineage_unaware", "Per-output counting"),
        ("shuffled_equal_cardinality", "Shuffled grouping (post hoc)"),
        ("exact_dedup", "Exact deduplication"),
        ("all_view_merge", "All-view merge"),
        ("native_one_per_view", "One output per view"),
    )
    y = np.arange(len(arms))
    for offset, (label, color, marker, _, points) in zip((-0.10, 0.10), loaded):
        values = np.asarray([points[(4, arm, "budget_ratio")] for arm, _ in arms])
        ax_c.scatter(values, y + offset, color=color, marker=marker, s=28, zorder=3, label=label)
    ax_c.axvline(1, color="#777777", linewidth=0.8)
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.02, 5.5)
    ax_c.set_xticks((0.03, 0.1, 0.3, 1, 3), ("0.03", "0.1", "0.3", "1", "3"))
    ax_c.set_yticks(y, [label for _, label in arms])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Evidence budget relative to one output per physical view (log scale)")
    ax_c.set_title("Registration determines the counted budget")
    ax_c.grid(axis="x", color="#D9D9D9", linewidth=0.6)

    for label, ax in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
        ax.text(
            -0.16 if ax is not ax_c else -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=2, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
