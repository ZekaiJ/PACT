#!/usr/bin/env python3
"""Plot the exhaustive six-view PACT coarsening surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_LABELS = {
    "HandWritten-Mfeat__TMC": "TMC",
    "HandWritten-Mfeat__RCML": "RCML",
}
COLORS = {"TMC": "#0072B2", "RCML": "#D55E00"}


def panel_surface(axis: plt.Axes, frame: pd.DataFrame, pair_id: str) -> None:
    subset = frame[
        (frame["pair_id"] == pair_id) & (frame["score"] == "native_nonvacuity")
    ].copy()
    label = PAIR_LABELS[pair_id]
    for component_count in range(1, 7):
        values = subset.loc[
            subset["component_count"] == component_count, "ncsAURC_0p10_0p90"
        ].to_numpy()
        offsets = np.linspace(-0.22, 0.22, len(values)) if len(values) > 1 else [0.0]
        axis.scatter(
            component_count + np.asarray(offsets),
            values,
            s=13,
            color=COLORS[label],
            alpha=0.48,
            edgecolors="none",
            zorder=2,
        )
        median = float(np.median(values))
        lower, upper = np.quantile(values, [0.25, 0.75])
        axis.plot(
            [component_count - 0.25, component_count + 0.25],
            [median, median],
            color="black",
            linewidth=1.5,
            zorder=3,
        )
        axis.vlines(component_count, lower, upper, color="black", linewidth=1.1, zorder=3)
    axis.set_title(f"{label}: 203 partitions", loc="left", fontweight="bold")
    axis.set_xlabel("Provenance component count")
    axis.set_xlim(0.6, 6.4)
    axis.set_xticks(range(1, 7))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def panel_edges(axis: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary[summary["estimand"] == "cover_edge_fraction_coarsening_improves"].copy()
    order = [
        ("HandWritten-Mfeat__TMC", "native_nonvacuity"),
        ("HandWritten-Mfeat__TMC", "posterior_confidence"),
        ("HandWritten-Mfeat__RCML", "native_nonvacuity"),
        ("HandWritten-Mfeat__RCML", "posterior_confidence"),
    ]
    for index, (pair_id, score) in enumerate(order):
        row = rows[(rows["pair_id"] == pair_id) & (rows["score"] == score)].iloc[0]
        label = PAIR_LABELS[pair_id]
        marker = "o" if score == "native_nonvacuity" else "s"
        point = 100.0 * row["point"]
        low = 100.0 * row["ci_low"]
        high = 100.0 * row["ci_high"]
        axis.errorbar(
            index,
            point,
            yerr=[[point - low], [high - point]],
            fmt=marker,
            markersize=6.5,
            color=COLORS[label],
            ecolor=COLORS[label],
            capsize=3,
            linewidth=1.3,
            zorder=3,
        )
        axis.text(index, high + 1.2, f"{point:.1f}%", ha="center", va="bottom", fontsize=8)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(4), ["TMC\nselection score", "TMC\nconfidence", "RCML\nselection score", "RCML\nconfidence"])
    axis.set_ylabel("Single-merge relations with\nlower ncsAURC (%)")
    axis.set_title("Budget order does not fix\ndecision order", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(30.0, 100.0 * rows["ci_high"].max() + 5.0))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.results / "PARTITION_METRICS.csv")
    summary = pd.read_csv(args.results / "COVER_EDGE_SUMMARY.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.45),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.12)},
        constrained_layout=True,
    )
    panel_surface(axes[0], metrics, "HandWritten-Mfeat__TMC")
    axes[0].set_ylabel(r"ncsAURC$_{[0.10,0.90]}$ (lower is better)")
    panel_surface(axes[1], metrics, "HandWritten-Mfeat__RCML")
    panel_edges(axes[2], summary)
    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
