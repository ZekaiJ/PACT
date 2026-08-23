#!/usr/bin/env python3
"""Render the provenance-corruption figure from frozen summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "split": "#D55E00",
    "merge": "#009E73",
    "edge": "#0072B2",
    "baseline": "#4D4D4D",
}


def _rows(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    return frame.loc[frame["condition"] == condition].sort_values("rate")


def _draw_series(
    axis: plt.Axes,
    rows: pd.DataFrame,
    metric: str,
    *,
    label: str,
    color: str,
    marker: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = 100.0 * rows["rate"].to_numpy(dtype=float)
    y = rows[f"{metric}_mean"].to_numpy(dtype=float)
    low = rows[f"{metric}_ci_low"].to_numpy(dtype=float)
    high = rows[f"{metric}_ci_high"].to_numpy(dtype=float)
    axis.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0, zorder=1)
    axis.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=3.8,
        linewidth=1.55,
        label=label,
        zorder=3,
    )
    return x, y


def _direct_label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    ha: str = "left",
) -> None:
    axis.text(
        x,
        y,
        text,
        color=color,
        fontsize=6.7,
        fontweight="normal",
        ha=ha,
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
        zorder=5,
    )


def _combined_coincident_rows(
    frame: pd.DataFrame,
    metric: str,
    first_condition: str,
    second_condition: str,
) -> pd.DataFrame:
    first = _rows(frame, first_condition).reset_index(drop=True).copy()
    second = _rows(frame, second_condition).reset_index(drop=True)
    if not np.allclose(
        first["rate"].to_numpy(dtype=float),
        second["rate"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Rate grids diverge for {first_condition} and {second_condition}.")
    if not np.allclose(
        first[f"{metric}_mean"].to_numpy(dtype=float),
        second[f"{metric}_mean"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Point estimates diverge for {first_condition} and {second_condition}."
        )
    first[f"{metric}_ci_low"] = np.minimum(
        first[f"{metric}_ci_low"].to_numpy(dtype=float),
        second[f"{metric}_ci_low"].to_numpy(dtype=float),
    )
    first[f"{metric}_ci_high"] = np.maximum(
        first[f"{metric}_ci_high"].to_numpy(dtype=float),
        second[f"{metric}_ci_high"].to_numpy(dtype=float),
    )
    return first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.results / "risk_coverage_by_corruption.csv")
    baseline = json.loads((args.results / "no_lineage_baseline.json").read_text())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    metrics = ("wrong_all", "coverage")
    titles = (
        "Wrong admission rises under false separation",
        "Separation raises coverage; merging lowers it",
    )
    ylabels = ("Wrong-admission rate", "Coverage")

    for axis, metric, title, ylabel in zip(axes, metrics, titles, ylabels, strict=True):
        split_x, split_y = _draw_series(
            axis,
            _combined_coincident_rows(
                frame,
                metric,
                "false_split",
                "forged_independence",
            ),
            metric,
            label="False separation / distinct-parent error",
            color=COLORS["split"],
            marker="o",
        )
        merge_rows = (
            _combined_coincident_rows(
                frame,
                metric,
                "false_merge",
                "hidden_edge_deletion",
            )
            if metric == "wrong_all"
            else _rows(frame, "false_merge")
        )
        merge_x, merge_y = _draw_series(
            axis,
            merge_rows,
            metric,
            label="False merge",
            color=COLORS["merge"],
            marker="s",
        )
        edge_x = edge_y = None
        if metric == "coverage":
            edge_x, edge_y = _draw_series(
                axis,
                _rows(frame, "hidden_edge_deletion"),
                metric,
                label="Recoverable-edge deletion",
                color=COLORS["edge"],
                marker="^",
            )
        baseline_value = float(baseline[metric])
        axis.axhline(
            baseline_value,
            color=COLORS["baseline"],
            linestyle=(0, (3.2, 2.4)),
            linewidth=1.05,
            label="Provenance-unaware",
            zorder=2,
        )
        axis.set_title(title, loc="left", fontweight="normal", pad=5)
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlim(-5, 113)
        axis.grid(False)
        axis.tick_params(direction="out", length=3.0, width=0.8, pad=2.5)
        axis.spines[["top", "right"]].set_visible(False)

        if metric == "wrong_all":
            axis.set_ylim(0.006, 0.095)
            axis.set_yticks([0.01, 0.03, 0.05, 0.07, 0.09])
            _direct_label(axis, 2.0, baseline_value + 0.0018, "Provenance-unaware", color=COLORS["baseline"])
            _direct_label(axis, 52.0, float(np.interp(52.0, split_x, split_y)) + 0.0045,
                          "Separation / distinct-parent error", color=COLORS["split"])
            _direct_label(axis, 52.0, float(np.interp(52.0, merge_x, merge_y)) + 0.0030,
                          "False merge / edge deletion", color=COLORS["merge"])
        else:
            axis.set_ylim(0.110, 0.1745)
            axis.set_yticks([0.11, 0.13, 0.15, 0.17])
            _direct_label(axis, 2.0, baseline_value + 0.0011, "Provenance-unaware", color=COLORS["baseline"])
            _direct_label(axis, 48.0, float(np.interp(48.0, split_x, split_y)) + 0.0042,
                          "Separation / distinct-parent error", color=COLORS["split"])
            _direct_label(axis, 99.0, float(merge_y[-1]) - 0.0020,
                          "False merge", color=COLORS["merge"], ha="right")
            assert edge_y is not None
            _direct_label(axis, 99.0, float(edge_y[-1]) + 0.0017,
                          "Recoverable-edge deletion", color=COLORS["edge"], ha="right")

    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    figure.supxlabel("Corrupted scene–cue clusters (%)", y=0.025, fontsize=8)
    figure.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.22, wspace=0.28)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
