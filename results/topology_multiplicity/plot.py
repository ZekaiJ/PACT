#!/usr/bin/env python3
"""Render the verified topology-multiplicity stress result."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "topology_multiplicity_stress.pdf"


def main() -> None:
    contrast = pd.read_csv(ROOT / "matched_contrast.csv")
    contrast = contrast[contrast["support"] == "primary_0.10_0.39"].sort_values("multiplicity")
    stress = pd.read_csv(ROOT / "stress_summary.csv")
    stress = stress[stress["support"] == "primary_0.10_0.39"]

    teal, red, gray = "#0B7A75", "#C43C35", "#6B7280"
    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 9.2, "axes.labelsize": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)

    ax = axes[0]
    x = contrast["multiplicity"].to_numpy()
    y = contrast["point"].to_numpy()
    ax.fill_between(x, contrast["ci_low"], contrast["ci_high"], color=teal, alpha=0.17, linewidth=0)
    ax.plot(x, y, "o-", color=teal, lw=1.8, ms=4)
    ax.axhline(0, color="#9CA3AF", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("Same-source multiplicity $m$")
    ax.set_ylabel(r"ncsAURC$(F_0,V_0)-$ncsAURC$(F_1,V_0)$")
    ax.set_title("a  Fusion-stage contrast")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)

    ax = axes[1]
    for arm, color, label, marker in (
        ("singleton", red, "False split", "o"),
        ("registered", teal, "Registered copies", "s"),
        ("all_merge", gray, "All-source merge", "^"),
    ):
        subset = stress[stress["arm"] == arm].sort_values("multiplicity")
        ax.plot(
            subset["multiplicity"],
            subset["aggregate_budget_ratio_vs_m1"],
            marker=marker,
            color=color,
            lw=1.7,
            ms=4,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("Same-source multiplicity $m$")
    ax.set_ylabel("Evidence budget / value at $m=1$")
    ax.set_title("b  Evidence accounting")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
