from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows(ROOT / "results" / "reference" / "aurc_summary.csv")
    by_method = {row["method"]: row for row in rows}
    registered = float(
        by_method["registered_lineage_pooling"]["normalized_aurc"]
    )
    evidential = float(
        by_method["nested_evidential_composition"]["normalized_aurc"]
    )
    absolute_reduction = registered - evidential
    relative_reduction = absolute_reduction / registered * 100.0

    fixed_rows = read_rows(
        ROOT / "results" / "reference" / "fixed_target_summary.csv"
    )
    fixed = next(
        row
        for row in fixed_rows
        if row["method"] == "nested_evidential_composition"
    )
    admitted = int(float(fixed["admitted"]))
    total = int(float(fixed["n"]))
    correct = int(float(fixed["correct"]))
    wrong = int(float(fixed["wrong"]))
    achieved_coverage = admitted / total

    assert round(absolute_reduction, 4) == 0.0616
    assert round(relative_reduction, 1) == 29.4
    assert round(achieved_coverage, 4) == 0.1302
    assert admitted == correct + wrong
    assert wrong == 0

    print(f"absolute nAURC reduction: {absolute_reduction:.4f}")
    print(f"relative nAURC reduction: {relative_reduction:.1f}%")
    print(f"fixed-target coverage: {achieved_coverage:.4f}")
    print("Summary values verified.")


if __name__ == "__main__":
    main()
