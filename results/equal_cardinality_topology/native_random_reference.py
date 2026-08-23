"""Derive random-order references from the released OOF predictions."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "results" / "p0_estimand_closure" / "v1" / "inputs" / "oof_predictions.jsonl.gz"
FACTORIAL = ROOT / "results" / "score_verifier_factorial" / "score_factorial.csv"
METHODS = (
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "pcecf",
)
EXPECTED_HASHES = {
    OOF.name: "7337911F20D36DFBB9AE66FC40BB2E6BDF63AFD60ABF0070837921C518DA2A1C",
    FACTORIAL.name: "4B50A88284F39C55112DA1FB04287E2613863A53A05520D6EBA01A4EF23AD833",
}
BOOTSTRAPS = 2000
SEED = 56202


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    for path in (OOF, FACTORIAL):
        if sha256(path) != EXPECTED_HASHES[path.name]:
            raise AssertionError(f"frozen input changed: {path}")

    native = {}
    with FACTORIAL.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] in METHODS
                and row["score"] == "native"
                and row["subset"] == "all"
                and row["verifier"] == "False"
            ):
                native[row["method"]] = float(row["csaurc_0p10_0p39"])
    if set(native) != set(METHODS):
        raise AssertionError("native-score summary is incomplete")

    by_method_scene: dict[str, dict[str, list[int]]] = {
        method: defaultdict(lambda: [0, 0]) for method in METHODS
    }
    with gzip.open(OOF, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            method = str(row["method"])
            if method not in METHODS or not bool(row["common_eligible"]):
                continue
            cell = by_method_scene[method][str(row["scene_id"])]
            cell[0] += 1
            cell[1] += int(row["prediction"] != row["y"])

    scenes = sorted(by_method_scene["pcecf"])
    if len(scenes) != 48 or any(sorted(values) != scenes for values in by_method_scene.values()):
        raise AssertionError("scene support differs across methods")
    arrays = {
        method: np.asarray([by_method_scene[method][scene] for scene in scenes], dtype=float)
        for method in METHODS
    }
    rng = np.random.default_rng(SEED)
    draws = {method: np.empty(BOOTSTRAPS) for method in METHODS}
    for replicate in range(BOOTSTRAPS):
        weights = rng.multinomial(len(scenes), np.full(len(scenes), 1.0 / len(scenes)))
        for method in METHODS:
            total = arrays[method].T @ weights
            draws[method][replicate] = total[1] / total[0]

    rows = []
    for method in METHODS:
        total = arrays[method].sum(axis=0)
        reference = float(total[1] / total[0])
        rows.append(
            {
                "method": method,
                "common_eligible_records": int(total[0]),
                "eligible_random_reference": reference,
                "random_reference_bootstrap_mean": float(np.mean(draws[method])),
                "random_reference_ci_low": float(np.quantile(draws[method], 0.025)),
                "random_reference_ci_high": float(np.quantile(draws[method], 0.975)),
                "native_ncsaurc_0p10_0p39": native[method],
                "native_minus_random": native[method] - reference,
            }
        )

    output = HERE / "native_random_references.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
