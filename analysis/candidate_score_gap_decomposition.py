"""Decompose the product-minus-PACT native-score ncsAURC gap algebraically."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUSION = ROOT / "results" / "reference" / "pcecf_fusion_only.csv"
RANDOM = (
    ROOT
    / "results"
    / "equal_cardinality_topology"
    / "native_random_references.csv"
)
OUTPUT = (
    ROOT
    / "results"
    / "equal_cardinality_topology"
    / "candidate_score_gap_decomposition.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def main() -> None:
    fusion = keyed_rows(FUSION)
    random = keyed_rows(RANDOM)
    product_native = float(fusion["product_evidence_fusion"]["normalized_aurc_point"])
    pact_native = float(fusion["pcecf"]["normalized_aurc_point"])
    product_reference = float(
        random["product_evidence_fusion"]["eligible_random_reference"]
    )
    pact_reference = float(random["pcecf"]["eligible_random_reference"])

    total_gap = product_native - pact_native
    reference_gap = product_reference - pact_reference
    product_excess = product_native - product_reference
    pact_excess = pact_native - pact_reference
    excess_gap = product_excess - pact_excess
    if abs(total_gap - reference_gap - excess_gap) > 1e-12:
        raise AssertionError("Gap decomposition does not close")

    result = {
        "analysis": "algebraic candidate-formation and score-ordering decomposition",
        "status": "PASS",
        "support": [0.10, 0.39],
        "product_native_ncsAURC": product_native,
        "pact_native_ncsAURC": pact_native,
        "total_product_minus_pact_gap": total_gap,
        "product_random_order_reference": product_reference,
        "pact_random_order_reference": pact_reference,
        "candidate_conditional_reference_gap": reference_gap,
        "product_excess_above_reference": product_excess,
        "pact_excess_above_reference": pact_excess,
        "excess_above_reference_gap": excess_gap,
        "reference_gap_fraction": reference_gap / total_gap,
        "excess_gap_fraction": excess_gap / total_gap,
        "input_sha256": {
            str(FUSION.relative_to(ROOT)).replace("\\", "/"): sha256(FUSION),
            str(RANDOM.relative_to(ROOT)).replace("\\", "/"): sha256(RANDOM),
        },
        "claim_boundary": (
            "Exact algebraic decomposition of the observed native-score gap; "
            "not a causal mediation analysis."
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
