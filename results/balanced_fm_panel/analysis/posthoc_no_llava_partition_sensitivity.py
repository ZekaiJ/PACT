#!/usr/bin/env python3
"""Recompute the family-pair statistic after removing both LLaVA checkpoints."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


REGISTERED = (
    ("qwen3vl_8b", "qwen3vl_32b"),
    ("internvl3_2b", "internvl3_8b"),
    ("smolvlm2_0_5b", "smolvlm2_2_2b"),
)


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def pairings(items: tuple[str, ...]):
    if not items:
        yield ()
        return
    first = items[0]
    for index in range(1, len(items)):
        second = items[index]
        remainder = items[1:index] + items[index + 1 :]
        for rest in pairings(remainder):
            yield (canonical_pair(first, second),) + rest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.pairs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    phi = {
        canonical_pair(row["left_model"], row["right_model"]): float(row["checkpoint_pair_phi"])
        for row in rows
    }

    models = tuple(sorted(set(itertools.chain.from_iterable(REGISTERED))))
    registered = tuple(sorted(canonical_pair(*pair) for pair in REGISTERED))
    all_pairs = sorted({tuple(sorted(p)) for p in pairings(models)})
    assert len(models) == 6 and len(all_pairs) == 15

    results = []
    for partition in all_pairs:
        within_keys = set(partition)
        within = [phi[key] for key in within_keys]
        between = [value for key, value in phi.items() if key[0] in models and key[1] in models and key not in within_keys]
        contrast = sum(within) / len(within) - sum(between) / len(between)
        results.append((partition, contrast))

    results.sort(key=lambda item: item[1], reverse=True)
    registered_contrast = next(value for partition, value in results if partition == registered)
    tolerance = 1e-14
    rank = 1 + sum(value > registered_contrast + tolerance for _, value in results)
    exact_p = sum(value >= registered_contrast - tolerance for _, value in results) / len(results)
    distinct_values = len({round(value, 14) for _, value in results})
    minimum_p = 1 / len(results)

    # Small executable guard against changing the estimand or pairing enumeration.
    assert abs(registered_contrast - 0.406037607001406) < 1e-12
    assert rank == 3
    assert abs(exact_p - 0.2) < 1e-12
    assert distinct_values == 15
    assert minimum_p > 0.05

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "NO_LLAVA_PAIR_PARTITIONS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "partition", "phi_contrast", "is_registered"))
        for position, (partition, value) in enumerate(results, start=1):
            writer.writerow((position, " | ".join(" + ".join(pair) for pair in partition), f"{value:.16g}", partition == registered))

    summary = {
        "analysis_status": "POST_HOC_SENSITIVITY",
        "excluded_checkpoints": ["llava_onevision_0_5b", "llava_onevision_7b"],
        "remaining_checkpoints": list(models),
        "partition_count": len(results),
        "registered_phi_contrast": registered_contrast,
        "registered_rank": rank,
        "exact_one_sided_p": exact_p,
        "distinct_statistic_values": distinct_values,
        "minimum_attainable_one_sided_p": minimum_p,
        "can_reject_at_alpha_0_05": False,
        "interpretation": "Removing both constant-READY LLaVA checkpoints does not confirm the declared family pairing.",
    }
    (args.output_dir / "NO_LLAVA_PARTITION_SENSITIVITY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
