"""Frozen-policy cost and Pareto sensitivity for the controlled PACT study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "results" / "decision_cost"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"), str(ROOT / "analysis")]

import pcecf_study as study  # noqa: E402
import score_verifier_factorial as svf  # noqa: E402
from action_admission import VerifierConfig, graph_from_parent_sets, verify_source_state  # noqa: E402
from controlled_study import read_records, scene_fold_map, source_parents  # noqa: E402

TARGET = 0.13
RATIOS = (1, 2, 5, 10, 20, 50)
BOOTSTRAPS = 2000
SEED = 2052
POLICIES = {
    **svf.POLICIES,
    "A_unanimous_two_component": VerifierConfig(minimum_registered_components=2),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_policy(
    rows_by_outer: Mapping[int, list[dict[str, Any]]],
    records: Mapping[str, dict[str, Any]],
    config: VerifierConfig,
) -> dict[int, list[dict[str, Any]]]:
    graphs = {
        record_id: graph_from_parent_sets(source_parents(record))
        for record_id, record in records.items()
    }
    result: dict[int, list[dict[str, Any]]] = {}
    for outer, rows in rows_by_outer.items():
        updated = []
        for row in rows:
            decision = verify_source_state(
                records[row["record_id"]],
                row["predicted_contract"],
                graphs[row["record_id"]],
                config=config,
            )
            copy = dict(row)
            copy["verifier_pass"] = bool(decision.admissible)
            updated.append(copy)
        result[outer] = updated
    return result


def heldout_decisions(
    rows_by_outer: Mapping[int, list[dict[str, Any]]],
    records: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions = []
    for outer in range(5):
        rows = rows_by_outer[outer]
        train = [row for row in rows if int(row["fold"]) != outer]
        test = [row for row in rows if int(row["fold"]) == outer]
        scores = sorted(
            (
                float(row["score"])
                for row in train
                if row["eligible"] and row["verifier_pass"]
            ),
            reverse=True,
        )
        count = int(round(TARGET * len(train)))
        cutoff = float("inf") if not scores or count <= 0 else scores[min(count, len(scores)) - 1]
        for row in test:
            admitted = bool(
                row["eligible"]
                and row["verifier_pass"]
                and float(row["score"]) >= cutoff
            )
            correct = row["predicted_contract"] == row["preferred_contract"]
            record = records[row["record_id"]]
            decisions.append(
                {
                    "record_id": row["record_id"],
                    "scene_id": str(record["metadata"]["scene_id"]),
                    "admitted": admitted,
                    "correct": bool(correct),
                }
            )
    assert len(decisions) == len(records)
    assert len({row["record_id"] for row in decisions}) == len(records)
    return decisions


def summarize_policy(name: str, rows_by_outer: Mapping[int, list[dict[str, Any]]]) -> dict[str, Any]:
    row = svf.curve_row(
        rows_by_outer,
        method="pcecf",
        score_name=name,
        verifier=True,
        subset="all",
    )
    return {
        "policy": name,
        "ncsaurc_0p10_0p39": row["csaurc_0p10_0p39"],
        "coverage_at_0p13": row["coverage_at_target_0p13"],
        "wrong_all_at_0p13": row["wrong_all_at_target_0p13"],
        "correct_all_at_0p13": row["correct_all_at_target_0p13"],
    }


def scene_arrays(
    decisions_by_policy: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    scenes = sorted({row["scene_id"] for rows in decisions_by_policy.values() for row in rows})
    arrays: dict[str, np.ndarray] = {}
    for policy, rows in decisions_by_policy.items():
        by_scene = {scene: [0, 0, 0, 0] for scene in scenes}  # n, admitted, wrong, correct
        for row in rows:
            cell = by_scene[row["scene_id"]]
            cell[0] += 1
            cell[1] += int(row["admitted"])
            cell[2] += int(row["admitted"] and not row["correct"])
            cell[3] += int(row["admitted"] and row["correct"])
        arrays[policy] = np.asarray([by_scene[scene] for scene in scenes], dtype=np.int64)
    return scenes, arrays


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    records_list = read_records(study.DATA)
    labels = {str(row["record_id"]): row for row in read_records(study.LABELS)}
    folds = scene_fold_map(records_list, 5)
    fold_manifest = json.loads((study.OUTPUT / "fold_manifest.json").read_text(encoding="utf-8"))
    concentrations = {
        int(row["outer_fold"]): float(row["fold_local_dirichlet_concentration"])
        for row in fold_manifest["folds"]
    }
    all_rows = study.make_rows(records_list, labels, folds, concentrations)
    pact_rows = all_rows["pcecf"]
    records = {str(record["record_id"]): record for record in records_list}

    rows_by_policy = {
        name: apply_policy(pact_rows, records, config)
        for name, config in POLICIES.items()
    }
    summaries = [summarize_policy(name, rows) for name, rows in rows_by_policy.items()]
    summary_by_name = {row["policy"]: row for row in summaries}

    # Exact frozen anchors guard the analysis path.
    assert math.isclose(
        summary_by_name["A_unanimous_three_source"]["ncsaurc_0p10_0p39"],
        0.08612194505919865,
        abs_tol=1e-12,
    )
    assert math.isclose(
        summary_by_name["C_required_role_two_component"]["ncsaurc_0p10_0p39"],
        0.3891164502870042,
        abs_tol=1e-12,
    )

    decisions = {
        name: heldout_decisions(rows, records)
        for name, rows in rows_by_policy.items()
    }
    # Boundaries use the same held-out PACT candidates.
    pact_heldout = svf.held_out_rows(pact_rows)
    decisions["always_withhold"] = [
        {
            "record_id": row["record_id"],
            "scene_id": str(records[row["record_id"]]["metadata"]["scene_id"]),
            "admitted": False,
            "correct": row["predicted_contract"] == row["preferred_contract"],
        }
        for row in pact_heldout
    ]
    decisions["always_continue"] = [
        {**row, "admitted": True}
        for row in decisions["always_withhold"]
    ]

    scenes, arrays = scene_arrays(decisions)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(scenes), size=(BOOTSTRAPS, len(scenes)))
    base = "A_unanimous_three_source"
    cost_rows = []
    bootstrap_rows = []
    for ratio in RATIOS:
        boot_cost: dict[str, np.ndarray] = {}
        point_cost: dict[str, float] = {}
        for policy, values in arrays.items():
            total = values.sum(axis=0)
            point_cost[policy] = float((ratio * total[2] + total[0] - total[1]) / total[0])
            sampled = values[samples].sum(axis=1)
            boot_cost[policy] = (ratio * sampled[:, 2] + sampled[:, 0] - sampled[:, 1]) / sampled[:, 0]
        best = min(point_cost.values())
        for policy in arrays:
            low, high = ci(boot_cost[policy])
            delta = boot_cost[policy] - boot_cost[base]
            delta_low, delta_high = ci(delta)
            cost_rows.append(
                {
                    "wrong_to_nonadmission_ratio": ratio,
                    "policy": policy,
                    "relative_cost": point_cost[policy],
                    "ci_low": low,
                    "ci_high": high,
                    "delta_vs_A": point_cost[policy] - point_cost[base],
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "point_optimal": math.isclose(point_cost[policy], best, abs_tol=1e-15),
                }
            )
        bootstrap_rows.append(
            {
                "wrong_to_nonadmission_ratio": ratio,
                "point_optimal_policies": [name for name, value in point_cost.items() if math.isclose(value, best, abs_tol=1e-15)],
                "minimum_relative_cost": best,
            }
        )

    # Pareto status at the fixed operating point; no outcome-based policy promotion follows.
    operating = {
        row["policy"]: (row["wrong_all_at_0p13"], row["correct_all_at_0p13"])
        for row in summaries
    }
    pareto_rows = []
    for policy, (wrong, correct) in operating.items():
        dominators = [
            other
            for other, (other_wrong, other_correct) in operating.items()
            if other != policy
            and other_wrong <= wrong
            and other_correct >= correct
            and (other_wrong < wrong or other_correct > correct)
        ]
        pareto_rows.append(
            {
                **summary_by_name[policy],
                "pareto_nondominated": not dominators,
                "dominated_by": ";".join(dominators),
            }
        )

    write_csv(HERE / "policy_summary.csv", summaries)
    write_csv(HERE / "policy_pareto.csv", pareto_rows)
    write_csv(HERE / "cost_ratio_sensitivity.csv", cost_rows)
    (HERE / "cost_optima.json").write_text(
        json.dumps(bootstrap_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = {
        "status": "PASS",
        "analysis_unit": "scene-clustered held-out decision record",
        "scenes": len(scenes),
        "records": len(records),
        "target_coverage": TARGET,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "cost_definition": "ratio * wrong_admission + nonadmission, divided by all records; always-withhold = 1",
        "manuscript_weights": "5:0.5 corresponds to ratio 10",
        "policies_frozen_before_run": list(POLICIES),
        "policy_selection": "none; all prespecified policies and boundaries retained",
        "nu_sensitivity": "A_unanimous_three_source versus A_unanimous_two_component holds the trigger semantics fixed",
        "cost_optima": bootstrap_rows,
    }
    (HERE / "gate.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        path.name: sha256(path)
        for path in sorted(HERE.iterdir())
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    (HERE / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in manifest.items()),
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
