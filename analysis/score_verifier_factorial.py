"""Score-by-fusion and verifier-policy controls from the frozen controlled study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
sys.path[:0] = [str(REPOSITORY / "src"), str(REPOSITORY / "experiments"), str(HERE)]

import pcecf_study as study  # noqa: E402
import strong_paper_controls as controls  # noqa: E402
from action_admission import (  # noqa: E402
    CONTRACT_CLASSES,
    VerifierConfig,
    graph_from_parent_sets,
    verify_source_state,
)
from controlled_study import read_records, scene_fold_map, source_parents  # noqa: E402

OUTPUT = REPOSITORY / "outputs" / "score_verifier_factorial"
COMMON_GRID = np.linspace(0.10, 0.39, 36)
MAIN_METHODS = (
    "product_evidence_fusion",
    "nested_evidential_composition",
    "cautious_evidence_fusion",
    "pcecf",
)
POLICIES = {
    "A_unanimous_three_source": VerifierConfig(),
    "B_multi_source_three_component": VerifierConfig(
        minimum_registered_components=3,
        provenance_policy="multi_source_two_component",
    ),
    "C_required_role_two_component": VerifierConfig(
        minimum_registered_components=2,
        provenance_policy="required_role_two_component",
    ),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_score(row: Mapping[str, Any], score_name: str) -> float:
    probabilities = np.asarray(
        [float(row["probabilities"][label]) for label in CONTRACT_CLASSES],
        dtype=np.float64,
    )
    ordered = np.sort(probabilities)[::-1]
    if score_name == "native":
        return float(row["score"])
    if score_name == "posterior_peak":
        return float(ordered[0])
    if score_name == "top_two_margin":
        return float(ordered[0] - ordered[1])
    if score_name == "inverse_normalized_entropy":
        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
        return 1.0 - entropy / math.log(len(CONTRACT_CLASSES))
    raise ValueError(score_name)


def rescore(rows: Sequence[Mapping[str, Any]], score_name: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        copy = dict(row)
        copy["score"] = static_score(row, score_name)
        result.append(copy)
    return result


def fast_curve(
    rows_by_outer: Mapping[int, list[dict[str, Any]]],
    *,
    verifier: bool,
) -> list[dict[str, float]]:
    totals = {
        target: {"n": 0, "admitted": 0, "wrong": 0, "correct": 0}
        for target in controls.TARGETS
    }
    for outer in range(5):
        rows = rows_by_outer[outer]
        train = [row for row in rows if int(row["fold"]) != outer]
        test = [row for row in rows if int(row["fold"]) == outer]
        available_scores = sorted(
            (
                float(row["score"])
                for row in train
                if row["eligible"] and (row["verifier_pass"] or not verifier)
            ),
            reverse=True,
        )
        test_scores = np.asarray([float(row["score"]) for row in test])
        test_base = np.asarray(
            [bool(row["eligible"] and (row["verifier_pass"] or not verifier)) for row in test]
        )
        test_correct = np.asarray(
            [row["predicted_contract"] == row["preferred_contract"] for row in test]
        )
        for target in controls.TARGETS:
            count = int(round(target * len(train)))
            cutoff = float("inf") if not available_scores or count <= 0 else available_scores[min(count, len(available_scores)) - 1]
            accepted = test_base & (test_scores >= cutoff)
            cell = totals[target]
            cell["n"] += len(test)
            cell["admitted"] += int(np.sum(accepted))
            cell["wrong"] += int(np.sum(accepted & ~test_correct))
            cell["correct"] += int(np.sum(accepted & test_correct))
    curve = []
    for target in controls.TARGETS:
        cell = totals[target]
        n = cell["n"]
        admitted = cell["admitted"]
        curve.append(
            {
                "target": float(target),
                "n": int(n),
                "admitted": int(admitted),
                "wrong": int(cell["wrong"]),
                "correct": int(cell["correct"]),
                "coverage": admitted / n,
                "wrong_all": cell["wrong"] / n,
                "correct_all": cell["correct"] / n,
                "wrong_admitted": cell["wrong"] / admitted if admitted else 0.0,
            }
        )
    return curve


def curve_row(
    rows_by_outer: Mapping[int, list[dict[str, Any]]],
    *,
    method: str,
    score_name: str,
    verifier: bool,
    subset: str,
) -> dict[str, Any]:
    curve = fast_curve(rows_by_outer, verifier=verifier)
    point = next(row for row in curve if math.isclose(float(row["target"]), 0.13))
    return {
        "method": method,
        "score": score_name,
        "subset": subset,
        "verifier": verifier,
        "csaurc_0p10_0p39": study.interpolate_naurc(curve, COMMON_GRID),
        "coverage_at_target_0p13": point["coverage"],
        "wrong_all_at_target_0p13": point["wrong_all"],
        "correct_all_at_target_0p13": point["correct_all"],
    }


def held_out_rows(rows_by_outer: Mapping[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        row
        for outer, rows in rows_by_outer.items()
        for row in rows
        if int(row["fold"]) == outer
    ]


def macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for label in CONTRACT_CLASSES:
        tp = sum(row["preferred_contract"] == label and row["predicted_contract"] == label for row in rows)
        fp = sum(row["preferred_contract"] != label and row["predicted_contract"] == label for row in rows)
        fn = sum(row["preferred_contract"] == label and row["predicted_contract"] != label for row in rows)
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def posterior_metrics(
    rows_by_method: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    complete: Mapping[str, bool],
) -> list[dict[str, Any]]:
    result = []
    for method in MAIN_METHODS:
        oof = held_out_rows(rows_by_method[method])
        for subset in ("all", "complete"):
            selected = oof if subset == "all" else [row for row in oof if complete[row["record_id"]]]
            metrics = study.calibration(selected)
            result.append(
                {
                    "method": method,
                    "subset": subset,
                    "n": len(selected),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": macro_f1(selected),
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "ece10": metrics["ece10"],
                }
            )
    return result


def shared_scored_rows(
    rows_by_method: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    observed: Mapping[str, int],
    *,
    complete: Mapping[str, bool] | None = None,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in MAIN_METHODS}
    for outer in range(5):
        train = [
            row
            for method in MAIN_METHODS
            for row in rows_by_method[method][outer]
            if int(row["fold"]) != outer and (complete is None or complete[row["record_id"]])
        ]
        x_train = np.asarray([controls.posterior_features(row, observed[row["record_id"]]) for row in train])
        y_train = np.asarray([row["predicted_contract"] == row["preferred_contract"] for row in train], dtype=int)
        model = controls.fit_logistic(x_train, y_train)
        for method in MAIN_METHODS:
            rows = [
                row
                for row in rows_by_method[method][outer]
                if complete is None or complete[row["record_id"]]
            ]
            features = np.asarray([controls.posterior_features(row, observed[row["record_id"]]) for row in rows])
            scores = controls.predict_logistic(features, model)
            scored = []
            for row, score in zip(rows, scores, strict=True):
                copy = dict(row)
                copy["score"] = float(score)
                scored.append(copy)
            result[method][outer] = scored
    return result


def score_factorial(
    rows_by_method: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    observed: Mapping[str, int],
    complete: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = []
    for method in MAIN_METHODS:
        for score_name in ("native", "posterior_peak", "top_two_margin", "inverse_normalized_entropy"):
            scored = {outer: rescore(rows_by_method[method][outer], score_name) for outer in range(5)}
            for verifier in (False, True):
                result.append(curve_row(scored, method=method, score_name=score_name, verifier=verifier, subset="all"))
    shared = shared_scored_rows(rows_by_method, observed)
    for method in MAIN_METHODS:
        for verifier in (False, True):
            result.append(curve_row(shared[method], method=method, score_name="shared_outer_train_logistic", verifier=verifier, subset="all"))
    complete_shared = shared_scored_rows(rows_by_method, observed, complete=complete)
    complete_result = []
    for method in MAIN_METHODS:
        for verifier in (False, True):
            complete_result.append(curve_row(complete_shared[method], method=method, score_name="shared_outer_train_logistic", verifier=verifier, subset="complete"))
    return result, complete_result


def policy_rows(
    rows_by_method: Mapping[str, Mapping[int, list[dict[str, Any]]]],
    records: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    transitions = []
    graphs = {record_id: graph_from_parent_sets(source_parents(record)) for record_id, record in records.items()}
    for method in MAIN_METHODS:
        by_policy: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for policy, config in POLICIES.items():
            by_outer: dict[int, list[dict[str, Any]]] = {}
            for outer in range(5):
                updated = []
                for row in rows_by_method[method][outer]:
                    decision = verify_source_state(
                        records[row["record_id"]],
                        row["predicted_contract"],
                        graphs[row["record_id"]],
                        config=config,
                    )
                    copy = dict(row)
                    copy["verifier_pass"] = bool(decision.admissible)
                    copy["verifier_route"] = str(decision.route)
                    copy["verifier_reason"] = str(decision.reason)
                    updated.append(copy)
                    if policy == "A_unanimous_three_source":
                        assert bool(decision.admissible) == bool(row["verifier_pass"])
                        assert str(decision.reason) == str(row["verifier_reason"])
                by_outer[outer] = updated
            by_policy[policy] = by_outer
            summary.append(curve_row(by_outer, method=method, score_name=policy, verifier=True, subset="all"))
        current = held_out_rows(by_policy["A_unanimous_three_source"])
        for policy in ("B_multi_source_three_component", "C_required_role_two_component"):
            alternative = held_out_rows(by_policy[policy])
            counts: Counter[tuple[bool, bool, bool]] = Counter()
            for left, right in zip(current, alternative, strict=True):
                correct = left["predicted_contract"] == left["preferred_contract"]
                counts[(bool(left["verifier_pass"]), bool(right["verifier_pass"]), correct)] += 1
            for (current_pass, alternative_pass, correct), count in sorted(counts.items()):
                transitions.append(
                    {
                        "method": method,
                        "alternative_policy": policy,
                        "current_pass": current_pass,
                        "alternative_pass": alternative_pass,
                        "candidate_correct": correct,
                        "records": count,
                    }
                )
    return summary, transitions


def truth_table() -> list[dict[str, Any]]:
    scenarios = [
        (1, 1, False, "single source"),
        (2, 1, True, "language + physical, reused component"),
        (2, 2, True, "language + physical, distinct components"),
        (2, 1, False, "two physical sources, reused component"),
        (2, 2, False, "two physical sources, distinct components"),
        (3, 1, True, "three sources, one component"),
        (3, 2, True, "three sources, two components"),
        (3, 3, True, "three sources, three components"),
    ]
    result = []
    for supporters, components, roles, description in scenarios:
        current_applies = supporters >= 3
        current_confirm = current_applies and components < 3
        multi_applies = supporters >= 2
        multi_confirm = multi_applies and components < 3
        role_confirm = multi_applies and (components < 2 or not roles)
        result.append(
            {
                "supporting_sources": supporters,
                "supporting_components": components,
                "language_plus_physical_roles": roles,
                "scenario": description,
                "A_unanimous_three_source": "confirm" if current_confirm else ("admit" if current_applies else "not_invoked"),
                "B_multi_source_three_component": "confirm" if multi_confirm else ("admit" if multi_applies else "not_invoked"),
                "C_required_role_two_component": "confirm" if role_confirm else ("admit" if multi_applies else "not_invoked"),
            }
        )
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcecf-output", type=Path, default=study.OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    records_list = read_records(study.DATA)
    labels_list = read_records(study.LABELS)
    labels = {str(row["record_id"]): row for row in labels_list}
    folds = scene_fold_map(records_list, 5)
    fold_manifest = json.loads(
        (arguments.pcecf_output / "fold_manifest.json").read_text(encoding="utf-8")
    )
    concentrations = {
        int(row["outer_fold"]): float(row["fold_local_dirichlet_concentration"])
        for row in fold_manifest["folds"]
    }
    rows_by_method = study.make_rows(records_list, labels, folds, concentrations)
    records = {str(record["record_id"]): record for record in records_list}
    observed = {record_id: study.observed_source_count(record) for record_id, record in records.items()}
    complete = {record_id: count == len(study.SOURCE_NAMES) for record_id, count in observed.items()}

    predictive = posterior_metrics(rows_by_method, complete)
    factorial, complete_shared = score_factorial(rows_by_method, observed, complete)
    policies, transitions = policy_rows(rows_by_method, records)
    truth = truth_table()

    # Exact reproduction of the two canonical native PACT anchors guards the protocol.
    anchors = {(row["method"], row["verifier"]): row for row in factorial if row["score"] == "native"}
    print("native PACT anchors", anchors[("pcecf", False)]["csaurc_0p10_0p39"], anchors[("pcecf", True)]["csaurc_0p10_0p39"], flush=True)
    assert math.isclose(anchors[("pcecf", False)]["csaurc_0p10_0p39"], 0.6294374684440601, abs_tol=1e-12)
    assert math.isclose(anchors[("pcecf", True)]["csaurc_0p10_0p39"], 0.08612194505919865, abs_tol=1e-12)

    outputs = {
        "posterior_metrics.csv": predictive,
        "score_factorial.csv": factorial,
        "complete_shared_score.csv": complete_shared,
        "verifier_policy_summary.csv": policies,
        "verifier_policy_transitions.csv": transitions,
        "verifier_policy_truth_table.csv": truth,
    }
    for name, rows in outputs.items():
        write_csv(arguments.output / name, rows)
    manifest = {
        "status": "PASS",
        "records": len(records),
        "methods": list(MAIN_METHODS),
        "outer_fold_protocol": "each outer-fold score model and verifier threshold uses predictions generated under that fold's training-selected concentration",
        "current_policy_reproduction": "exact row-level match to canonical verifier outcomes",
        "policy_definitions": {
            "A_unanimous_three_source": "three-source unanimity; require three registered components",
            "B_multi_source_three_component": "at least two supporting sources; retain the current requirement of three registered components",
            "C_required_role_two_component": "at least two supporting sources; require language plus a physical role and two registered components",
        },
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256(path)
        for path in sorted(arguments.output.glob("*"))
        if path.is_file() and path.name != "sha256.json"
    }
    (arguments.output / "sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
