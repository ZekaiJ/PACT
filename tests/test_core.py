from __future__ import annotations

import copy
import gzip
import importlib.util
import inspect
import json
import math
import unittest
from pathlib import Path

from action_admission import (
    SourceOpinion,
    VerifierConfig,
    dirichlet_predict,
    graph_from_parent_sets,
    restrict_dirichlet_input,
    verify_source_state,
)
from action_admission.lineage import registered_component_count, unnormalized_group_mass


ROOT = Path(__file__).resolve().parents[1]


def load_controlled_module():
    path = ROOT / "experiments" / "controlled_study.py"
    spec = importlib.util.spec_from_file_location("controlled_study_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load controlled-study module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pcecf_study_module():
    path = ROOT / "experiments" / "pcecf_study.py"
    spec = importlib.util.spec_from_file_location("pcecf_study_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load PC-ECF study module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_controlled_record() -> dict:
    path = ROOT / "data" / "controlled" / "source_records.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.loads(next(handle))


def load_record() -> dict:
    return json.loads(
        (ROOT / "data" / "examples" / "source_record.json").read_text(
            encoding="utf-8"
        )
    )


def peaked(label: str) -> dict[str, float]:
    probabilities = {
        "normal": 0.025,
        "slow_clearance": 0.025,
        "hold_confirm": 0.025,
        "bounded_urgent": 0.025,
        "retreat_fallback": 0.025,
    }
    probabilities[label] = 0.9
    return probabilities


class CoreMethodTests(unittest.TestCase):
    def test_dirichlet_prediction_is_normalized(self) -> None:
        record = load_record()
        restricted = restrict_dirichlet_input(record)
        self.assertFalse(hasattr(restricted, "registered_parents"))
        self.assertTrue(
            all(
                set(vars(source_input))
                == {"probabilities", "quality", "conflict", "missing"}
                for source_input in restricted.sources.values()
            )
        )
        with self.assertRaises(TypeError):
            dirichlet_predict(record)
        prediction = dirichlet_predict(restricted)
        self.assertTrue(prediction.eligible)
        self.assertEqual(prediction.predicted_contract, "hold_confirm")
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0)
        self.assertAlmostEqual(
            prediction.selection_score,
            1.0 - prediction.uncertainty,
        )

    def test_exact_copy_group_mass_is_invariant_at_lambda_one(self) -> None:
        single = [
            SourceOpinion(
                "language",
                peaked("hold_confirm"),
                quality=0.8,
            )
        ]
        copies = [
            SourceOpinion(
                "language",
                peaked("hold_confirm"),
                quality=0.8,
            ),
            SourceOpinion(
                "geometry",
                peaked("hold_confirm"),
                quality=0.8,
            ),
        ]
        single_mass = unnormalized_group_mass(single, {}, lineage_exponent=1.0)
        copied_mass = unnormalized_group_mass(
            copies,
            {("language", "geometry"): 1.0},
            lineage_exponent=1.0,
        )
        self.assertTrue(math.isclose(single_mass, copied_mass, rel_tol=1e-12))

    def test_support_count_uses_full_registered_partition(self) -> None:
        graph = {("left", "bridge"): 1.0, ("bridge", "right"): 1.0}
        self.assertEqual(registered_component_count(("left", "right"), graph), 2)
        self.assertEqual(
            registered_component_count(
                ("left", "right"),
                graph,
                partition_nodes=("left", "bridge", "right"),
            ),
            1,
        )

    def test_two_registered_groups_request_confirmation(self) -> None:
        record = load_record()
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        decision = verify_source_state(
            record,
            candidate.predicted_contract,
            graph,
        )
        self.assertFalse(decision.admissible)
        self.assertEqual(
            (decision.route, decision.reason),
            ("confirm", "insufficient_registered_corroboration"),
        )
        self.assertEqual(decision.diagnostics["registered_components"], 2)

    def test_two_group_sensitivity_setting_admits(self) -> None:
        record = load_record()
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        decision = verify_source_state(
            record,
            candidate.predicted_contract,
            graph,
            config=VerifierConfig(minimum_registered_components=2),
        )
        self.assertTrue(decision.admissible)

    def test_exactly_two_sources_do_not_invoke_registered_corroboration(self) -> None:
        record = load_record()
        record["sources"]["geometry"]["missing"] = True
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        decision = verify_source_state(
            record,
            candidate.predicted_contract,
            graph,
        )
        self.assertTrue(decision.admissible)
        self.assertEqual((decision.route, decision.reason), ("none", "evidence_eligible"))
        self.assertEqual(decision.diagnostics["available_sources"], 2)
        self.assertFalse(
            decision.diagnostics["insufficient_registered_corroboration"]
        )

    def test_multi_source_policy_accepts_two_distinct_components(self) -> None:
        record = load_record()
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        decision = verify_source_state(
            record,
            candidate.predicted_contract,
            graph,
            config=VerifierConfig(
                minimum_registered_components=2,
                provenance_policy="multi_source_two_component",
            ),
        )
        self.assertTrue(decision.admissible)
        self.assertEqual(decision.diagnostics["registered_components"], 2)

    def test_multi_source_policy_rejects_one_reused_component(self) -> None:
        record = load_record()
        record["sources"]["language"]["missing"] = True
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        decision = verify_source_state(
            record,
            candidate.predicted_contract,
            graph,
            config=VerifierConfig(
                minimum_registered_components=2,
                provenance_policy="multi_source_two_component",
            ),
        )
        self.assertFalse(decision.admissible)
        self.assertEqual(decision.reason, "insufficient_registered_corroboration")
        self.assertEqual(decision.diagnostics["registered_components"], 1)

    def test_required_role_policy_distinguishes_role_composition(self) -> None:
        record = load_record()
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        config = VerifierConfig(
            minimum_registered_components=2,
            provenance_policy="required_role_two_component",
        )
        language_and_risk = copy.deepcopy(record)
        language_and_risk["sources"]["geometry"]["missing"] = True
        self.assertTrue(
            verify_source_state(
                language_and_risk, candidate.predicted_contract, graph, config=config
            ).admissible
        )
        geometry_and_risk = copy.deepcopy(record)
        geometry_and_risk["sources"]["language"]["missing"] = True
        self.assertFalse(
            verify_source_state(
                geometry_and_risk, candidate.predicted_contract, graph, config=config
            ).admissible
        )

    def test_common_eligibility_requires_structural_validity(self) -> None:
        study = load_pcecf_study_module()
        record = load_controlled_record()
        record["sources"]["geometry"]["missing"] = True
        record["sources"]["language"]["schema_valid"] = False
        self.assertEqual(study.observed_source_count(record), 1)
        self.assertFalse(study.common_eligibility(record))

    def test_available_zero_mass_source_fails_structural_verification(self) -> None:
        record = load_record()
        record["sources"]["language"]["probabilities"] = {
            label: 0.0 for label in record["sources"]["language"]["probabilities"]
        }
        graph = graph_from_parent_sets(record["registered_parents"])
        decision = verify_source_state(record, "hold_confirm", graph)
        self.assertFalse(decision.admissible)
        self.assertEqual((decision.route, decision.reason), ("fallback", "incomplete_source_schema"))
        self.assertIn("language.probabilities", decision.diagnostics["schema_issues"])

    def test_evaluation_fields_do_not_change_verification(self) -> None:
        record = load_record()
        graph = graph_from_parent_sets(record["registered_parents"])
        candidate = dirichlet_predict(restrict_dirichlet_input(record))
        augmented = copy.deepcopy(record)
        augmented.update(
            {
                "reference_contract": "retreat_fallback",
                "stress_label": "arbitrary",
                "wrong_authorization": True,
                "execution_log": {"contact": True},
            }
        )
        self.assertEqual(
            verify_source_state(
                record,
                candidate.predicted_contract,
                graph,
            ),
            verify_source_state(
                augmented,
                candidate.predicted_contract,
                graph,
            ),
        )
        source_code = inspect.getsource(verify_source_state)
        for field in (
            "reference_contract",
            "stress_label",
            "wrong_authorization",
            "execution_log",
        ):
            self.assertNotIn(field, source_code)

    def test_prediction_is_label_free_until_evaluation_join(self) -> None:
        controlled = load_controlled_module()
        record = load_controlled_record()
        prediction = controlled.prediction_row(
            record,
            0,
            method="nested_evidential_composition",
            concentration=8.0,
        )
        self.assertNotIn("preferred_contract", prediction)
        joined = controlled.join_evaluation_labels(
            [prediction],
            {
                prediction["record_id"]: {
                    "record_id": prediction["record_id"],
                    "scene_id": prediction["scene_id"],
                    "preferred_contract": "normal",
                }
            },
        )
        self.assertEqual(joined[0]["preferred_contract"], "normal")

    def test_all_controlled_predictors_are_label_free(self) -> None:
        controlled = load_controlled_module()
        record = load_controlled_record()
        for method in controlled.METHODS:
            prediction = controlled.prediction_row(
                record,
                0,
                method=method,
                concentration=(
                    8.0 if method == "nested_evidential_composition" else None
                ),
            )
            self.assertNotIn("preferred_contract", prediction)

    def test_label_join_rejects_scene_mismatch(self) -> None:
        controlled = load_controlled_module()
        prediction = {
            "record_id": "r1",
            "scene_id": "scene-a",
            "predicted_contract": "normal",
        }
        with self.assertRaises(ValueError):
            controlled.join_evaluation_labels(
                [prediction],
                {
                    "r1": {
                        "record_id": "r1",
                        "scene_id": "scene-b",
                        "preferred_contract": "normal",
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()

