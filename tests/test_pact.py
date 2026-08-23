from __future__ import annotations

import unittest

import numpy as np

from action_admission import SourceEvidence, pact_fuse as forward, pact_registered_components as registered_components


def source(
    name: str,
    probabilities: tuple[float, ...],
    parents: tuple[str, ...],
    *,
    quality: float = 1.0,
    conflict: float = 0.0,
    missing: bool = False,
    evidence: tuple[float, ...] | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        source_id=name,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        quality=quality,
        conflict=conflict,
        missing=missing,
        parents=parents,
        evidence=None if evidence is None else np.asarray(evidence, dtype=np.float64),
    )


class PACTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.language = source("language", (0.8, 0.1, 0.1), ("command",))
        self.geometry = source("geometry", (0.7, 0.2, 0.1), ("scene",))
        self.risk = source("risk", (0.6, 0.3, 0.1), ("scene",))

    def test_output_is_normalized_and_finite(self) -> None:
        output = forward([self.language, self.geometry, self.risk], concentration=8.0)
        self.assertAlmostEqual(float(output.posterior.sum()), 1.0)
        self.assertTrue(np.all(np.isfinite(output.posterior)))
        self.assertEqual(output.predicted_index, 0)

    def test_exact_registered_duplicate_is_invariant(self) -> None:
        base = forward([self.language, self.geometry, self.risk], concentration=8.0)
        duplicate = source("geometry_copy", (0.7, 0.2, 0.1), ("scene",))
        copied = forward(
            [self.language, self.geometry, self.risk, duplicate], concentration=8.0
        )
        np.testing.assert_allclose(base.posterior, copied.posterior, atol=1e-12)
        self.assertAlmostEqual(base.selection_score, copied.selection_score)

    def test_distinct_copy_accumulates_evidence(self) -> None:
        base = forward([self.language, self.geometry], concentration=8.0)
        distinct = source("geometry_independent", (0.7, 0.2, 0.1), ("new_scene",))
        expanded = forward(
            [self.language, self.geometry, distinct], concentration=8.0
        )
        self.assertGreater(expanded.selection_score, base.selection_score)

    def test_near_copy_obeys_conservative_bound(self) -> None:
        base = forward([self.language, self.geometry, self.risk], concentration=8.0)
        near = source("geometry_near", (0.69, 0.21, 0.10), ("scene",))
        perturbed = forward(
            [self.language, self.geometry, self.risk, near], concentration=8.0
        )
        delta = 8.0 * np.linalg.norm(
            near.probabilities / near.probabilities.sum()
            - self.geometry.probabilities / self.geometry.probabilities.sum(),
            ord=1,
        )
        bound = 2.0 * delta / 3.0
        self.assertLessEqual(
            float(np.linalg.norm(perturbed.posterior - base.posterior, ord=1)),
            bound + 1e-12,
        )

    def test_available_zero_vector_is_rejected(self) -> None:
        invalid = source("invalid", (0.0, 0.0, 0.0), ("invalid_parent",))
        with self.assertRaisesRegex(ValueError, "positive mass"):
            forward([invalid], concentration=8.0)

    def test_same_component_insertion_never_increases_evidence(self) -> None:
        base = forward([self.language, self.geometry, self.risk], concentration=8.0)
        weak = source("geometry_weak", (0.2, 0.7, 0.1), ("scene",))
        expanded = forward(
            [self.language, self.geometry, self.risk, weak], concentration=8.0
        )
        self.assertTrue(
            np.all(expanded.group_evidence.sum(axis=0) <= base.group_evidence.sum(axis=0) + 1e-12)
        )

    def test_near_copy_score_bound(self) -> None:
        base = forward([self.language, self.geometry, self.risk], concentration=8.0)
        near = source("geometry_near_score", (0.69, 0.21, 0.10), ("scene",))
        expanded = forward(
            [self.language, self.geometry, self.risk, near], concentration=8.0
        )
        evidence_change = float(
            np.linalg.norm(
                expanded.group_evidence.sum(axis=0) - base.group_evidence.sum(axis=0),
                ord=1,
            )
        )
        prior = 3.0
        z = prior + float(base.group_evidence.sum())
        z_prime = prior + float(expanded.group_evidence.sum())
        self.assertLessEqual(
            abs(expanded.selection_score - base.selection_score),
            prior * evidence_change / (z * z_prime) + 1e-12,
        )

    def test_missing_slot_contributes_zero_evidence(self) -> None:
        missing = source(
            "geometry_missing", (0.7, 0.2, 0.1), ("scene",), missing=True
        )
        output = forward([self.language, self.geometry, missing], concentration=8.0)
        scene_group = next(
            row for ids, row in zip(output.group_ids, output.group_evidence) if "geometry" in ids
        )
        np.testing.assert_allclose(scene_group, np.zeros(3), atol=1e-12)

    def test_direct_evidence_is_discounted(self) -> None:
        direct = source(
            "direct",
            (0.5, 0.3, 0.2),
            ("direct_parent",),
            quality=0.5,
            conflict=0.2,
            evidence=(4.0, 2.0, 0.0),
        )
        output = forward([direct], concentration=99.0)
        np.testing.assert_allclose(output.group_evidence[0], (1.6, 0.8, 0.0))

    def test_transitive_parent_overlap_forms_one_component(self) -> None:
        a = source("a", (0.7, 0.2, 0.1), ("p1",))
        b = source("b", (0.6, 0.3, 0.1), ("p1", "p2"))
        c = source("c", (0.5, 0.4, 0.1), ("p2",))
        self.assertEqual(registered_components([a, b, c]), ((0, 1, 2),))

    def test_empty_parent_set_is_rejected(self) -> None:
        invalid = source("invalid", (0.7, 0.2, 0.1), ())
        with self.assertRaisesRegex(ValueError, "registered parents"):
            forward([invalid], concentration=8.0)

    def test_catalog_order_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match expected"):
            forward(
                [self.geometry, self.language],
                concentration=8.0,
                expected_source_ids=("language", "geometry"),
            )

    def test_duplicate_source_identifier_is_rejected(self) -> None:
        duplicate = source("language", (0.6, 0.2, 0.2), ("other",))
        with self.assertRaisesRegex(ValueError, "identifiers must be unique"):
            forward([self.language, duplicate], concentration=8.0)


if __name__ == "__main__":
    unittest.main()
