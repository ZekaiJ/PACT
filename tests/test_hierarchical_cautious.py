from __future__ import annotations

import unittest

import numpy as np

from action_admission import (
    CONTRACT_CLASSES,
    SourceEvidence,
    hierarchy_matched_cautious,
    pcecf_fuse,
    pcecf_registered_components,
)


class HierarchyMatchedCautiousTests(unittest.TestCase):
    def test_exact_registered_copy_preserves_posterior_and_native_score(self) -> None:
        sources = [
            SourceEvidence(
                "language",
                np.asarray((0.70, 0.10, 0.08, 0.07, 0.05)),
                0.90,
                0.10,
                False,
                ("command",),
            ),
            SourceEvidence(
                "geometry",
                np.asarray((0.55, 0.18, 0.12, 0.10, 0.05)),
                0.85,
                0.05,
                False,
                ("scene",),
            ),
            SourceEvidence(
                "risk",
                np.asarray((0.45, 0.20, 0.15, 0.12, 0.08)),
                0.80,
                0.15,
                False,
                ("scene",),
            ),
        ]
        base = hierarchy_matched_cautious(sources)
        copied = hierarchy_matched_cautious(
            [
                *sources,
                SourceEvidence(
                    "geometry_copy",
                    sources[1].probabilities.copy(),
                    sources[1].quality,
                    sources[1].conflict,
                    sources[1].missing,
                    sources[1].parents,
                ),
            ]
        )

        np.testing.assert_allclose(
            [base.probabilities[label] for label in CONTRACT_CLASSES],
            [copied.probabilities[label] for label in CONTRACT_CLASSES],
            atol=1e-12,
        )
        self.assertAlmostEqual(base.selection_score, copied.selection_score)
        self.assertAlmostEqual(base.selection_score, 1.0 - base.frame_mass)

    def test_five_class_pact_is_not_hierarchy_matched_cautious(self) -> None:
        sources = [
            SourceEvidence(
                "a",
                np.asarray((0.70, 0.10, 0.08, 0.07, 0.05)),
                0.8,
                0.0,
                False,
                ("root_a",),
            ),
            SourceEvidence(
                "b",
                np.asarray((0.55, 0.18, 0.12, 0.10, 0.05)),
                0.7,
                0.0,
                False,
                ("root_b",),
            ),
        ]
        self.assertEqual(pcecf_registered_components(sources), ((0,), (1,)))
        pact = pcecf_fuse(sources, concentration=4.0)
        cautious = hierarchy_matched_cautious(sources)
        cautious_posterior = np.asarray(
            [cautious.probabilities[label] for label in CONTRACT_CLASSES],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            pact.posterior,
            (
                0.4345454545454545,
                0.16581818181818184,
                0.14472727272727273,
                0.13672727272727273,
                0.11818181818181818,
            ),
            atol=1e-15,
            rtol=0.0,
        )
        self.assertAlmostEqual(pact.selection_score, 0.5454545454545454)
        np.testing.assert_allclose(
            cautious_posterior,
            (
                0.6987237867694891,
                0.10538517037860604,
                0.07891468650573055,
                0.06907400617408066,
                0.04790235017209363,
            ),
            atol=1e-15,
            rtol=0.0,
        )
        self.assertAlmostEqual(cautious.selection_score, 0.9112919441257525)
        self.assertFalse(np.allclose(pact.posterior, cautious_posterior))
        self.assertNotAlmostEqual(pact.selection_score, cautious.selection_score)


if __name__ == "__main__":
    unittest.main()
