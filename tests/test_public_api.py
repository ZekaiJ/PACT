import unittest

import numpy as np

from action_admission import (
    AdmissionConfig,
    AdmissionDecision,
    PACTOutput,
    SourceEvidence,
    pact_fuse,
    provenance_components,
    typed_admit,
)
from action_admission.pcecf import PCECFOutput
from action_admission.verifier import (
    VerificationDecision,
    VerifierConfig,
    verify_source_state,
)


class PublicAPITests(unittest.TestCase):
    def test_pact_names_preserve_the_reference_implementation(self):
        sources = [
            SourceEvidence(
                source_id="language",
                probabilities=np.asarray([0.8, 0.2]),
                quality=1.0,
                conflict=0.0,
                missing=False,
                parents=("command",),
            ),
            SourceEvidence(
                source_id="geometry",
                probabilities=np.asarray([0.7, 0.3]),
                quality=1.0,
                conflict=0.0,
                missing=False,
                parents=("scene",),
            ),
        ]

        result = pact_fuse(sources, concentration=8.0)

        self.assertIsInstance(result, PACTOutput)
        self.assertIsInstance(result, PCECFOutput)
        self.assertEqual(provenance_components(sources), ((1,), (0,)))

    def test_admission_names_are_backward_compatible(self):
        self.assertIs(AdmissionConfig, VerifierConfig)
        self.assertIs(AdmissionDecision, VerificationDecision)
        self.assertIs(typed_admit, verify_source_state)


if __name__ == "__main__":
    unittest.main()
