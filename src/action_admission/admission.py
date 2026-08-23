"""Public typed-admission interface."""

from .verifier import (
    VerificationDecision,
    VerifierConfig,
    verify_source_state,
)


AdmissionConfig = VerifierConfig
AdmissionDecision = VerificationDecision
typed_admit = verify_source_state


__all__ = [
    "AdmissionConfig",
    "AdmissionDecision",
    "typed_admit",
]
