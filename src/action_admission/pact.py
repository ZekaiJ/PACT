"""Public PACT fusion interface.

The implementation retains the earlier ``pcecf`` module name so released
commands and artifact hashes remain stable. New code should import the PACT
names defined here.
"""

from __future__ import annotations

from typing import Sequence

from .pcecf import PCECFOutput, SourceEvidence, forward, registered_components


PACTOutput = PCECFOutput


def pact_fuse(
    sources: Sequence[SourceEvidence],
    *,
    concentration: float,
    prior_per_class: float = 1.0,
    expected_source_ids: Sequence[str] | None = None,
) -> PACTOutput:
    """Fuse source evidence with the PACT conservation operator."""

    return forward(
        sources,
        concentration=concentration,
        prior_per_class=prior_per_class,
        expected_source_ids=expected_source_ids,
    )


def provenance_components(
    sources: Sequence[SourceEvidence],
) -> tuple[tuple[int, ...], ...]:
    """Return components induced by overlap of provenance parent sets."""

    return registered_components(sources)


__all__ = [
    "PACTOutput",
    "SourceEvidence",
    "pact_fuse",
    "provenance_components",
]
