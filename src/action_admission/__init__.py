"""PACT evidence fusion and typed action-admission reference implementation."""

from .contracts import CONTRACT_CLASSES, OBSERVABLE_SOURCES
from .dirichlet import (
    DirichletInput,
    EvidentialPrediction,
    predict as dirichlet_predict,
    restrict_input as restrict_dirichlet_input,
)
from .lineage import (
    SourceOpinion,
    graph_from_parent_sets,
    log_linear_posterior,
    registered_component_count,
)
from .hierarchical_cautious import (
    HierarchyMatchedCautiousOutput,
    hierarchy_matched_cautious,
)
from .admission import AdmissionConfig, AdmissionDecision, typed_admit
from .pact import PACTOutput, pact_fuse, provenance_components
from .pcecf import (
    PCECFOutput,
    SourceEvidence,
    forward as pcecf_fuse,
    registered_components as pcecf_registered_components,
)
from .verifier import VerificationDecision, VerifierConfig, verify_source_state

pact_registered_components = pcecf_registered_components

__all__ = [
    "CONTRACT_CLASSES",
    "OBSERVABLE_SOURCES",
    "AdmissionConfig",
    "AdmissionDecision",
    "PACTOutput",
    "PCECFOutput",
    "DirichletInput",
    "EvidentialPrediction",
    "HierarchyMatchedCautiousOutput",
    "SourceOpinion",
    "SourceEvidence",
    "VerificationDecision",
    "VerifierConfig",
    "dirichlet_predict",
    "graph_from_parent_sets",
    "hierarchy_matched_cautious",
    "log_linear_posterior",
    "pact_fuse",
    "pact_registered_components",
    "provenance_components",
    "pcecf_fuse",
    "pcecf_registered_components",
    "registered_component_count",
    "restrict_dirichlet_input",
    "typed_admit",
    "verify_source_state",
]
