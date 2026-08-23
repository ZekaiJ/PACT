"""Observable pre-action eligibility verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .contracts import CONTRACT_CLASSES, OBSERVABLE_SOURCES, top_label
from .lineage import registered_component_count


@dataclass(frozen=True)
class VerifierConfig:
    missing_risk_language_confidence: float = 0.90
    missing_risk_language_quality: float = 0.90
    missing_risk_language_margin: float = 0.70
    consensus_confidence: float = 0.40
    consensus_quality: float = 0.30
    consensus_conflict: float = 0.15
    minimum_available_sources: int = 1
    minimum_registered_components: int = 3
    provenance_policy: str = "unanimous_three_source"


@dataclass(frozen=True)
class VerificationDecision:
    admissible: bool
    route: str
    reason: str
    diagnostics: Mapping[str, Any]


DEFAULT_CONFIG = VerifierConfig()


def _source_payload(record: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    sources = record.get("sources", {})
    payload = sources.get(source, {}) if isinstance(sources, Mapping) else {}
    return payload if isinstance(payload, Mapping) else {}


def _source_missing(record: Mapping[str, Any], source: str) -> bool:
    payload = _source_payload(record, source)
    return not payload or bool(payload.get("missing", False))


def _schema_issues(record: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    sources = record.get("sources", {})
    if not isinstance(sources, Mapping):
        return ("sources",)
    for name, payload in sources.items():
        if name not in OBSERVABLE_SOURCES or not isinstance(payload, Mapping):
            issues.append(str(name))
            continue
        if bool(payload.get("missing", False)):
            continue
        if payload.get("schema_valid", True) is False:
            issues.append(f"{name}.schema_valid")
        for field in ("probabilities", "quality", "conflict"):
            if field not in payload:
                issues.append(f"{name}.{field}")
        probabilities = payload.get("probabilities")
        if (
            not isinstance(probabilities, Mapping)
            or set(probabilities) != set(CONTRACT_CLASSES)
        ):
            issues.append(f"{name}.probabilities")
        else:
            try:
                values = tuple(float(probabilities[label]) for label in CONTRACT_CLASSES)
            except (TypeError, ValueError):
                issues.append(f"{name}.probabilities")
            else:
                if (
                    any(not math.isfinite(value) or value < 0.0 for value in values)
                    or not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-8)
                ):
                    issues.append(f"{name}.probabilities")
    language = _source_payload(record, "language")
    if language and not bool(language.get("missing", False)):
        route = language.get("source_route")
        if not isinstance(route, Mapping):
            issues.append("language.source_route")
        else:
            for field in ("current_command_hash", "payload_text_hash", "source_status"):
                if not route.get(field):
                    issues.append(f"language.source_route.{field}")
    return tuple(sorted(set(issues)))


def _command_currentness_issue(record: Mapping[str, Any]) -> str:
    language = _source_payload(record, "language")
    if not language or bool(language.get("missing", False)):
        return ""
    route = language.get("source_route", {})
    if not isinstance(route, Mapping):
        return "missing_route"
    current_hash = str(route.get("current_command_hash", ""))
    payload_hash = str(route.get("payload_text_hash", ""))
    if current_hash and payload_hash and current_hash != payload_hash:
        return "payload_hash_mismatch"
    if (
        route.get("source_status") == "stale_previous_command"
        and route.get("previous_command_hash")
        and route.get("carryover_from_record_id")
    ):
        return "stale_previous_command"
    return ""


def verify_source_state(
    record: Mapping[str, Any],
    predicted_contract: str,
    lineage_graph: Mapping[tuple[str, str], float],
    *,
    lineage_complete: bool = True,
    config: VerifierConfig = DEFAULT_CONFIG,
) -> VerificationDecision:
    """Evaluate observable eligibility without reading evaluation outcomes."""

    currentness_issue = _command_currentness_issue(record)
    available = [
        source for source in OBSERVABLE_SOURCES if not _source_missing(record, source)
    ]
    schema_issues = _schema_issues(record)
    if len(available) < config.minimum_available_sources:
        schema_issues = tuple(sorted((*schema_issues, "available_source_count")))

    tops = {
        source: top_label(_source_payload(record, source)["probabilities"])
        for source in available
    }
    supporters = [
        source
        for source, (label, _, _) in tops.items()
        if label == predicted_contract
    ]
    component_count = registered_component_count(
        supporters,
        lineage_graph,
        partition_nodes=OBSERVABLE_SOURCES,
    )

    language = _source_payload(record, "language")
    _, language_confidence, language_margin = top_label(
        language.get("probabilities", {})
    )
    language_quality = float(language.get("quality", 0.0))
    missing_risk_overconfidence = (
        _source_missing(record, "risk")
        and not _source_missing(record, "language")
        and language_confidence >= config.missing_risk_language_confidence
        and language_quality >= config.missing_risk_language_quality
        and language_margin >= config.missing_risk_language_margin
    )

    unanimous = len(available) >= 3 and len(supporters) == len(available)
    corroborative_candidate = len(supporters) >= 2
    support_confidence = min(
        (tops[source][1] for source in supporters),
        default=0.0,
    )
    support_quality = min(
        (
            float(_source_payload(record, source).get("quality", 0.0))
            for source in supporters
        ),
        default=0.0,
    )
    support_conflict = max(
        (
            float(_source_payload(record, source).get("conflict", 0.0))
            for source in supporters
        ),
        default=0.0,
    )
    support_is_eligible_for_corroboration = (
        corroborative_candidate
        and support_confidence >= config.consensus_confidence
        and support_quality >= config.consensus_quality
        and support_conflict <= config.consensus_conflict
    )
    required_roles_satisfied = (
        "language" in supporters
        and any(source in supporters for source in ("geometry", "risk"))
    )
    if config.provenance_policy == "unanimous_three_source":
        provenance_predicate_applies = unanimous
        required_components = config.minimum_registered_components
        required_roles_apply = False
    elif config.provenance_policy == "multi_source_two_component":
        provenance_predicate_applies = corroborative_candidate
        required_components = config.minimum_registered_components
        required_roles_apply = False
    elif config.provenance_policy == "required_role_two_component":
        provenance_predicate_applies = corroborative_candidate
        required_components = config.minimum_registered_components
        required_roles_apply = True
    else:
        raise ValueError(f"unknown provenance policy: {config.provenance_policy}")
    insufficient_registered_corroboration = (
        provenance_predicate_applies
        and support_is_eligible_for_corroboration
        and (
            not lineage_complete
            or component_count < required_components
            or (required_roles_apply and not required_roles_satisfied)
        )
    )

    diagnostics = {
        "available_sources": len(available),
        "minimum_available_sources": config.minimum_available_sources,
        "supporting_sources": len(supporters),
        "registered_components": component_count,
        "minimum_registered_components": required_components,
        "provenance_policy": config.provenance_policy,
        "provenance_predicate_applies": provenance_predicate_applies,
        "corroborative_candidate": corroborative_candidate,
        "required_roles_satisfied": required_roles_satisfied,
        "lineage_complete": lineage_complete,
        "schema_issues": schema_issues,
        "currentness_issue": currentness_issue,
        "missing_risk_overconfidence": missing_risk_overconfidence,
        "insufficient_registered_corroboration": (
            insufficient_registered_corroboration
        ),
    }
    if currentness_issue:
        return VerificationDecision(
            False,
            "hold",
            "current_command_mismatch",
            diagnostics,
        )
    if schema_issues:
        return VerificationDecision(
            False,
            "fallback",
            "incomplete_source_schema",
            diagnostics,
        )
    if missing_risk_overconfidence:
        return VerificationDecision(
            False,
            "hold",
            "missing_risk_support",
            diagnostics,
        )
    if insufficient_registered_corroboration:
        return VerificationDecision(
            False,
            "confirm",
            "insufficient_registered_corroboration",
            diagnostics,
        )
    return VerificationDecision(
        True,
        "none",
        "evidence_eligible",
        diagnostics,
    )

