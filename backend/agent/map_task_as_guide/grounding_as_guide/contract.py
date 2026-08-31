"""Validation for public, inspectable grounding-stage outputs."""
from __future__ import annotations

import json
from typing import Any


VALID_H_CODES = frozenset(f"H{i}" for i in range(1, 8))
VALID_W_CODES = frozenset(f"W{i}" for i in range(1, 14))
VALID_R_CODES = frozenset(f"R{i}" for i in range(1, 8))


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_h(raw: str) -> dict[str, str]:
    value = parse_json_object(raw)
    code = value.get("h_code")
    if not isinstance(code, str) or code not in VALID_H_CODES:
        return {}
    return {
        "h_code": code,
        "confidence": str(value.get("confidence", "low")),
        "evidence": str(value.get("evidence", "")),
    }


def parse_w(raw: str) -> dict[str, Any]:
    """Validate multi-label W diagnosis while retaining only observable evidence."""
    value = parse_json_object(raw)
    raw_codes = value.get("w_codes")
    raw_assessments = value.get("assessments")
    if not isinstance(raw_codes, list) or not isinstance(raw_assessments, list):
        return {}

    codes: list[str] = []
    for code in raw_codes:
        if isinstance(code, str) and code in VALID_W_CODES and code not in codes:
            codes.append(code)

    evidence_by_code: dict[str, str] = {}
    for assessment in raw_assessments:
        if not isinstance(assessment, dict):
            continue
        code = assessment.get("w_code")
        evidence = assessment.get("evidence")
        if isinstance(code, str) and code in codes and isinstance(evidence, str) and evidence.strip():
            evidence_by_code.setdefault(code, evidence.strip())

    codes = [code for code in codes if code in evidence_by_code]
    return {
        "w_codes": tuple(codes),
        "assessments": tuple(
            {"w_code": code, "evidence": evidence_by_code[code]} for code in codes
        ),
    }


def parse_repair(raw: str) -> dict[str, Any]:
    """Validate a codebook R; priority rank deliberately does not constrain it."""
    value = parse_json_object(raw)
    raw_codes = value.get("applied_r_codes")
    if isinstance(raw_codes, list):
        codes = tuple(dict.fromkeys(
            code for code in raw_codes if isinstance(code, str) and code in VALID_R_CODES and code != "R7"
        ))
    else:
        code = value.get("r_code")
        codes = (code,) if isinstance(code, str) and code in VALID_R_CODES else ()
    if not codes:
        return {}
    return {
        "r_code": codes[0],
        "applied_r_codes": codes,
        "deferred_r_codes": tuple(dict.fromkeys(
            code for code in value.get("deferred_r_codes", [])
            if isinstance(code, str) and code in VALID_R_CODES
        )) if isinstance(value.get("deferred_r_codes", []), list) else (),
        "reply": str(value.get("reply", "")),
        "adds_missing_detail": str(value.get("adds_missing_detail", "false")).lower() == "true",
        "selection_evidence": str(value.get("selection_evidence", "")).strip(),
    }


def parse_repair_plan(raw: str) -> dict[str, Any]:
    """Validate an ordered, current-turn repair plan before reply drafting."""
    value = parse_json_object(raw)
    raw_immediate = value.get("immediate_steps")
    raw_deferred = value.get("deferred_steps")
    if not isinstance(raw_immediate, list) or not isinstance(raw_deferred, list):
        return {}

    immediate_steps = []
    used_codes: set[str] = set()
    for step in raw_immediate:
        if not isinstance(step, dict):
            continue
        code = step.get("r_code")
        # R7 verifies uptake after a participant can act; it is never an
        # immediate action in the same message that introduces a correction.
        if not isinstance(code, str) or code not in VALID_R_CODES or code == "R7" or code in used_codes:
            continue
        evidence = str(step.get("current_evidence", "")).strip()
        if not evidence:
            continue
        used_codes.add(code)
        example_ids = step.get("semantic_example_ids")
        immediate_steps.append({
            "order": len(immediate_steps) + 1,
            "r_code": code,
            "current_evidence": evidence,
            "semantic_example_ids": tuple(
                item for item in example_ids if isinstance(item, str) and item.strip()
            ) if isinstance(example_ids, list) else (),
            "semantic_fit": str(step.get("semantic_fit", "")).strip(),
        })

    deferred_steps = []
    for step in raw_deferred:
        if not isinstance(step, dict):
            continue
        code = step.get("r_code")
        trigger = str(step.get("trigger", "")).strip()
        reason = str(step.get("reason", "")).strip()
        if isinstance(code, str) and code in VALID_R_CODES and trigger and reason:
            deferred_steps.append({"r_code": code, "trigger": trigger, "reason": reason})

    if not immediate_steps:
        return {}
    return {
        "immediate_steps": tuple(immediate_steps),
        "deferred_steps": tuple(deferred_steps),
        "plan_rationale": str(value.get("plan_rationale", "")).strip(),
    }
