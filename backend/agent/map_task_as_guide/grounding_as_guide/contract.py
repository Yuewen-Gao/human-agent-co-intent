"""Validation for public, inspectable grounding-stage outputs."""
from __future__ import annotations

import json
from typing import Any


VALID_H_CODES = frozenset(f"H{i}" for i in range(1, 8))
VALID_W_CODES = frozenset(f"W{i}" for i in range(1, 14))
VALID_R_CODES = frozenset(f"R{i}" for i in range(1, 8))
VALID_TRAJECTORY_VERDICTS = frozenset({
    "correct", "incorrect", "insufficient_evidence", "not_applicable",
})


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def parse_h(raw: str) -> dict[str, Any]:
    value = parse_json_object(raw)
    code = value.get("h_code")
    if not isinstance(code, str) or code not in VALID_H_CODES:
        return {}
    raw_misalignment = value.get("misalignment")
    raw_misalignment = raw_misalignment if isinstance(raw_misalignment, dict) else {}
    misalignment = {
        "expected_state": str(raw_misalignment.get("expected_state", "")).strip(),
        "observed_state": str(raw_misalignment.get("observed_state", "")).strip(),
        "what_is_misaligned": str(raw_misalignment.get("what_is_misaligned", "")).strip(),
    }
    # A repair code without a concrete, inspectable mismatch cannot safely
    # drive a participant-facing repair message.
    if code != "H7" and not all(misalignment.values()):
        return {}
    raw_replan = value.get("route_replan")
    raw_replan = raw_replan if isinstance(raw_replan, dict) else {}
    route_replan = {
        "required": raw_replan.get("required") is True,
        "confirmed_prefix": str(raw_replan.get("confirmed_prefix", "")).strip(),
        "guide_route_basis": str(raw_replan.get("guide_route_basis", "")).strip(),
        "follower_local_target": str(raw_replan.get("follower_local_target", "")).strip(),
        "completion_criterion": str(raw_replan.get("completion_criterion", "")).strip(),
    }
    if route_replan["required"] and not all(
        route_replan[key] for key in (
            "confirmed_prefix", "guide_route_basis", "follower_local_target", "completion_criterion",
        )
    ):
        return {}
    trajectory_verdict = value.get("trajectory_verdict")
    if not isinstance(trajectory_verdict, str) or trajectory_verdict not in VALID_TRAJECTORY_VERDICTS:
        return {}
    if code == "H4" and trajectory_verdict != "incorrect":
        return {}
    next_guide_action = "repair" if code != "H7" else "normal_or_wait"
    if code == "H7" and value.get("next_guide_action") == "give_next_segment":
        if trajectory_verdict != "correct":
            return {}
        next_guide_action = "give_next_segment"
    return {
        "h_code": code,
        "confidence": str(value.get("confidence", "low")),
        "evidence": str(value.get("evidence", "")),
        "misalignment": misalignment,
        "route_replan": route_replan,
        "trajectory_verdict": trajectory_verdict,
        "next_guide_action": next_guide_action,
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

    assessment_by_code: dict[str, dict[str, str]] = {}
    for assessment in raw_assessments:
        if not isinstance(assessment, dict):
            continue
        code = assessment.get("w_code")
        evidence = assessment.get("evidence")
        proposition = assessment.get("proposition")
        alignment_needed = assessment.get("alignment_needed")
        if (
            isinstance(code, str)
            and code in codes
            and isinstance(evidence, str) and evidence.strip()
            and isinstance(proposition, str) and proposition.strip()
            and isinstance(alignment_needed, str) and alignment_needed.strip()
        ):
            assessment_by_code.setdefault(code, {
                "w_code": code,
                "evidence": evidence.strip(),
                "proposition": proposition.strip(),
                "alignment_needed": alignment_needed.strip(),
            })

    codes = [code for code in codes if code in assessment_by_code]
    return {
        "w_codes": tuple(codes),
        "assessments": tuple(assessment_by_code[code] for code in codes),
    }


def parse_joint_diagnosis(raw: str) -> dict[str, Any]:
    """Validate the joint H/W response without weakening either sub-contract."""
    value = parse_json_object(raw)
    h_raw = value.get("h")
    w_raw = value.get("w")
    h = parse_h(json.dumps(h_raw)) if isinstance(h_raw, dict) else {}
    w = parse_w(json.dumps(w_raw)) if isinstance(w_raw, dict) else {}
    return {"h": h, "w": w}


def parse_repair(raw: str) -> dict[str, Any]:
    """Validate one combined, inspectable repair-selection and reply result."""
    value = parse_json_object(raw)
    raw_steps = value.get("repair_steps")
    repair_steps = []
    used_codes: set[str] = set()
    if isinstance(raw_steps, list):
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            code = raw_step.get("r_code")
            evidence = str(raw_step.get("observable_evidence", "")).strip()
            if (
                not isinstance(code, str)
                or code not in VALID_R_CODES
                or code == "R7"
                or code in used_codes
                or not evidence
            ):
                continue
            used_codes.add(code)
            example_ids = raw_step.get("example_ids")
            repair_steps.append({
                "r_code": code,
                "observable_evidence": evidence,
                "example_ids": tuple(
                    item for item in example_ids if isinstance(item, str) and item.strip()
                ) if isinstance(example_ids, list) else (),
            })
    codes = tuple(step["r_code"] for step in repair_steps)
    if not codes:
        raw_codes = value.get("applied_r_codes")
        if isinstance(raw_codes, list):
            codes = tuple(dict.fromkeys(
                code for code in raw_codes
                if isinstance(code, str) and code in VALID_R_CODES and code != "R7"
            ))
        else:
            code = value.get("r_code")
            codes = (code,) if isinstance(code, str) and code in VALID_R_CODES else ()
    if not codes:
        return {}
    return {
        "r_code": codes[0],
        "applied_r_codes": codes,
        "repair_steps": tuple(repair_steps),
        "deferred_r_codes": tuple(dict.fromkeys(
            code for code in value.get("deferred_r_codes", [])
            if isinstance(code, str) and code in VALID_R_CODES
        )) if isinstance(value.get("deferred_r_codes", []), list) else (),
        "reply": str(value.get("reply", "")),
        "adds_missing_detail": str(value.get("adds_missing_detail", "false")).lower() == "true",
        "selection_evidence": str(value.get("selection_evidence", "")).strip(),
    }
