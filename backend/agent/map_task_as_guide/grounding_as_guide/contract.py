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
VALID_TRAJECTORY_RELATIONS = frozenset({
    "advances", "completes", "departs", "not_assessable", "not_applicable",
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
    trajectory_verdict = value.get("trajectory_verdict")
    if not isinstance(trajectory_verdict, str) or trajectory_verdict not in VALID_TRAJECTORY_VERDICTS:
        return {}
    raw_assessment = value.get("trajectory_assessment")
    raw_assessment = raw_assessment if isinstance(raw_assessment, dict) else {}
    trajectory_assessment = {
        "fresh_trace_relation": str(raw_assessment.get("fresh_trace_relation", "")).strip(),
        "observable_departure": str(raw_assessment.get("observable_departure", "")).strip(),
    }
    if code == "H4" and (
        trajectory_verdict != "incorrect"
        or trajectory_assessment["fresh_trace_relation"] != "departs"
        or not trajectory_assessment["observable_departure"]
    ):
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
        "trajectory_verdict": trajectory_verdict,
        "trajectory_assessment": trajectory_assessment,
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
        repair_focus = assessment.get("repair_focus")
        if (
            isinstance(code, str)
            and code in codes
            and isinstance(evidence, str) and evidence.strip()
            and isinstance(repair_focus, str) and repair_focus.strip()
        ):
            assessment_by_code.setdefault(code, {
                "w_code": code,
                "evidence": evidence.strip(),
                "repair_focus": repair_focus.strip(),
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


def parse_repair(raw: str, *, allow_r7: bool = False) -> dict[str, Any]:
    """Validate R metadata plus its Map Task action-envelope reply.

    R7 (acceptance check) is accepted only when the caller allows it, i.e. when
    the target item already has a repair to check.
    """
    value = parse_json_object(raw)
    raw_actions = value.get("actions")
    if raw_actions is not None:
        if not isinstance(raw_actions, list):
            return {}
        action_replies = [
            str(action.get("content", "")).strip()
            for action in raw_actions
            if (
                isinstance(action, dict)
                and action.get("type") == "send_map_guidance"
                and str(action.get("content", "")).strip()
            )
        ]
        if len(action_replies) != 1:
            return {}
        reply = action_replies[0]
    else:
        # Keep legacy traces and isolated tests readable while new R prompts
        # use the same action envelope as the main Guide agent.
        reply = str(value.get("reply", "")).strip()
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
                or (code == "R7" and not allow_r7)
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
                if isinstance(code, str) and code in VALID_R_CODES and (code != "R7" or allow_r7)
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
        "reply": reply,
        "adds_missing_detail": str(value.get("adds_missing_detail", "false")).lower() == "true",
        "selection_evidence": str(value.get("selection_evidence", "")).strip(),
    }


VALID_OUTCOME_CODES = frozenset({"O1", "O2", "O3", "O4"})
VALID_CLOSURE_EVIDENCE = frozenset({
    "explicit_acknowledgement", "accurate_restatement", "consistent_task_action", "none_observed",
})
VALID_DETECTION_ACTIONS = frozenset({"merge", "new", "none"})


def parse_reconciliation(raw: str, open_item_ids: frozenset[str] | set[str]) -> dict[str, Any]:
    """Validate the item-maintenance response against the currently open items.

    Unknown ids, invalid codes, and missing entries are dropped item by item;
    an omitted item simply receives no outcome this turn.
    """
    value = parse_json_object(raw)
    raw_items = value.get("items")
    outcomes: dict[str, dict[str, str]] = {}
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item_id = raw_item.get("id")
            outcome = raw_item.get("outcome")
            closure = raw_item.get("closure_evidence")
            if (
                isinstance(item_id, str) and item_id in open_item_ids and item_id not in outcomes
                and outcome in VALID_OUTCOME_CODES
                and closure in VALID_CLOSURE_EVIDENCE
            ):
                outcomes[item_id] = {
                    "outcome": outcome,
                    "closure_evidence": closure,
                    "evidence": str(raw_item.get("evidence", "")).strip(),
                }
    raw_detection = value.get("detection")
    raw_detection = raw_detection if isinstance(raw_detection, dict) else {}
    action = raw_detection.get("action")
    item_id = str(raw_detection.get("item_id") or "").strip()
    proposition = str(raw_detection.get("proposition") or "").strip()
    if action not in VALID_DETECTION_ACTIONS or (action == "merge" and item_id not in open_item_ids):
        detection = {}
    else:
        detection = {
            "action": action,
            "item_id": item_id if action == "merge" else "",
            "proposition": proposition if action == "new" else "",
        }
    return {"outcomes": outcomes, "detection": detection}


VALID_PROGRESS = frozenset({"in_progress", "no_evidence", "not_applicable"})
VALID_STATES = frozenset({"no_issue", "specific_correctable", "missing_information", "instruction_invalidated"})


def parse_decision(raw: str, open_item_ids: frozenset[str] | set[str], acts_vocabulary: tuple[str, ...]) -> dict[str, Any]:
    """Validate the combined item-upkeep and act-selection response.

    Item outcomes and merge/new reuse `parse_reconciliation`. The act part is
    exclusive: either no_message (no acts) or at least one known act.
    """
    value = parse_json_object(raw)
    upkeep = parse_reconciliation(raw, open_item_ids)
    for entry in value.get("items") or []:
        if isinstance(entry, dict) and entry.get("id") in upkeep["outcomes"]:
            progress = entry.get("progress")
            upkeep["outcomes"][entry["id"]]["progress"] = progress if progress in VALID_PROGRESS else ""
    raw_acts = [a for a in value.get("acts") or [] if isinstance(a, dict) and a.get("act") in acts_vocabulary]
    acts = [a["act"] for a in raw_acts]
    targets = [str(a.get("target_item_id") or "").strip() for a in raw_acts]
    targets = [t if t in open_item_ids else "" for t in targets]
    no_message = value.get("no_message") is True
    primary = value.get("primary_act") if value.get("primary_act") in acts else (acts[0] if acts else None)
    reason = str(value.get("reason") or "").strip() or "; ".join(
        str(a.get("evidence", "")).strip() for a in raw_acts if str(a.get("evidence", "")).strip())
    valid = (no_message and not acts) or (not no_message and bool(acts))
    return {
        **upkeep,
        "valid": valid,
        "no_message": no_message and not acts,
        "acts": acts if valid and not no_message else [],
        "targets": targets if valid and not no_message else [],
        "primary_act": primary if valid and not no_message else None,
        "state": value.get("state") if value.get("state") in VALID_STATES else None,
        "reason": reason,
    }
