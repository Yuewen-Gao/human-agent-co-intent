"""Parallel H/W diagnosis, one item-upkeep and act-selection decision, then the reply."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import grounding_acts
from .codebook import definitions_for, definitions_for_codes
from .contract import parse_decision, parse_h, parse_json_object, parse_repair, parse_w
from .fewshot_repository import (
    format_h_fewshots,
    format_repair_examples,
    retrieve_repair_examples,
)
from .grounding_rules import (
    associated_repair_codes,
    update_mode,
)
from services.llm_trace_logger import TraceContext, record_trace_result, trace_chat_completion
from agent.map_task_as_guide.agent_session_context import _trajectory_cells, follower_trajectory_grid_text


_PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts"
_CURRENT_GAME_STATE_MARKER = "\n\n<CURRENT GAME STATE>\n"

# GPT-5.6 Luna may use completion tokens for reasoning before emitting the
# required JSON. Leave enough budget for both the diagnosis and its payload.
JOINT_DIAGNOSIS_MAX_TOKENS = 4096
REPAIR_MAX_TOKENS = 2048

_MISALIGNMENT_HISTORY_KEY = "grounding_misalignment_history"
_MISALIGNMENT_STATUSES = frozenset({"Resolved", "Unresolved"})
_DETECTED_MISALIGNMENT_CODES = frozenset(f"H{i}" for i in range(1, 7))
# Resolved means the episode is closed; the O code records how (O4 keeps it open).
_CLOSING_OUTCOMES = frozenset({"O1", "O2", "O3"})


def _joint_diagnosis_max_tokens(llm_client: Any) -> int:
    """Use a validated DeepSeek-specific H/W budget without changing other providers."""
    override = getattr(llm_client, "joint_diagnosis_max_tokens", None)
    if isinstance(override, int) and override > 0:
        return override
    return JOINT_DIAGNOSIS_MAX_TOKENS


# The Follower's historical execute-or-ask contract remains a compatibility
# boundary, but its validation belongs to orchestration rather than a separate
# "clarification gate" policy module. It validates only the public payload;
# semantic ambiguity and route geometry are decided elsewhere.
_VALID_FOLLOWER_DECISIONS = frozenset({"execute", "ask_clarification"})


def _validated_follower_path_points(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list) or not value:
        return None
    points: list[list[int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            row, col = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return None
        points.append([row, col])
    return points


def _decision_text(value: dict[str, Any], field: str) -> str:
    candidate = value.get(field)
    return candidate.strip() if isinstance(candidate, str) else ""


def parse_follower_action_decision(raw: str) -> dict[str, Any]:
    """Validate the legacy Follower public action envelope.

    This is deliberately not a clarification policy: the model chooses its
    action, while the orchestrator admits only one well-formed public action.
    Route geometry and latest-message checks remain server-side at execution.
    """
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}

    decision = value.get("decision")
    if decision not in _VALID_FOLLOWER_DECISIONS:
        return {}

    if decision == "execute":
        path_points = _validated_follower_path_points(value.get("path_points"))
        if not path_points:
            return {}
        return {
            "decision": "execute",
            "source_message_id": _decision_text(value, "source_message_id"),
            "segment": _decision_text(value, "segment"),
            "completion_criterion": _decision_text(value, "completion_criterion"),
            "message": _decision_text(value, "message"),
            "path_points": path_points,
            "question": "",
        }

    question = _decision_text(value, "question")
    if not question:
        return {}
    return {
        "decision": "ask_clarification",
        "source_message_id": "",
        "segment": "",
        "completion_criterion": "",
        "message": "",
        "path_points": [],
        "question": question,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _normalized_misalignment_item(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    item_id = value.get("id")
    description = value.get("description")
    status = value.get("status")
    if (
        not isinstance(item_id, str) or not item_id.strip()
        or not isinstance(description, str) or not description.strip()
        or status not in _MISALIGNMENT_STATUSES
    ):
        return None
    outcome = value.get("outcome")
    return {
        "id": item_id.strip(),
        "description": description.strip(),
        "status": status,
        "outcome": outcome if outcome in _CLOSING_OUTCOMES else None,
        "h_codes": _string_list(value.get("h_codes")),
        "w_codes": _string_list(value.get("w_codes")),
        "repair_attempts": _dict_list(value.get("repair_attempts")),
        "evidence_log": _dict_list(value.get("evidence_log")),
    }


def load_misalignment_history(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Return well-formed history items without exposing arbitrary session data."""
    raw_history = session.get(_MISALIGNMENT_HISTORY_KEY)
    if not isinstance(raw_history, list):
        return []
    return [item for raw_item in raw_history if (item := _normalized_misalignment_item(raw_item))]


def unresolved_misalignment_queue(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the open episodes that the next turn must reconcile."""
    return [
        item for item in load_misalignment_history(session)
        if item["status"] == "Unresolved"
    ]


def _next_misalignment_id(history: list[dict[str, Any]]) -> str:
    existing_numbers = [
        int(item["id"][2:])
        for item in history
        if item["id"].startswith("m-") and item["id"][2:].isdigit()
    ]
    return f"m-{max(existing_numbers, default=0) + 1:03d}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_item(session: dict[str, Any], item_id: str, update: Any) -> dict[str, Any] | None:
    """Apply ``update(item)`` to one stored item and persist the whole history."""
    history = load_misalignment_history(session)
    for item in history:
        if item["id"] == item_id:
            update(item)
            session[_MISALIGNMENT_HISTORY_KEY] = history
            return item
    return None


def _detection_codes(h: dict[str, Any], w: dict[str, Any]) -> tuple[list[str], list[str]]:
    h_codes = [h["h_code"]] if h.get("h_code") in _DETECTED_MISALIGNMENT_CODES else []
    return h_codes, [code for code in (w.get("w_codes") or ()) if isinstance(code, str)]


def open_misalignment_item(
    session: dict[str, Any], h: dict[str, Any], w: dict[str, Any], proposition: str = ""
) -> dict[str, Any] | None:
    """Open a new Unresolved episode for one detected H1-H6 cue."""
    if h.get("h_code") not in _DETECTED_MISALIGNMENT_CODES:
        return None
    history = load_misalignment_history(session)
    evidence = str(h.get("evidence") or "").strip()
    h_codes, w_codes = _detection_codes(h, w)
    item = {
        "id": _next_misalignment_id(history),
        "description": proposition.strip() or f"{h['h_code']}: {evidence or 'Observable misalignment detected.'}",
        "status": "Unresolved",
        "outcome": None,
        "h_codes": h_codes,
        "w_codes": w_codes,
        "repair_attempts": [],
        "evidence_log": [{"timestamp": _now(), "kind": "detected", "h_code": h["h_code"], "evidence": evidence}],
    }
    session[_MISALIGNMENT_HISTORY_KEY] = [*history, item]
    return item


def merge_misalignment_detection(
    session: dict[str, Any], item_id: str, h: dict[str, Any], w: dict[str, Any]
) -> dict[str, Any] | None:
    """Attach a repeated detection of the same proposition to its open episode."""
    h_codes, w_codes = _detection_codes(h, w)

    def update(item: dict[str, Any]) -> None:
        item["h_codes"] = list(dict.fromkeys([*item["h_codes"], *h_codes]))
        item["w_codes"] = list(dict.fromkeys([*item["w_codes"], *w_codes]))
        item["evidence_log"].append({
            "timestamp": _now(), "kind": "merged",
            "h_code": h.get("h_code"), "evidence": str(h.get("evidence") or "").strip(),
        })

    return _update_item(session, item_id, update)


def close_misalignment_item(
    session: dict[str, Any], item_id: str, outcome: str, closure_evidence: str = "", evidence: str = ""
) -> bool:
    """Close one episode as Resolved; the O code records how it ended."""
    if outcome not in _CLOSING_OUTCOMES:
        return False

    def update(item: dict[str, Any]) -> None:
        item["status"] = "Resolved"
        item["outcome"] = outcome
        item["evidence_log"].append({
            "timestamp": _now(), "kind": "outcome", "outcome": outcome,
            "closure_evidence": closure_evidence, "evidence": evidence,
        })

    target = next((item for item in unresolved_misalignment_queue(session) if item["id"] == item_id), None)
    return bool(target and _update_item(session, item_id, update))


def note_misalignment_evidence(session: dict[str, Any], item_id: str, outcome: dict[str, str]) -> None:
    """Log a not-yet-closing (O4) assessment without changing status."""
    _update_item(session, item_id, lambda item: item["evidence_log"].append({
        "timestamp": _now(), "kind": "outcome", **outcome,
    }))


def record_repair_attempt(
    session: dict[str, Any],
    item_id: str | None,
    r_codes: Any,
    reply: str,
    trajectory_grid: str | None = None,
) -> bool:
    """Remember a participant-visible repair and the Follower trace at that moment.

    The trace snapshot lets reconciliation see everything drawn since this
    repair, not only since the most recent Guide message.
    """
    if not item_id or not str(reply or "").strip():
        return False
    grid = follower_trajectory_grid_text(session) if trajectory_grid is None else trajectory_grid
    attempt = {"timestamp": _now(), "r_codes": list(r_codes or ()), "reply": str(reply).strip()}
    if _trajectory_cells(grid) is not None:
        attempt["trajectory_grid_at_repair"] = grid.strip()
    return _update_item(session, item_id, lambda item: item["repair_attempts"].append(attempt)) is not None


def _trajectory_since_last_repair(item: dict[str, Any], current_grid: str | None) -> str:
    attempts = item["repair_attempts"]
    if not attempts:
        return "no repair sent yet"
    before = _trajectory_cells(attempts[-1].get("trajectory_grid_at_repair") or "")
    after = _trajectory_cells(current_grid or "")
    if before is None or after is None:
        return "not available"
    added = [list(cell) for cell in sorted(after - before)]
    removed = [list(cell) for cell in sorted(before - after)]
    return f"added_cells: {json.dumps(added)}; removed_cells: {json.dumps(removed)}"


def _format_open_items(items: list[dict[str, Any]], current_grid: str | None = None) -> str:
    lines = []
    for item in items:
        attempts = item["repair_attempts"]
        last_repair = f'last repair: "{attempts[-1]["reply"]}"' if attempts else "no repair sent yet"
        lines.append(
            f"- {item['id']} | {item['description']} | H: {', '.join(item['h_codes']) or '-'}"
            f" | W: {', '.join(item['w_codes']) or '-'} | repairs sent: {len(attempts)} ({last_repair})"
            f"\n  Follower trajectory changes since last repair: {_trajectory_since_last_repair(item, current_grid)}"
            + (f"\n  Recent judgements: {_recent_judgements(item)}" if item["evidence_log"] else "")
        )
    return "\n".join(lines)


def _recent_judgements(item: dict[str, Any], limit: int = 4) -> str:
    """Earlier detections, outcomes, and acts for this item, newest last."""
    parts = []
    for entry in item["evidence_log"][-limit:]:
        kind = entry.get("kind")
        if kind in ("detected", "merged"):
            parts.append(f"{kind} {entry.get('h_code')}: {str(entry.get('evidence', ''))[:160]}")
        elif kind == "outcome":
            parts.append(f"outcome {entry.get('outcome')} {entry.get('progress', '')}".rstrip()
                         + f": {str(entry.get('evidence', ''))[:160]}")
        elif kind == "act":
            parts.append(f"act {'+'.join(entry.get('acts') or [])}")
    return " | ".join(parts)


def record_selected_acts(session: dict[str, Any], item_ids: Any, acts: list[str]) -> None:
    """Remember which act(s) this turn chose for each targeted item."""
    for item_id in dict.fromkeys(i for i in item_ids if i):
        _update_item(session, item_id, lambda item: item["evidence_log"].append({
            "timestamp": _now(), "kind": "act", "acts": list(acts),
        }))


_H_RESPONSE_SCHEMA = (
    '{"h":{"h_code":"H1..H7","confidence":"low|medium|high",'
    '"evidence":"short observable detection cue",'
    '"trajectory_verdict":"correct|incorrect|insufficient_evidence|not_applicable",'
    '"trajectory_assessment":{"fresh_trace_relation":"advances|completes|departs|not_assessable|not_applicable",'
    '"observable_departure":"specific landmark-relative departure or empty"},'
    '"next_guide_action":"repair|give_next_segment|normal_or_wait"}}'
)
_W_RESPONSE_SCHEMA = (
    '{"w":{"w_codes":["W1..W13"],"assessments":[{"w_code":"W1..W13",'
    '"evidence":"short observable support",'
    '"repair_focus":"neutral sentence describing what the reply should clarify"}]}}'
)


def _h_diagnosis_prompt(evidence: str) -> str:
    """H detection only; W is diagnosed by an independent, parallel request."""
    h_instruction = (_PROMPTS / "grounding_detection_prompt.txt").read_text(encoding="utf-8")
    return (
        "<H DETECTION INSTRUCTIONS>\n"
        + h_instruction
        + "\n</H DETECTION INSTRUCTIONS>\n\n<H CODEBOOK>\n"
        + definitions_for("H")
        + "\n</H CODEBOOK>\n\n<H FEW-SHOT REFERENCE ONLY>\n"
        + "Examples are reference-only labeled patterns, not part of the current episode. "
        + "Do not treat an example utterance, landmark, route, or expected output as current evidence. "
        + "Do not copy example content into H or evidence fields. "
        + "Classify only facts inside the later <CURRENT OBSERVABLE EVIDENCE> block.\n"
        + format_h_fewshots()
        + "\n</H FEW-SHOT REFERENCE ONLY>\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n<H RESPONSE CONTRACT>\n"
        + "Apply the H instructions and codebook to the current evidence. Return one JSON object only:\n"
        + _H_RESPONSE_SCHEMA
        + "\n</H RESPONSE CONTRACT>"
    )


def _w_diagnosis_prompt(evidence: str) -> str:
    """W diagnosis without H: in evaluation, W-only outperformed joint and H-conditioned W."""
    w_instruction = (_PROMPTS / "grounding_object_prompt.txt").read_text(encoding="utf-8")
    return (
        "<W DETECTION INSTRUCTIONS>\n"
        + w_instruction
        + "\n</W DETECTION INSTRUCTIONS>\n\n<W CODEBOOK>\n"
        + definitions_for("W")
        + "\n</W CODEBOOK>\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n<W RESPONSE CONTRACT>\n"
        + "Apply the W instructions and codebook to the current evidence. Return one JSON object only:\n"
        + _W_RESPONSE_SCHEMA
        + "\n</W RESPONSE CONTRACT>"
    )


def _nested(raw: str, key: str) -> str:
    """Accept both the contracted {"h": {...}} envelope and a bare object."""
    value = parse_json_object(raw)
    return json.dumps(value[key] if isinstance(value.get(key), dict) else value)


def _call_json(llm_client: Any, prompt: str, max_tokens: int, *, trace_context: TraceContext | None = None, stage: str = "") -> tuple[str, Any | None]:
    messages = [{"role": "user", "content": prompt}]
    if trace_context is None:
        return llm_client.chat_completions_create(messages=messages, temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"}), None
    completion = trace_chat_completion(llm_client, replace(trace_context, stage=stage), messages=messages, temperature=0, max_tokens=max_tokens, response_format={"type": "json_object"})
    return completion.response, completion


def _format_w_repair_context(w: dict[str, Any]) -> str:
    """Give R selected W definitions and neutral repair focus, never route facts."""
    codes = tuple(w.get("w_codes") or ())
    definitions = definitions_for_codes("W", codes)
    assessments = w.get("assessments") or ()
    return (
        "<W REPAIR CONTEXT>\n"
        + (definitions or "- no content-level W diagnosis\n")
        + "\n<CURRENT W ASSESSMENTS>\n"
        + json.dumps(assessments, ensure_ascii=False)
        + "\n</CURRENT W ASSESSMENTS>\n"
        + "</W REPAIR CONTEXT>"
    )


def _split_guide_background(evidence: str) -> tuple[str, str]:
    """Separate the rendered static Guide prompt from its current observations."""
    background, marker, observations = evidence.partition(_CURRENT_GAME_STATE_MARKER)
    if not marker:
        return evidence, ""
    return background, marker + observations


def _repair_prompt(
    evidence: str,
    h: dict[str, Any],
    w: dict[str, Any],
    retrieved: tuple[dict[str, Any], ...],
) -> str:
    """Build the R request with fixed Guide rules before repair-specific context."""
    guide_background, observable_evidence = _split_guide_background(evidence)
    selected_h_definition = definitions_for_codes("H", (str(h.get("h_code") or ""),))
    repair_instruction = (_PROMPTS / "grounding_repair_prompt.txt").read_text(encoding="utf-8")
    return (
        "<MAP TASK GUIDE BACKGROUND>\n"
        + guide_background
        + "\n</MAP TASK GUIDE BACKGROUND>\n\n<REPAIR INSTRUCTIONS>\n"
        + repair_instruction
        + "\n</REPAIR INSTRUCTIONS>\n\n<SELECTED H DEFINITION>\n"
        + (selected_h_definition or "- no selected H definition\n")
        + "\n</SELECTED H DEFINITION>\n<R CODEBOOK>\n"
        + definitions_for("R")
        + "\n</R CODEBOOK>\n<DETECTED H>\n"
        + json.dumps(h, ensure_ascii=False)
        + "\n</DETECTED H>\n"
        + _format_w_repair_context(w)
        + "\n"
        + format_repair_examples(retrieved)
        + "\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + observable_evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>"
    )


def _decide_prompt(
    evidence: str,
    open_items: list[dict[str, Any]],
    h: dict[str, Any],
    w: dict[str, Any],
    current_trajectory_grid: str | None = None,
) -> str:
    """Build the combined item-upkeep and act-selection request.

    It receives this turn's H/W results and each item's recent judgements so
    that earlier judgements are reused rather than re-derived.
    """
    _, observable_evidence = _split_guide_background(evidence)
    instruction = (_PROMPTS / "grounding_decide_prompt.txt").read_text(encoding="utf-8")
    detection = {
        "h": {key: h.get(key) for key in ("h_code", "evidence", "trajectory_verdict", "trajectory_assessment")},
        "w": [{key: a.get(key) for key in ("w_code", "evidence", "repair_focus")} for a in (w.get("assessments") or ())],
    }
    return (
        "<DECISION INSTRUCTIONS>\n"
        + instruction
        + "\n</DECISION INSTRUCTIONS>\n\n<GROUNDING ACT DEFINITIONS>\n"
        + grounding_acts.definitions_text()
        + "\n</GROUNDING ACT DEFINITIONS>\n\n<O CODEBOOK>\n"
        + definitions_for("O")
        + "\n</O CODEBOOK>\n\n<UNRESOLVED MISALIGNMENT ITEMS>\n"
        + (_format_open_items(open_items, current_trajectory_grid) if open_items else "- none")
        + "\n</UNRESOLVED MISALIGNMENT ITEMS>\n\n<CURRENT DETECTION>\n"
        + json.dumps(detection, ensure_ascii=False)
        + "\n</CURRENT DETECTION>\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + (observable_evidence or evidence)
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n<RESPONSE CONTRACT>\n"
        + _DECISION_CONTRACT
        + "\n</RESPONSE CONTRACT>"
    )


_DECISION_CONTRACT = (
    'Return one JSON object only:\n'
    '{"items":[{"id":"open item id","outcome":"O1|O2|O3|O4","progress":"in_progress|no_evidence|not_applicable",'
    '"closure_evidence":"explicit_acknowledgement|accurate_restatement|consistent_task_action|none_observed",'
    '"evidence":"short observable reason"}],'
    '"detection":{"action":"merge|new|none","item_id":"open item id for merge, otherwise empty",'
    '"proposition":"short disputed proposition for new, otherwise empty"},'
    '"state":"no_issue|specific_correctable|missing_information|instruction_invalidated",'
    '"no_message":false,"reason":"short reason",'
    '"acts":[{"act":"Initiate|Continue|Acknowledge|Repair|Request Repair|Request Acknowledge|Cancel",'
    '"target_item_id":"open item id or empty","evidence":"short observable reason"}],"primary_act":"one of the acts"}\n'
    'When no_message is true, acts must be empty. Return exactly one entry per open item.'
)


def safe_decide(
    orchestrator: Any,
    evidence: str,
    *,
    session: dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
) -> dict[str, Any] | None:
    """Return no grounding decision when diagnosis cannot complete.

    The caller can then use the normal Guide reply path instead of dropping a
    participant-visible turn because a treatment-only LLM call failed.
    """
    try:
        return orchestrator.decide(evidence, session=session, trace_context=trace_context)
    except Exception:
        return None


class GroundingOrchestrator:
    def __init__(self, llm_client: Any):
        self._llm = llm_client

    def decide(
        self,
        evidence: str,
        *,
        session: dict[str, Any] | None = None,
        trace_context: TraceContext | None = None,
    ) -> dict[str, Any]:
        """Diagnose H/W, then one decision call for item upkeep and act selection, then the reply.

        Each judgement is made once and reused: H owns the trajectory judgement,
        the decision call owns item outcomes and acts, and the reply stage only
        words the selected act.
        """
        h, w = self._diagnose_h_and_w(evidence, trace_context)
        session = session if session is not None else {}
        decision = self._decide(session, evidence, h, w, trace_context)
        w_codes = w.get("w_codes", ())
        item_fields = {
            "misalignment_item_id": decision["item_id"],
            "decision": {key: decision[key] for key in
                         ("outcomes", "detection", "acts", "targets", "no_message", "state", "reason", "valid")
                         if key in decision},
        }
        if decision.get("error"):
            item_fields["decision"]["error"] = decision["error"]

        if not decision["valid"]:
            # Fallback when the decision call fails: the earlier H-gated routing.
            if not h or h.get("h_code") == "H7":
                return {**self._no_repair(h, w, w_codes), **item_fields}
            decision.update(acts=["Repair"], targets=[decision["item_id"] or ""], reason=str(h.get("evidence") or ""))
        elif decision["no_message"]:
            return {**self._no_repair(h, w, w_codes), **item_fields,
                    "needs_base_reply": False, "suppress_base_reply": True}

        queue = {item["id"]: item for item in unresolved_misalignment_queue(session)}
        target_id = next((t for t in decision["targets"] if t), decision["item_id"])
        target = queue.get(target_id) if target_id else None
        status = decision.get("outcomes", {}).get(target_id) if target_id else None
        record_selected_acts(session, [t for t in decision["targets"] if t] or [target_id], decision["acts"])
        for_repair = bool(set(decision["acts"]) & grounding_acts.REPAIR_STAGE_ACTS)
        targets = decision["targets"] or [""] * len(decision["acts"])
        block = grounding_acts.act_block(decision["acts"], targets, decision["reason"], target, status,
                                         for_repair=for_repair)
        if not for_repair:
            return {**self._no_repair(h, w, w_codes), **item_fields, "selected_acts": decision["acts"],
                    "act_block": block}

        h_for_r = h if h.get("h_code") in _DETECTED_MISALIGNMENT_CODES else {
            "h_code": (target or {}).get("h_codes", ["H2"])[-1] if target and target.get("h_codes") else "H2",
            "evidence": (target or {}).get("description", ""),
            "trajectory_verdict": "not_applicable",
            "trajectory_assessment": {"fresh_trace_relation": "not_applicable", "observable_departure": ""},
        }
        retrieved = retrieve_repair_examples(
            evidence,
            h_for_r["h_code"],
            w_codes,
            associated_repair_codes(h_for_r["h_code"], w_codes),
        )
        repair_prompt = _repair_prompt(evidence, h_for_r, w, retrieved).replace(
            "\n<CURRENT OBSERVABLE EVIDENCE>\n", "\n" + block + "\n\n<CURRENT OBSERVABLE EVIDENCE>\n", 1)
        repair_raw, repair_trace = _call_json(
            self._llm,
            repair_prompt,
            REPAIR_MAX_TOKENS,
            trace_context=trace_context,
            stage="grounding.repair",
        )
        repair = parse_repair(repair_raw, allow_r7=grounding_acts.r7_allowed(target))
        if not repair:
            if repair_trace:
                record_trace_result(repair_trace, repair, error="invalid_or_empty_repair")
            return {**self._no_repair(h, w, w_codes), **item_fields}
        r_codes = repair["applied_r_codes"]
        result = {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "selected_acts": decision["acts"],
            "repair_steps": repair["repair_steps"],
            "r": r_codes[0],
            "r_codes": r_codes,
            "deferred_r_codes": repair["deferred_r_codes"],
            "m": update_mode(r_codes[0], adds_missing_detail=repair["adds_missing_detail"]),
            "m_codes": tuple(
                update_mode(r_code, adds_missing_detail=repair["adds_missing_detail"])
                for r_code in r_codes
            ),
            "reply": repair["reply"],
            "selection_evidence": repair["selection_evidence"],
            "needs_base_reply": False,
            "suppress_base_reply": False,
            **item_fields,
            # The repair is recorded against the item the act targeted.
            "misalignment_item_id": target_id or decision["item_id"],
        }
        if repair_trace:
            result["_trace_repair_completion"] = repair_trace
            if not str(repair.get("reply") or "").strip():
                record_trace_result(repair_trace, repair)
        return result

    def _diagnose_h_and_w(
        self, evidence: str, trace_context: TraceContext | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run H and W as independent requests in parallel.

        An H failure propagates (``safe_decide`` then falls back to the normal
        Guide path). A W failure only drops W, which enriches repair but never
        gates it.
        """
        budget = _joint_diagnosis_max_tokens(self._llm)

        def run(prompt: str, stage: str, key: str, parser: Any) -> dict[str, Any]:
            raw, trace = _call_json(self._llm, prompt, budget, trace_context=trace_context, stage=stage)
            parsed = parser(_nested(raw, key))
            if trace:
                record_trace_result(trace, parsed)
            return parsed

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="grounding-diagnosis") as pool:
            h_future = pool.submit(run, _h_diagnosis_prompt(evidence), "grounding.h", "h", parse_h)
            w_future = pool.submit(run, _w_diagnosis_prompt(evidence), "grounding.w", "w", parse_w)
            h = h_future.result()
            try:
                w = w_future.result()
            except Exception:
                w = {}
        return h, w

    def _decide(
        self,
        session: dict[str, Any],
        evidence: str,
        h: dict[str, Any],
        w: dict[str, Any],
        trace_context: TraceContext | None,
    ) -> dict[str, Any]:
        """One call: item outcomes, merge/new for this turn's detection, and act(s).

        Runs every turn (acts are chosen with or without open items). A failed or
        invalid call changes no item status, never drops a detected H1-H6 cue,
        and returns valid=False so the caller can fall back.
        """
        detected = h.get("h_code") in _DETECTED_MISALIGNMENT_CODES
        open_items = unresolved_misalignment_queue(session)
        open_ids = frozenset(item["id"] for item in open_items)
        try:
            raw, trace = _call_json(
                self._llm,
                _decide_prompt(evidence, open_items, h, w, follower_trajectory_grid_text(session)),
                _joint_diagnosis_max_tokens(self._llm),
                trace_context=trace_context,
                stage="grounding.decide",
            )
            result = parse_decision(raw, open_ids, grounding_acts.ACTS)
            if trace:
                record_trace_result(trace, result)
        except Exception as error:  # Upkeep and act choice must never drop the turn.
            result = {"outcomes": {}, "detection": {}, "valid": False, "no_message": False, "acts": [],
                      "targets": [], "state": None, "reason": "", "error": f"{type(error).__name__}: {error}"}

        for item_id, outcome in result["outcomes"].items():
            if outcome["outcome"] in _CLOSING_OUTCOMES:
                close_misalignment_item(
                    session, item_id, outcome["outcome"], outcome["closure_evidence"], outcome["evidence"]
                )
            else:
                note_misalignment_evidence(session, item_id, outcome)

        result["item_id"] = None
        if detected:
            detection = result["detection"]
            still_open = {item["id"] for item in unresolved_misalignment_queue(session)}
            if detection.get("action") == "merge" and detection["item_id"] in still_open:
                merge_misalignment_detection(session, detection["item_id"], h, w)
                result["item_id"] = detection["item_id"]
            else:
                # A new episode is described by the decision's proposition, or by
                # W's observable evidence, before falling back to the H cue.
                w_evidence = next((a.get("evidence", "") for a in (w.get("assessments") or ()) if a.get("evidence")), "")
                item = open_misalignment_item(session, h, w, detection.get("proposition") or w_evidence)
                result["item_id"] = item["id"] if item else None
        # Untargeted acts on a turn with a fresh detection concern that detection's item.
        result["targets"] = [t or (result["item_id"] or "") for t in result["targets"]]
        return result

    @staticmethod
    def _no_repair(
        h: dict[str, Any],
        w: dict[str, Any],
        w_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "h": h,
            "w": w,
            "w_codes": w_codes,
            "repair_steps": (),
            "r": None,
            "r_codes": (),
            "deferred_r_codes": (),
            "m": "M5",
            "m_codes": ("M5",),
            "reply": "",
            "needs_base_reply": True,
            "normal_guide_action": (
                h.get("next_guide_action", "normal_or_wait") if h else "normal_or_wait"
            ),
            "suppress_base_reply": False,
        }
