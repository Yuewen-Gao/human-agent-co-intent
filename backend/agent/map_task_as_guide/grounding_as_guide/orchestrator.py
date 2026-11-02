"""Parallel H/W diagnosis, one item-upkeep and act-selection decision, then the reply."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from . import grounding_acts, grounding_images
from .codebook import definitions_for, definitions_for_codes
from .contract import parse_decision, parse_follower_turn, parse_h, parse_json_object, parse_repair, parse_w
from .grounding_rules import update_mode
from services.llm_trace_logger import TraceContext, record_trace_result, trace_chat_completion
from agent.map_task_as_guide.agent_session_context import (
    LAST_GUIDE_MESSAGE_KEY as _LAST_GUIDE_MESSAGE_KEY,
    _trajectory_cells,
    follower_trajectory_grid_text,
    guide_sees_follower_drawing,
)


_PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "groudingPrompts"
_CURRENT_GAME_STATE_MARKER = "\n\n<CURRENT GAME STATE>\n"

# GPT-5.6 Luna may use completion tokens for reasoning before emitting the
# required JSON. Leave enough budget for both the diagnosis and its payload.
JOINT_DIAGNOSIS_MAX_TOKENS = 4096
REPAIR_MAX_TOKENS = 2048
# A Follower turn may carry a long route; 2048 left the output empty once in replay.
FOLLOWER_TURN_MAX_TOKENS = 4096

_MISALIGNMENT_HISTORY_KEY = "grounding_misalignment_history"
_DETECTED_MISALIGNMENT_CODES = frozenset(f"H{i}" for i in range(1, 7))
# Resolved means the episode is closed; the O code records how (O4 keeps it open).
_CLOSING_OUTCOMES = frozenset({"O1", "O2", "O3"})
# The Guide's last request for evidence (Request Acknowledge or Request Repair), shown to the decision.
_EVIDENCE_REQUEST_KEY = "grounding_last_evidence_request"
_EVIDENCE_REQUEST_ACTS = frozenset({"Request Acknowledge", "Request Repair"})
# The last H/W, reused when a turn is judged on the same input as the one before.
_LAST_DETECTION_KEY = "grounding_last_detection"


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


def _validated_follower_path_points(value: Any, allow_empty: bool = False) -> list[list[int]] | None:
    if not isinstance(value, list) or (not value and not allow_empty):
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
        keep_until = value.get("keep_until")
        add = _validated_follower_path_points(value.get("add"), allow_empty=True)
        if isinstance(keep_until, bool) or not isinstance(keep_until, int) or keep_until < 0 or add is None:
            return {}
        return {
            "decision": "execute",
            "source_message_id": _decision_text(value, "source_message_id"),
            "segment": _decision_text(value, "segment"),
            "completion_criterion": _decision_text(value, "completion_criterion"),
            "message": _decision_text(value, "message"),
            "keep_until": keep_until,
            "add": add,
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
        "keep_until": None,
        "add": [],
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
    if (
        not isinstance(item_id, str) or not item_id.strip()
        or not isinstance(description, str) or not description.strip()
    ):
        return None
    outcome = value.get("outcome") if value.get("outcome") in _CLOSING_OUTCOMES else None
    return {
        "id": item_id.strip(),
        "description": description.strip(),
        # Derived: an item is closed once it has a closing O code.
        "status": "Resolved" if outcome else "Unresolved",
        "outcome": outcome,
        "strength": str(value.get("strength") or ""),
        "resolution": str(value.get("resolution") or ""),
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
        "outcome": None,
        "strength": "",
        "resolution": "",
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


def _apply_judgement(item: dict[str, Any], judgement: dict[str, Any]) -> None:
    """Keep the latest strength on the item and log the whole judgement."""
    if judgement.get("strength"):
        item["strength"] = judgement["strength"]
    item["evidence_log"].append({"timestamp": _now(), "kind": "judged", **judgement})


def close_misalignment_item(session: dict[str, Any], item_id: str, judgement: dict[str, Any]) -> bool:
    """Close one episode with its closing O code; O1/O3 also record what was settled."""
    if judgement.get("outcome") not in _CLOSING_OUTCOMES:
        return False

    def update(item: dict[str, Any]) -> None:
        _apply_judgement(item, judgement)
        item["outcome"] = judgement["outcome"]
        item["resolution"] = str(judgement.get("resolution") or "")

    target = next((item for item in unresolved_misalignment_queue(session) if item["id"] == item_id), None)
    return bool(target and _update_item(session, item_id, update))


def reopen_misalignment_item(session: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    """Reopen a closed episode the same problem came back to; its repairs stay counted.

    Its resolution did not hold, so it no longer counts as settled.
    """

    def update(item: dict[str, Any]) -> None:
        item["evidence_log"].append({"timestamp": _now(), "kind": "reopened", "previous_outcome": item.get("outcome"),
                                     "previous_resolution": item.get("resolution", "")})
        item["outcome"] = None
        item["resolution"] = ""

    target = next((item for item in load_misalignment_history(session) if item["id"] == item_id), None)
    return _update_item(session, item_id, update) if target and target["status"] == "Resolved" else None


def note_misalignment_evidence(session: dict[str, Any], item_id: str, judgement: dict[str, Any]) -> None:
    """Record a not-yet-closing (O4) judgement; the item stays open."""
    _update_item(session, item_id, lambda item: _apply_judgement(item, judgement))


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


def record_guide_message(
    session: dict[str, Any], acts: Any, reply: str, trajectory_grid: str | None = None,
    messages_seen_until: str | None = None, route_edit: dict[str, Any] | None = None,
) -> bool:
    """Remember the agent's last sent message, its act(s), and the trace it answered.

    The next decision uses it to see what the agent just asked for and what the
    Follower has drawn since, independent of any misalignment item. Pass the
    trace and the newest message time the turn was judged on: strokes and
    messages that arrived while the turn ran must count as new next turn. A
    Follower agent passes its route edit ({"keep_until", "add"}), or {} for none.
    """
    if not str(reply or "").strip():
        return False
    grid = follower_trajectory_grid_text(session) if trajectory_grid is None else trajectory_grid
    record = {
        "timestamp": _now(),
        "acts": [a for a in (acts or []) if isinstance(a, str)],
        "reply": str(reply).strip(),
        "trajectory_grid": grid.strip() if _trajectory_cells(grid) is not None else "",
    }
    if messages_seen_until is not None:
        record["messages_seen_until"] = messages_seen_until
    if route_edit is not None:
        record["route_edit"] = dict(route_edit)
    session[_LAST_GUIDE_MESSAGE_KEY] = record
    request = session.get(_EVIDENCE_REQUEST_KEY)
    if _EVIDENCE_REQUEST_ACTS & set(record["acts"]):
        session[_EVIDENCE_REQUEST_KEY] = {"acts": record["acts"], "reply": record["reply"], "guide_messages_since": 0}
    elif isinstance(request, dict):
        request["guide_messages_since"] = int(request.get("guide_messages_since") or 0) + 1
    return True


def _cell_changes(before_grid: str, current_grid: str | None) -> str:
    before, after = _trajectory_cells(before_grid or ""), _trajectory_cells(current_grid or "")
    if before is None or after is None:
        return "not available"
    added = [list(cell) for cell in sorted(after - before)]
    removed = [list(cell) for cell in sorted(before - after)]
    if not added and not removed:
        return "none (no drawing change since this message)"
    return f"added_cells: {json.dumps(added)}; removed_cells: {json.dumps(removed)}"


def _previous_guide_turn(session: dict[str, Any], current_grid: str | None) -> str:
    last = session.get(_LAST_GUIDE_MESSAGE_KEY)
    if not isinstance(last, dict) or not last.get("reply"):
        return "- none"
    acts = "+".join(last.get("acts") or []) or "unknown"
    turn = f'- acts: {acts}\n- message: "{last["reply"]}"'
    if guide_sees_follower_drawing(session):
        turn += f"\n- Follower trajectory changes since this message: {_cell_changes(last.get('trajectory_grid', ''), current_grid)}"
    return turn + _last_evidence_request(session, "Guide")


def _previous_follower_turn(session: dict[str, Any]) -> str:
    last = session.get(_LAST_GUIDE_MESSAGE_KEY)
    if not isinstance(last, dict) or not last.get("reply"):
        return "- none"
    acts = "+".join(last.get("acts") or []) or "unknown"
    edit = last.get("route_edit") or {}
    if edit.get("keep_until") is None:
        drawing = "no drawing update"
    else:
        added = edit.get("add") or []
        kept = f"kept points 1-{edit['keep_until']}" if edit["keep_until"] else "started a new route"
        drawing = (f"{kept}, added {len(added)} point(s)"
                   + (f" ending at {json.dumps(added[-1])}" if added else ""))
    return (f'- acts: {acts}\n- message: "{last["reply"]}"\n- drawing update sent with it: {drawing}'
            + _last_evidence_request(session, "Follower"))


def _last_evidence_request(session: dict[str, Any], role_name: str) -> str:
    request = session.get(_EVIDENCE_REQUEST_KEY)
    if not isinstance(request, dict) or not request.get("reply"):
        return ""
    since = int(request.get("guide_messages_since") or 0)
    when = "this message" if since == 0 else f"{since} {role_name} message(s) before this one"
    return (f'\n- your last request for evidence ({"+".join(request.get("acts") or [])}): {when}: '
            f'"{request["reply"]}"; the partner\'s answer, if any, is in the conversation')


def _recently_closed(session: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    """The closed items the decision sees, and so the only ones it may reopen."""
    return [item for item in load_misalignment_history(session) if item["status"] == "Resolved"][-limit:]


def _recently_closed_items(session: dict[str, Any]) -> str:
    closed = _recently_closed(session)
    if not closed:
        return "- none"
    lines = []
    for item in closed:
        last = next((e for e in reversed(item["evidence_log"]) if e.get("kind") in ("judged", "outcome")), {})
        lines.append(f"- {item['id']} | {item['description']} | repairs sent: {len(item['repair_attempts'])}"
                     f" | closed as {item.get('outcome')}: {str(last.get('evidence', ''))[:160]}")
    return "\n".join(lines)


def settled_agreements(session: dict[str, Any]) -> str:
    """What closed items established (O1) or explicitly left different (O3), oldest first."""
    lines = [f"- {item['id']} ({item['outcome']}): {item['resolution'] or item['description']}"
             for item in load_misalignment_history(session)
             if item["outcome"] in ("O1", "O3")]
    return "\n".join(lines) or "- none"


def _settled_block(session: dict[str, Any]) -> str:
    return (
        "<SETTLED AGREEMENTS>\n"
        "What earlier repairs established with the partner. Keep to them unless new evidence contradicts one; "
        "do not raise a settled difference again as if it were new.\n"
        + settled_agreements(session)
        + "\n</SETTLED AGREEMENTS>"
    )


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
            f" | W: {', '.join(item['w_codes']) or '-'}"
            f" | strength: {item['strength'] or 'not judged'}"
            f" | repairs sent: {len(attempts)} ({last_repair})"
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
        elif kind == "judged":
            parts.append(f"judged {entry.get('outcome')} {entry.get('strength') or '?'}"
                         + f": {str(entry.get('evidence', ''))[:160]}")
        elif kind == "outcome":  # judgements logged before evidence strength
            parts.append(f"outcome {entry.get('outcome')}: {str(entry.get('evidence', ''))[:160]}")
        elif kind == "reopened":
            parts.append(f"reopened after {entry.get('previous_outcome')}: the same problem came back")
        elif kind == "act":
            parts.append(f"act {'+'.join(entry.get('acts') or [])}")
    return " | ".join(parts)


def acts_by_item(acts: list[str], targets: list[str]) -> dict[str, list[str]]:
    """Each targeted item with the act(s) of this turn that target it, in order."""
    grouped: dict[str, list[str]] = {}
    for act, target in zip(acts, targets):
        if target:
            grouped.setdefault(target, []).append(act)
    return grouped


def _item_rounds(
    open_before: list[dict[str, Any]], result: dict[str, Any], closed_before: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """This turn for each item open before it or touched by its detection: what happened and the acts aimed at it.

    An item with no acts this turn (including a no_message turn) still gets an entry, so an
    item's lifecycle can be read turn by turn from the decision records.
    """
    grouped = acts_by_item(result.get("acts") or [], result.get("targets") or [])
    selected = acts_by_item(result.get("selected_acts_all") or [], result.get("selected_targets_all") or [])
    open_ids = [item["id"] for item in open_before]
    ids = open_ids + ([result["item_id"]] if result.get("item_id") and result["item_id"] not in open_ids else [])
    rounds = []
    for item_id in ids:
        judged = (result.get("outcomes") or {}).get(item_id) or {}
        if item_id != result.get("item_id"):
            event = ""
        elif item_id in open_ids:
            event = "merged"
        else:
            event = "reopened" if item_id in closed_before else "opened"
        rounds.append({"id": item_id, "event": event, "outcome": judged.get("outcome"),
                       "strength": judged.get("strength"), "acts": grouped.get(item_id, []),
                       "selected_acts": selected.get(item_id, [])})
    return rounds


def record_selected_acts(session: dict[str, Any], acts: list[str], targets: list[str]) -> None:
    """Remember, for each targeted item, the act(s) of this turn that target it."""
    for item_id, item_acts in acts_by_item(acts, targets).items():
        _update_item(session, item_id, lambda item, item_acts=item_acts: item["evidence_log"].append({
            "timestamp": _now(), "kind": "act", "acts": list(item_acts),
        }))


_H_RESPONSE_SCHEMA = (
    '{"h":{"h_code":"H1..H7","confidence":"low|medium|high",'
    '"evidence":"short observable detection cue",'
    '"trajectory_verdict":"correct|incorrect|insufficient_evidence|not_applicable",'
    '"trajectory_assessment":{"fresh_trace_relation":"advances|completes|departs|not_assessable|not_applicable",'
    '"observable_departure":"specific landmark-relative departure or empty",'
    '"instruction_conflict":"the part of the latest Guide instruction that the Guide map contradicts, or empty"},'
    '"next_guide_action":"repair|give_next_segment|normal_or_wait"}}'
)
# The Follower makes no trajectory judgement: those fields stay not_applicable.
_FOLLOWER_H_RESPONSE_SCHEMA = (
    '{"h":{"h_code":"H1..H7","confidence":"low|medium|high",'
    '"evidence":"short observable detection cue",'
    '"trajectory_verdict":"not_applicable",'
    '"trajectory_assessment":{"fresh_trace_relation":"not_applicable","observable_departure":"","instruction_conflict":""},'
    '"next_guide_action":"repair|normal_or_wait"}}'
)
_W_RESPONSE_SCHEMA = (
    '{"w":{"w_codes":["W1..W13"],"assessments":[{"w_code":"W1..W13",'
    '"evidence":"short observable support",'
    '"repair_focus":"neutral sentence describing what the reply should clarify"}]}}'
)


def _h_diagnosis_prompt(evidence: str, session: dict[str, Any] | None = None, role: str = "guide") -> str:
    """H detection only; W is diagnosed by an independent, parallel request.

    H sees what earlier repairs settled (not the open items), so a known
    difference is not detected again as new. The codebook is shared; the
    detection instructions say how the agent's role applies it.
    """
    follower = role == "follower"
    h_instruction = (_PROMPTS / ("grounding_detection_prompt_follower.txt" if follower
                                 else "grounding_detection_prompt.txt")).read_text(encoding="utf-8")
    return (
        "<H DETECTION INSTRUCTIONS>\n"
        + h_instruction
        + "\n</H DETECTION INSTRUCTIONS>\n\n<H CODEBOOK>\n"
        + definitions_for("H")
        + "\n</H CODEBOOK>\n\n"
        + _settled_block(session or {})
        + "\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n<H RESPONSE CONTRACT>\n"
        + "Apply the H instructions and codebook to the current evidence. Return one JSON object only:\n"
        + (_FOLLOWER_H_RESPONSE_SCHEMA if follower else _H_RESPONSE_SCHEMA)
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


def _call_json(llm_client: Any, prompt: str, max_tokens: int, *, trace_context: TraceContext | None = None, stage: str = "",
               images: list[dict[str, Any]] | None = None, image_note: str | None = None) -> tuple[str, Any | None]:
    messages = [{"role": "user", "content": grounding_images.user_content(prompt, images, image_note)}]
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
    session: dict[str, Any] | None = None,
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
        + _settled_block(session or {})
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
    session: dict[str, Any] | None = None,
    reused: bool = False,
    role: str = "guide",
) -> str:
    """Build the combined item-upkeep and act-selection request.

    It receives this turn's H/W results and each item's recent judgements so
    that earlier judgements are reused rather than re-derived. Item upkeep is
    shared; `<ROLE ACTS>` says how the agent's role chooses acts.
    """
    background, observable_evidence = _split_guide_background(evidence)
    follower = role == "follower"
    instruction = (_PROMPTS / "grounding_decide_prompt.txt").read_text(encoding="utf-8")
    # The task rules and goal are basic context for every call; the rest of the static prompt
    # (the plain agent's actions and response format) does not belong in this request.
    task = "\n\n".join(re.findall(r"<EXPERIMENT (?:RULES|GOALS)>.*?</EXPERIMENT (?:RULES|GOALS)>", background, re.S))
    role_acts = (_PROMPTS / ("decide_acts_follower.txt" if follower else "decide_acts_guide.txt")).read_text(encoding="utf-8")
    previous_turn = (
        "<PREVIOUS FOLLOWER TURN>\n" + _previous_follower_turn(session or {}) + "\n</PREVIOUS FOLLOWER TURN>"
        if follower else
        "<PREVIOUS GUIDE TURN>\n" + _previous_guide_turn(session or {}, current_trajectory_grid) + "\n</PREVIOUS GUIDE TURN>"
    )
    detection = {
        "h": {key: h.get(key) for key in ("h_code", "evidence", "trajectory_verdict", "trajectory_assessment")},
        "w": [{key: a.get(key) for key in ("w_code", "evidence", "repair_focus")} for a in (w.get("assessments") or ())],
    }
    if reused:
        detection["reused_from_previous_turn"] = True
    return (
        (task + "\n\n" if task else "")
        + "<DECISION INSTRUCTIONS>\n"
        + instruction
        + "\n</DECISION INSTRUCTIONS>\n\n<ROLE ACTS>\n"
        + role_acts
        + "\n</ROLE ACTS>\n\n<GROUNDING ACT DEFINITIONS>\n"
        + grounding_acts.definitions_text()
        + "\n</GROUNDING ACT DEFINITIONS>\n\n<O CODEBOOK>\n"
        + definitions_for("O")
        + "\n</O CODEBOOK>\n\n<EVIDENCE STRENGTH>\n"
        + definitions_for("E")
        + "\n</EVIDENCE STRENGTH>\n\n"
        + _settled_block(session or {})
        + "\n\n<UNRESOLVED MISALIGNMENT ITEMS>\n"
        + (_format_open_items(open_items, current_trajectory_grid) if open_items else "- none")
        + "\n</UNRESOLVED MISALIGNMENT ITEMS>\n\n<RECENTLY CLOSED ITEMS>\n"
        + _recently_closed_items(session or {})
        + "\n</RECENTLY CLOSED ITEMS>\n\n"
        + previous_turn
        + "\n\n<CURRENT DETECTION>\n"
        + json.dumps(detection, ensure_ascii=False)
        + "\n</CURRENT DETECTION>\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + (observable_evidence or evidence)
        + "\n</CURRENT OBSERVABLE EVIDENCE>\n\n<RESPONSE CONTRACT>\n"
        + (_FOLLOWER_DECISION_CONTRACT if follower else _DECISION_CONTRACT)
        + "\n</RESPONSE CONTRACT>"
    )


_DECISION_CONTRACT = (
    'Return one JSON object only:\n'
    '{"items":[{"id":"open item id","outcome":"O1|O2|O3|O4","strength":"E0|E1|E2|E3|E4",'
    '"evidence":"the strongest evidence type and what it showed",'
    '"resolution":"for O1/O3: what both now share that settles the problem (interpretation, map fact, or reference), in one sentence; otherwise empty"}],'
    '"detection":{"action":"merge|new|none","item_id":"open or recently closed item id for merge, otherwise empty",'
    '"proposition":"short disputed proposition for new, otherwise empty"},'
    '"state":"no_issue|specific_correctable|missing_information|instruction_invalidated",'
    '"no_message":false,"reason":"short reason",'
    '"acts":[{"act":"Initiate|Continue|Acknowledge|Repair|Request Repair|Request Acknowledge|Cancel",'
    '"target_item_id":"open or reopened item id, \\"new\\" for the item this turn\'s detection opens, '
    'or empty when the act concerns no item","evidence":"short observable reason"}],"primary_act":"one of the acts"}\n'
    'When no_message is true, acts must be empty; otherwise give one to three different acts. '
    'Return exactly one entry per open item.'
)
_FOLLOWER_DECISION_CONTRACT = _DECISION_CONTRACT.replace(
    '"no_message":false,', '"no_message":false,"execute":false,', 1,
) + ' "execute" is true only when this turn draws.'


_PLAIN_FOLLOWER_SECTIONS = re.compile(r"<(DECISION|RESPONSE FORMAT)>.*?</\1>\n*", re.DOTALL)


def follower_evidence(evidence: str) -> str:
    """The Follower evidence without the plain Follower's own decision rule and output format.

    The grounding pipeline decides and has its own contracts; the task
    prompt's role and information boundary stay as background.
    """
    background, observations = _split_guide_background(evidence)
    return _PLAIN_FOLLOWER_SECTIONS.sub("", background) + observations


_PLAIN_GUIDE_SECTIONS = "\n<ACTION PLANNING AND RESPONSES>\n"


def guide_evidence(evidence: str) -> str:
    """The Guide evidence without the plain Guide's own planning, action space and output format.

    As for the Follower: the grounding pipeline decides and has its own contracts;
    the task prompt's role, rules, goals, setup and perception stay as background.
    These plain-Guide sections close the static prompt.
    """
    background, observations = _split_guide_background(evidence)
    return background.split(_PLAIN_GUIDE_SECTIONS, 1)[0].rstrip("\n") + observations


def _follower_turn_prompt(
    evidence: str, act_block: str, session: dict[str, Any], execute: bool, for_repair: bool,
) -> str:
    """The Follower's generation request: task background, the decided drawing and acts, then the evidence."""
    background, observable_evidence = _split_guide_background(evidence)
    instruction = (_PROMPTS / "grounding_realize_prompt_follower.txt").read_text(encoding="utf-8")
    drawing = ("yes: draw the newest instruction (or the redraw) on your map" if execute
               else 'no: return "execute": false and no route edit')
    return (
        "<MAP TASK FOLLOWER BACKGROUND>\n"
        + background.strip()
        + "\n</MAP TASK FOLLOWER BACKGROUND>\n\n<GENERATION INSTRUCTIONS>\n"
        + instruction
        + "\n</GENERATION INSTRUCTIONS>\n\n"
        + ("<R CODEBOOK>\n" + definitions_for("R") + "\n</R CODEBOOK>\n\n" if for_repair else "")
        + "<DRAWING DECISION>\nDraw this turn: " + drawing + "\n</DRAWING DECISION>\n\n"
        + act_block
        + "\n\n"
        + _settled_block(session)
        + "\n\n<CURRENT OBSERVABLE EVIDENCE>\n"
        + observable_evidence
        + "\n</CURRENT OBSERVABLE EVIDENCE>"
    )


def safe_decide_follower(
    orchestrator: Any,
    evidence: str,
    *,
    session: dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
) -> dict[str, Any] | None:
    """Return no grounding Follower turn when it cannot complete; the caller then uses the plain Follower."""
    try:
        result = orchestrator.decide_follower(evidence, session=session, trace_context=trace_context)
    except Exception:
        return None
    return result if result.get("valid") else None


def safe_decide(
    orchestrator: Any,
    evidence: str,
    *,
    session: dict[str, Any] | None = None,
    trace_context: TraceContext | None = None,
    reuse_detection: bool = False,
    on_reply: Any = None,
) -> dict[str, Any] | None:
    """Return no grounding decision when diagnosis cannot complete.

    The caller can then use the normal Guide reply path instead of dropping a
    participant-visible turn because a treatment-only LLM call failed.
    """
    try:
        return orchestrator.decide(evidence, session=session, trace_context=trace_context,
                                   reuse_detection=reuse_detection, on_reply=on_reply)
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
        reuse_detection: bool = False,
        on_reply: Any = None,
    ) -> dict[str, Any]:
        """Diagnose H/W, then one decision call for item upkeep and act selection, then the reply.

        `on_reply` is called once the turn will send a message, before the reply is written.

        Each judgement is made once and reused: H owns the trajectory judgement,
        the decision call owns item outcomes and acts, and the reply stage only
        words the selected act. With `reuse_detection` (same input as the last
        turn, only the waiting time differs) the last H/W stand, and their cue
        is not recorded again.
        """
        session = session if session is not None else {}
        evidence = guide_evidence(evidence)
        # Map images help the stages that judge route direction: H and the reply.
        images = (grounding_images.map_image_parts(session)
                  if getattr(self._llm, "supports_multimodal_images", lambda: False)() else [])
        last = session.get(_LAST_DETECTION_KEY)
        reused = reuse_detection and isinstance(last, dict) and isinstance(last.get("h"), dict) and bool(last["h"])
        if reused:
            h, w = last["h"], last.get("w") or {}
        else:
            h, w = self._diagnose_h_and_w(evidence, trace_context, images, session)
            session[_LAST_DETECTION_KEY] = {"h": h, "w": w}
        decision = self._decide(session, evidence, h, w, trace_context, reused)
        w_codes = w.get("w_codes", ())
        item_fields = {
            "misalignment_item_id": decision["item_id"],
            "decision": {key: decision[key] for key in
                         ("outcomes", "detection", "acts", "targets", "item_rounds", "h_invalid", "no_message", "state", "reason", "valid",
                          "selected_acts_all", "selected_targets_all", "primary_act", "reply_act_mode", "acts_dropped")
                         if key in decision},
        }
        if decision.get("error"):
            item_fields["decision"]["error"] = decision["error"]

        if not decision["valid"]:
            # Fallback when the decision call fails: the earlier H-gated routing (a reused cue was handled).
            if not h or h.get("h_code") == "H7" or reused:
                return {**self._no_repair(h, w, w_codes), **item_fields}
            decision.update(acts=["Repair"], targets=[decision["item_id"] or ""], reason=str(h.get("evidence") or ""))
        elif decision["no_message"]:
            return {**self._no_repair(h, w, w_codes), **item_fields,
                    "needs_base_reply": False, "suppress_base_reply": True}
        if on_reply:
            on_reply()

        queue = {item["id"]: item for item in unresolved_misalignment_queue(session)}
        target_id = next((t for t in decision["targets"] if t), decision["item_id"])
        target = queue.get(target_id) if target_id else None
        status = decision.get("outcomes", {}).get(target_id) if target_id else None
        record_selected_acts(session, decision["acts"], decision["targets"])
        for_repair = bool(set(decision["acts"]) & grounding_acts.REPAIR_STAGE_ACTS)
        targets = decision["targets"] or [""] * len(decision["acts"])
        block = grounding_acts.act_block(decision["acts"], targets, decision["reason"], target, status,
                                         for_repair=for_repair)
        if not for_repair:
            # The normal reply reuses this turn's images, so it sees the same drawing as H.
            return {**self._no_repair(h, w, w_codes), **item_fields, "selected_acts": decision["acts"],
                    "act_block": block + "\n" + _settled_block(session), "_map_images": images}

        h_for_r = h if h.get("h_code") in _DETECTED_MISALIGNMENT_CODES else {
            "h_code": (target or {}).get("h_codes", ["H2"])[-1] if target and target.get("h_codes") else "H2",
            "evidence": (target or {}).get("description", ""),
            "trajectory_verdict": "not_applicable",
            "trajectory_assessment": {"fresh_trace_relation": "not_applicable", "observable_departure": ""},
        }
        repair_prompt = _repair_prompt(evidence, h_for_r, w, session).replace(
            "\n<CURRENT OBSERVABLE EVIDENCE>\n", "\n" + block + "\n\n<CURRENT OBSERVABLE EVIDENCE>\n", 1)
        repair_raw, repair_trace = _call_json(
            self._llm,
            repair_prompt,
            REPAIR_MAX_TOKENS,
            trace_context=trace_context,
            stage="grounding.repair",
            images=images,
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

    def decide_follower(
        self,
        evidence: str,
        *,
        session: dict[str, Any] | None = None,
        trace_context: TraceContext | None = None,
    ) -> dict[str, Any]:
        """Grounding Follower turn: H/W on the Guide's newest message, the shared item
        upkeep with the Follower's act selection, then one call that writes the
        drawing update and the message.

        Returns valid=False when the decision or the generated turn fails, so the
        caller can use the plain Follower path. A turn draws only when both the
        decision and the generated turn say so.
        """
        session = session if session is not None else {}
        evidence = follower_evidence(evidence)
        # As for the Guide, the image goes to H and the generated turn: here the Follower's own map and drawing.
        images = (grounding_images.follower_image_parts(session)
                  if getattr(self._llm, "supports_multimodal_images", lambda: False)() else [])
        note = grounding_images.FOLLOWER_IMAGE_NOTE
        h, w = self._diagnose_h_and_w(evidence, trace_context, images, session, role="follower", image_note=note)
        session[_LAST_DETECTION_KEY] = {"h": h, "w": w}
        decision = self._decide(session, evidence, h, w, trace_context, role="follower")
        result = {
            "h": h, "w": w, "w_codes": w.get("w_codes", ()), "valid": decision["valid"],
            "no_message": decision["no_message"], "execute": False, "keep_until": None, "add": [], "message": "",
            "selected_acts": decision["acts"], "repair_steps": (), "r_codes": (),
            "misalignment_item_id": decision["item_id"],
            "decision": {key: decision[key] for key in
                         ("outcomes", "detection", "acts", "targets", "item_rounds", "h_invalid", "no_message", "execute", "state", "reason",
                          "selected_acts_all", "selected_targets_all", "primary_act", "reply_act_mode", "acts_dropped",
                          "valid", "error") if key in decision},
        }
        if not decision["valid"] or decision["no_message"]:
            return result

        queue = {item["id"]: item for item in unresolved_misalignment_queue(session)}
        target_id = next((t for t in decision["targets"] if t), decision["item_id"])
        target = queue.get(target_id) if target_id else None
        status = decision.get("outcomes", {}).get(target_id) if target_id else None
        record_selected_acts(session, decision["acts"], decision["targets"])
        for_repair = bool(set(decision["acts"]) & grounding_acts.REPAIR_STAGE_ACTS)
        targets = decision["targets"] or [""] * len(decision["acts"])
        block = grounding_acts.act_block(decision["acts"], targets, decision["reason"], target, status,
                                         for_repair=for_repair)
        raw, trace = _call_json(
            self._llm,
            _follower_turn_prompt(evidence, block, session, decision["execute"], for_repair),
            FOLLOWER_TURN_MAX_TOKENS,
            trace_context=trace_context,
            stage="grounding.follower_turn",
            images=images,
            image_note=note,
        )
        turn = parse_follower_turn(raw, allow_r7=grounding_acts.r7_allowed(target))
        if trace:
            record_trace_result(trace, turn)
        execute = decision["execute"] and turn["execute"]
        if not turn["message"] and not execute:
            result.update(valid=False, error="empty_follower_turn")
            return result
        result.update(
            execute=execute,
            keep_until=turn["keep_until"] if execute else None,
            add=turn["add"] if execute else [],
            message=turn["message"],
            repair_steps=turn["repair_steps"],
            r_codes=turn["r_codes"],
            for_repair=for_repair,
            edit_dropped=turn["edit_dropped"],
            # The repair is recorded against the item the act targeted.
            misalignment_item_id=target_id or decision["item_id"],
        )
        return result

    def _diagnose_h_and_w(
        self, evidence: str, trace_context: TraceContext | None, images: list[dict[str, Any]] | None = None,
        session: dict[str, Any] | None = None, role: str = "guide", image_note: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run H and W as independent requests in parallel.

        An H failure propagates (``safe_decide`` then falls back to the normal
        Guide path). A W failure only drops W, which enriches repair but never
        gates it.
        """
        budget = _joint_diagnosis_max_tokens(self._llm)

        def run(prompt: str, stage: str, key: str, parser: Any, stage_images: Any = None) -> dict[str, Any]:
            raw, trace = _call_json(self._llm, prompt, budget, trace_context=trace_context, stage=stage,
                                    images=stage_images, image_note=image_note)
            parsed = parser(_nested(raw, key))
            if trace:
                record_trace_result(trace, parsed)
            return parsed

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="grounding-diagnosis") as pool:
            h_future = pool.submit(run, _h_diagnosis_prompt(evidence, session, role), "grounding.h", "h", partial(parse_h, role=role), images)
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
        reused: bool = False,
        role: str = "guide",
    ) -> dict[str, Any]:
        """One call: item outcomes, merge/new for this turn's detection, and act(s).

        Runs every turn (acts are chosen with or without open items). A failed or
        invalid call changes no item status, never drops a detected H1-H6 cue,
        and returns valid=False so the caller can fall back.
        """
        detected = h.get("h_code") in _DETECTED_MISALIGNMENT_CODES and not reused
        open_items = unresolved_misalignment_queue(session)
        open_ids = frozenset(item["id"] for item in open_items)
        closed_ids = frozenset(item["id"] for item in _recently_closed(session))
        try:
            raw, trace = _call_json(
                self._llm,
                _decide_prompt(evidence, open_items, h, w, follower_trajectory_grid_text(session), session, reused, role),
                _joint_diagnosis_max_tokens(self._llm),
                trace_context=trace_context,
                stage="grounding.decide",
            )
            result = parse_decision(raw, open_ids, grounding_acts.ACTS, closed_ids)
            if trace:
                record_trace_result(trace, result)
        except Exception as error:  # Upkeep and act choice must never drop the turn.
            result = {"outcomes": {}, "detection": {}, "valid": False, "no_message": False, "acts": [],
                      "targets": [], "state": None, "reason": "", "execute": False,
                      "error": f"{type(error).__name__}: {error}"}

        for item_id, outcome in result["outcomes"].items():
            if outcome["outcome"] in _CLOSING_OUTCOMES:
                close_misalignment_item(session, item_id, outcome)
            else:
                note_misalignment_evidence(session, item_id, outcome)

        result["item_id"] = None
        if detected:
            detection = result["detection"]
            still_open = {item["id"] for item in unresolved_misalignment_queue(session)}
            if detection.get("action") == "merge" and detection["item_id"] in closed_ids:
                # The same problem came back after its item was closed: continue that episode.
                reopen_misalignment_item(session, detection["item_id"])
                still_open.add(detection["item_id"])
            if detection.get("action") == "merge" and detection["item_id"] in still_open:
                merge_misalignment_detection(session, detection["item_id"], h, w)
                result["item_id"] = detection["item_id"]
            else:
                # A new episode is described by the decision's proposition, or by
                # W's observable evidence, before falling back to the H cue.
                w_evidence = next((a.get("evidence", "") for a in (w.get("assessments") or ()) if a.get("evidence")), "")
                item = open_misalignment_item(session, h, w, detection.get("proposition") or w_evidence)
                result["item_id"] = item["id"] if item else None
        # "new" is the item this turn's detection opened (or merged into); an untargeted repair-stage
        # act on such a turn concerns it too. Any other untargeted act concerns no item.
        this_item = result["item_id"] or ""
        result["targets"] = [
            this_item if target == "new" or (not target and act in grounding_acts.REPAIR_STAGE_ACTS) else target
            for act, target in zip(result["acts"], result["targets"])
        ]
        # Every selected act is recorded; the reply performs all of them, or only the primary act when
        # GROUNDING_REPLY_ACTS=primary (experiment switch, read per turn).
        result["selected_acts_all"], result["selected_targets_all"] = list(result["acts"]), list(result["targets"])
        result["reply_act_mode"] = os.environ.get("GROUNDING_REPLY_ACTS", "all")
        if result["reply_act_mode"] == "primary" and result["acts"]:
            i = result["acts"].index(result["primary_act"]) if result.get("primary_act") in result["acts"] else 0
            result["acts"], result["targets"] = [result["acts"][i]], [result["targets"][i]]
        result["item_rounds"] = _item_rounds(open_items, result, closed_ids)
        # An H answer that failed validation leaves no detection; recorded so it is not silent.
        result["h_invalid"] = not h
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
            "m": "M5",
            "m_codes": ("M5",),
            "reply": "",
            "needs_base_reply": True,
            "normal_guide_action": (
                h.get("next_guide_action", "normal_or_wait") if h else "normal_or_wait"
            ),
            "suppress_base_reply": False,
        }
