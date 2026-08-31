"""Session-scoped storage and conflict handling for visible interpretation state."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from config.grounding_treatment import grounding_treatment_enabled
from agent.map_task_as_guide.grounding_as_guide.mental_model_update import (
    MENTAL_MODEL_FIELD_KEYS,
)


UNKNOWN_SMM_VALUE = "Not yet observable from current collaboration."
_ANNOTATION_KEYS = (
    "explanation_transcription",
    "task_model_q1",
    "partner_model_q2",
    "self_model_q3",
    "alignment_q4",
)


def _revision(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _model(session: dict[str, Any]) -> dict[str, Any]:
    candidate = session.get("mental_model")
    if not isinstance(candidate, dict):
        candidate = {"revision": 0, "fields": {}}
        session["mental_model"] = candidate
    candidate["revision"] = _revision(candidate.get("revision"))
    if not isinstance(candidate.get("fields"), dict):
        candidate["fields"] = {}
    return candidate


def _unknown_field() -> dict[str, Any]:
    return {
        "value": UNKNOWN_SMM_VALUE,
        "status": "not_observable",
        "confidence": "low",
        "source": "system",
        "locked": False,
    }


def ensure_complete_mental_model(session: dict[str, Any]) -> dict[str, Any]:
    """Ensure all W-backed SMM fields exist without inventing task facts.

    This is deliberately treatment-independent: it supports background research
    records in control sessions.  Visibility and participant edits remain gated
    by the public treatment-facing functions below.
    """
    model = _model(session)
    for key in MENTAL_MODEL_FIELD_KEYS:
        existing = model["fields"].get(key)
        if isinstance(existing, Mapping):
            entry = dict(existing)
            value = entry.get("value")
            if not isinstance(value, str) or not value.strip():
                entry = _unknown_field()
            else:
                entry["value"] = value.strip()
                entry.setdefault(
                    "status",
                    "not_observable" if entry["value"] == UNKNOWN_SMM_VALUE else "inferred",
                )
                entry.setdefault("confidence", "low")
                entry.setdefault("source", "legacy")
                entry.setdefault("locked", False)
            model["fields"][key] = entry
        elif isinstance(existing, str) and existing.strip():
            model["fields"][key] = {
                "value": existing.strip(),
                "status": "inferred",
                "confidence": "low",
                "source": "legacy",
                "locked": False,
            }
        else:
            model["fields"][key] = _unknown_field()
    return model


def public_mental_model(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI-safe values and display metadata, never internal locks/source."""
    raw = session.get("mental_model") if isinstance(session, Mapping) else None
    if not isinstance(raw, Mapping):
        return {"revision": 0, "fields": {}}
    fields = raw.get("fields")
    result: dict[str, dict[str, str]] = {}
    if isinstance(fields, Mapping):
        for key in MENTAL_MODEL_FIELD_KEYS:
            item = fields.get(key)
            value = item.get("value") if isinstance(item, Mapping) else item
            if isinstance(value, str):
                result[key] = {
                    "value": value,
                    "status": (
                        str(item.get("status"))
                        if isinstance(item, Mapping) and isinstance(item.get("status"), str)
                        else ("not_observable" if value == UNKNOWN_SMM_VALUE else "inferred")
                    ),
                    "confidence": (
                        str(item.get("confidence"))
                        if isinstance(item, Mapping) and isinstance(item.get("confidence"), str)
                        else "low"
                    ),
                }
    return {"revision": _revision(raw.get("revision")), "fields": result}


def should_publish_mental_model_update(changed: bool) -> bool:
    """Keep background control records from becoming a participant UI effect."""
    return bool(changed) and grounding_treatment_enabled()


def append_agent_turn_record(
    session: dict[str, Any],
    *,
    action_id: str | None,
    message_id: str | None,
    annotation: Mapping[str, Any] | None,
    agent_participant_id: str | None = None,
    trajectory_grid_at_reply: str | None = None,
    recording_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Append one post-turn, exportable Guide-Agent record.

    ``action_id`` is the action logger UUID and therefore the primary key used
    in exports.  A deep copy prevents later SMM mutation from rewriting history.
    """
    model = ensure_complete_mental_model(session)
    normalized_annotation = {
        key: str((annotation or {}).get(key, "")).strip()
        for key in _ANNOTATION_KEYS
    }
    record = {
        "action_id": action_id if isinstance(action_id, str) and action_id else None,
        "message_id": message_id if isinstance(message_id, str) and message_id else None,
        "annotation": normalized_annotation,
        "current_smm": deepcopy(model),
        "agent_participant_id": (
            agent_participant_id
            if isinstance(agent_participant_id, str) and agent_participant_id
            else None
        ),
        "trajectory_grid_at_reply": (
            trajectory_grid_at_reply.strip()
            if isinstance(trajectory_grid_at_reply, str) and trajectory_grid_at_reply.strip()
            else None
        ),
        "recording_errors": [
            error for error in (recording_errors or [])
            if isinstance(error, str) and error
        ],
    }
    record["recording_status"] = "failed" if record["recording_errors"] else "complete"
    session.setdefault("agent_turn_records", []).append(record)
    return record


def apply_user_mental_model_update(
    session: dict[str, Any], base_revision: Any, fields: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply participant-authored fields and lock each changed field against agents."""
    if not grounding_treatment_enabled():
        return {"changed": False, "reason": "grounding_treatment_disabled", "mental_model": public_mental_model(session)}
    model = _model(session)
    if base_revision != model["revision"]:
        return {"changed": False, "reason": "stale_revision", "mental_model": public_mental_model(session)}
    if not isinstance(fields, Mapping):
        return {"changed": False, "reason": "invalid_fields", "mental_model": public_mental_model(session)}

    changed = False
    for key in MENTAL_MODEL_FIELD_KEYS:
        value = fields.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        previous = model["fields"].get(key)
        if (
            isinstance(previous, Mapping)
            and previous.get("value") == normalized
            and previous.get("source") == "user"
            and previous.get("locked") is True
        ):
            continue
        model["fields"][key] = {
            "value": normalized,
            "status": "confirmed",
            "confidence": "high",
            "source": "user",
            "locked": True,
        }
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}


def _apply_agent_mental_model_patch(
    session: dict[str, Any], patch: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply a validated agent patch after the caller chooses its visibility policy."""
    model = _model(session)
    if not isinstance(patch, Mapping):
        return {"changed": False, "reason": "invalid_patch", "mental_model": public_mental_model(session)}
    if patch.get("base_revision") != model["revision"]:
        return {"changed": False, "reason": "stale_revision", "mental_model": public_mental_model(session)}

    raw_changes = patch.get("changes")
    if not isinstance(raw_changes, Mapping):
        return {"changed": False, "reason": "invalid_patch", "mental_model": public_mental_model(session)}

    changed = False
    for key in MENTAL_MODEL_FIELD_KEYS:
        change = raw_changes.get(key)
        if not isinstance(change, Mapping) or not isinstance(change.get("value"), str):
            continue
        previous = model["fields"].get(key)
        if isinstance(previous, Mapping) and previous.get("locked") is True:
            continue

        value = change["value"].strip()
        if not value or (isinstance(previous, Mapping) and previous.get("value") == value):
            continue
        entry = {
            "value": value,
            "status": "inferred",
            "source": "agent",
            "locked": False,
        }
        if isinstance(change.get("confidence"), str):
            entry["confidence"] = change["confidence"]
        if isinstance(change.get("evidence_message_ids"), list):
            entry["evidence_message_ids"] = change["evidence_message_ids"]
        model["fields"][key] = entry
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}


def apply_internal_agent_mental_model_patch(
    session: dict[str, Any], patch: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply a background recorder patch without enabling treatment UI/prompt effects."""
    return _apply_agent_mental_model_patch(session, patch)
