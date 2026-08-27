"""Session-scoped storage and conflict handling for visible interpretation state."""
from __future__ import annotations

from typing import Any, Mapping

from agent.map_task.mental_model_update import MENTAL_MODEL_FIELD_KEYS


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


def public_mental_model(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return only values required by the participant UI, never internal locks."""
    raw = session.get("mental_model") if isinstance(session, Mapping) else None
    if not isinstance(raw, Mapping):
        return {"revision": 0, "fields": {}}
    fields = raw.get("fields")
    result: dict[str, str] = {}
    if isinstance(fields, Mapping):
        for key in MENTAL_MODEL_FIELD_KEYS:
            item = fields.get(key)
            value = item.get("value") if isinstance(item, Mapping) else item
            if isinstance(value, str):
                result[key] = value
    return {"revision": _revision(raw.get("revision")), "fields": result}


def apply_user_mental_model_update(
    session: dict[str, Any], base_revision: Any, fields: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Apply participant-authored fields and lock each changed field against agents."""
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
        model["fields"][key] = {"value": normalized, "source": "user", "locked": True}
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}


def apply_agent_mental_model_patch(session: dict[str, Any], patch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply a validated patch if it was generated from the latest revision."""
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
        entry = {"value": value, "source": "agent", "locked": False}
        if isinstance(change.get("confidence"), str):
            entry["confidence"] = change["confidence"]
        if isinstance(change.get("evidence_message_ids"), list):
            entry["evidence_message_ids"] = change["evidence_message_ids"]
        model["fields"][key] = entry
        changed = True

    if changed:
        model["revision"] += 1
    return {"changed": changed, "reason": None, "mental_model": public_mental_model(session)}
