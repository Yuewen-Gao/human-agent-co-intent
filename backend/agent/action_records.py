"""Small, framework-free helpers for preserving Agent action identities."""
from __future__ import annotations

from typing import Any, Mapping


def attach_logged_action_id(
    result: Mapping[str, Any], action_id: str | None
) -> dict[str, Any]:
    """Copy a sent-action result and attach its action-log outcome.

    A message may already be visible to the participant when the independent
    action-log write fails.  Preserve that message result so the Guide-turn
    recorder can export an explicit failed row without inventing an ID.
    """
    enriched = dict(result)
    if isinstance(action_id, str) and action_id:
        enriched["action_id"] = action_id
    else:
        errors = list(enriched.get("recording_errors") or [])
        if "action_log_missing" not in errors:
            errors.append("action_log_missing")
        enriched["recording_errors"] = errors
    return enriched
