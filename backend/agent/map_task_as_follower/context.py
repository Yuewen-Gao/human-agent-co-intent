"""Follower-only projection of the triggering human instruction."""
from __future__ import annotations

from typing import Any, Mapping

# Per Follower agent: why its last route update was rejected (cleared once one succeeds).
LAST_UPDATE_ERROR_KEY = "agent_follower_last_update_error"


def record_route_update_result(session: dict[str, Any], agent_participant_id: str, results: Mapping[str, Any]) -> bool:
    """Remember a rejected route update for the next decision, or clear it after a success.

    Returns whether the record changed.
    """
    def updates(key: str) -> list[Mapping[str, Any]]:
        return [
            item for item in results.get(key) or []
            if isinstance(item, Mapping) and (item.get("action") or {}).get("type") == "send_follower_update"
        ]

    errors = dict(session.get(LAST_UPDATE_ERROR_KEY) or {})
    failed, succeeded = updates("failed"), updates("successful")
    if failed and not succeeded:
        errors[agent_participant_id] = {
            "error": str(failed[-1].get("error") or "unknown error"),
            "edit": {key: (failed[-1].get("action") or {}).get(key)
                     for key in ("keep_until", "add", "path_points") if key in (failed[-1].get("action") or {})},
        }
    elif succeeded and agent_participant_id in errors:
        del errors[agent_participant_id]
    else:
        return False
    session[LAST_UPDATE_ERROR_KEY] = errors
    return True


def last_route_update_failure_context(agent_participant_id: str, session: Mapping[str, Any]) -> str:
    failure = (session.get(LAST_UPDATE_ERROR_KEY) or {}).get(agent_participant_id)
    if not isinstance(failure, Mapping):
        return ""
    return (
        "<LAST ROUTE UPDATE FAILED>\n"
        f"error: {failure.get('error')}\n"
        f"rejected edit: {failure.get('edit') or failure.get('path_points')}\n"
        "</LAST ROUTE UPDATE FAILED>"
    )


def latest_visible_human_guide_message_context(
    agent_participant_id: str, session: Mapping[str, Any]
) -> str:
    """Render the newest human Guide message, including its server message ID.

    The ID is public routing metadata needed for the server to bind an execute
    decision to the exact instruction. Private messages not addressed to the
    Follower are excluded.
    """
    participants = session.get("participants") if isinstance(session, Mapping) else []
    by_id = {
        p.get("id") or p.get("participant_id"): p
        for p in participants or []
        if isinstance(p, Mapping) and (p.get("id") or p.get("participant_id"))
    }
    for message in reversed(session.get("messages") or []):
        if not isinstance(message, Mapping):
            continue
        sender = by_id.get(message.get("sender"))
        content = message.get("content")
        if (
            not isinstance(sender, Mapping)
            or str(sender.get("type") or "").lower() in {"ai", "ai_agent"}
            or str(sender.get("role") or "").lower() != "guide"
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        receiver = message.get("receiver")
        if receiver is None or receiver == agent_participant_id:
            message_id = message.get("id")
            if isinstance(message_id, str) and message_id.strip():
                return (
                    "<LATEST HUMAN GUIDE MESSAGE>\n"
                    f"id: {message_id}\n"
                    f"content: {content.strip()}\n"
                    "</LATEST HUMAN GUIDE MESSAGE>"
                )
    return ""


def follower_visibility_note(guide_sees_drawing: bool) -> str:
    """Tell the grounding Follower what evidence the Guide has of its work."""
    seen = (
        "The Guide can see your drawing."
        if guide_sees_drawing
        else "The Guide cannot see your map or your drawing; your messages are their only evidence of what you did."
    )
    return (
        "<VISIBILITY>\n"
        f"{seen}\n"
        "</VISIBILITY>"
    )
