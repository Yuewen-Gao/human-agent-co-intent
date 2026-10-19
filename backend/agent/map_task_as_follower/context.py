"""Follower-only projection of the triggering human instruction."""
from __future__ import annotations

from typing import Any, Mapping


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
