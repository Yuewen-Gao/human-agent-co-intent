"""Session-room status events for visible Map Task AI replies."""
from __future__ import annotations

from typing import Any


VISIBLE_AGENT_REPLY_TRIGGERS = frozenset({
    "partner_message",
})


def should_publish_agent_reply_status(
    *,
    experiment_type: str | None,
    participant_role: str | None,
    trigger_context: str | None,
    partner_activity: bool = False,
) -> bool:
    """Return whether this turn answers the partner, so the room is told the agent is replying.

    A partner message wakes a reply; for the Guide, so does a scheduled check that finds new
    partner work (a drawing change it can see). A check that finds nothing new is announced
    only once it decides to speak (see `AgentRunner._perceive_and_act`).
    """
    if str(experiment_type or "").lower() != "maptask":
        return False
    role, trigger = str(participant_role or "").lower(), str(trigger_context or "")
    if trigger in VISIBLE_AGENT_REPLY_TRIGGERS:
        return role in ("guide", "follower")
    return role == "guide" and trigger == "periodic_check" and partner_activity


def emit_agent_reply_status(
    socketio: Any | None = None,
    *,
    session_id: str,
    agent_participant_id: str,
    active: bool,
    trigger_context: str,
) -> None:
    """Broadcast a reply-status transition to everyone in one session room."""
    if socketio is None:
        from websocket.handlers import get_socketio

        socketio = get_socketio()
    socketio.emit(
        "agent_reply_status",
        {
            "session_id": session_id,
            "agent_participant_id": agent_participant_id,
            "active": bool(active),
            "trigger_context": trigger_context,
        },
        room=session_id,
    )
