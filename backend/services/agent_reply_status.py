"""Session-room status events for visible Map Task Guide replies."""
from __future__ import annotations

from typing import Any


VISIBLE_AGENT_REPLY_TRIGGERS = frozenset({
    "follower_message",
    "follower_trajectory_changed",
})


def should_publish_agent_reply_status(
    *,
    experiment_type: str | None,
    participant_role: str | None,
    trigger_context: str | None,
) -> bool:
    """Return whether this request represents a user-visible Guide reply."""
    return (
        str(experiment_type or "").lower() == "maptask"
        and str(participant_role or "").lower() == "guide"
        and str(trigger_context or "") in VISIBLE_AGENT_REPLY_TRIGGERS
    )


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
