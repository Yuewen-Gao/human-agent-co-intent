"""Session-scoped context construction for the online Map Task agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.map_task_as_guide.map_processing.merge_follower_current_map import (
    build_follower_current_workspace_context as render_follower_workspace,
)
from config.map_sets import display_path, grid_path_for_map, session_grid_path

_DEFAULT_GUIDE_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "almanac_maps" / "map_guide.json"
)
_DEFAULT_FOLLOWER_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "almanac_maps" / "map.json"
)


def guide_grid_path(participant: Mapping[str, Any]) -> Path:
    """The Guide grid JSON for the map this participant was given (ALMANAC by default)."""
    params = participant.get("experiment_params") if isinstance(participant, Mapping) else None
    map_entry = params.get("map") if isinstance(params, Mapping) else None
    return grid_path_for_map(map_entry, "guide", _DEFAULT_GUIDE_GRID_PATH)


def follower_grid_path(session: Mapping[str, Any]) -> Path:
    """The Follower grid JSON for this session's Follower map (ALMANAC by default)."""
    return session_grid_path(session, "follower", _DEFAULT_FOLLOWER_GRID_PATH)


def _valid_cell(value: Any, rows: int, cols: int) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        row, col = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return (row, col) if 0 <= row < rows and 0 <= col < cols else None


def build_guide_map_grid_context(participant: Mapping[str, Any]) -> str:
    """Render the researcher-provided Guide ground-truth JSON as a text grid.

    The grid comes from the participant's map (see ``config/map_sets.py``).  It is deliberately
    independent of the displayed PNG so the route and landmark coordinates are
    available to the agent in a stable, inspectable format.
    """
    grid_path = guide_grid_path(participant)
    if not isinstance(participant, Mapping) or not grid_path.is_file():
        return ""

    try:
        map_data = json.loads(grid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    grid_size = map_data.get("grid_size")
    if not isinstance(grid_size, list) or len(grid_size) != 2:
        return ""
    try:
        rows, cols = int(grid_size[0]), int(grid_size[1])
    except (TypeError, ValueError):
        return ""
    if not (0 < rows <= 120 and 0 < cols <= 120):
        return ""

    grid = [["." for _ in range(cols)] for _ in range(rows)]
    landmarks = map_data.get("landmarks")
    landmark_lines: list[str] = []
    if isinstance(landmarks, Mapping):
        for name in sorted(landmarks):
            landmark = landmarks[name]
            if not isinstance(landmark, Mapping):
                continue
            cells = landmark.get("cells") or (
                landmark.get("summary", {}).get("boundary_cells", [])
                if isinstance(landmark.get("summary"), Mapping)
                else []
            )
            for raw_cell in cells if isinstance(cells, list) else []:
                cell = _valid_cell(raw_cell, rows, cols)
                if cell:
                    grid[cell[0]][cell[1]] = "#"
            summary = landmark.get("summary")
            centroid = summary.get("centroid") if isinstance(summary, Mapping) else None
            cell_count = len(cells) if isinstance(cells, list) else 0
            landmark_lines.append(
                f"- {name}: centroid={tuple(centroid) if isinstance(centroid, list) else 'unknown'}, cells={cell_count}"
            )

    route_cells = map_data.get("route_cells") or []
    ordered_route_cells = []
    for raw_cell in route_cells if isinstance(route_cells, list) else []:
        cell = _valid_cell(raw_cell, rows, cols)
        if cell:
            ordered_route_cells.append([cell[0], cell[1]])
            grid[cell[0]][cell[1]] = "*"

    start = _valid_cell(map_data.get("start_cell"), rows, cols)
    end = _valid_cell(map_data.get("end_cell"), rows, cols)
    if start:
        grid[start[0]][start[1]] = "S"
    if end:
        grid[end[0]][end[1]] = "F"

    map_rows = "\n".join("".join(row) for row in grid)
    landmarks_text = "\n".join(landmark_lines) or "- none"
    return (
        "<GUIDE MAP GRID>\n"
        f"source: {display_path(grid_path)}\n"
        f"grid_size: {rows} rows x {cols} columns\n"
        f"start: {start if start else 'unknown'}\n"
        f"end: {end if end else 'unknown'}\n"
        "legend: . = empty, # = landmark, * = target route, S = start, F = end\n"
        "coordinate_system: guide map only; do not use these coordinates as follower drawing coordinates\n"
        "ordered_route_cells: "
        f"{json.dumps(ordered_route_cells, separators=(',', ':'))}\n"
        "ordered_route_cells are start-to-end route evidence. Translate each active segment into "
        "Follower-local landmark relations before instructing the Follower.\n"
        "landmarks:\n"
        f"{landmarks_text}\n"
        "grid:\n"
        f"{map_rows}\n"
        "</GUIDE MAP GRID>"
    )


def _follower_progress(session: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for candidate in session.get("participants") or []:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("role") or "").strip().lower() != "follower":
            continue
        params = candidate.get("experiment_params")
        if not isinstance(params, Mapping):
            return None
        progress = params.get("map_progress")
        return progress if isinstance(progress, Mapping) else None
    return None


def follower_trajectory_grid_text(session: Mapping[str, Any]) -> str:
    """Return the follower's latest trajectory in the Follower grid system."""
    if not isinstance(session, Mapping):
        return ""
    progress = _follower_progress(session)
    if not progress:
        return ""

    precomputed_grid = progress.get("grid_text")
    if isinstance(precomputed_grid, str) and precomputed_grid.strip():
        grid_text = precomputed_grid.strip()
    else:
        canvas_data_url = progress.get("canvasDataUrl")
        if not isinstance(canvas_data_url, str) or not canvas_data_url.startswith("data:image/"):
            return ""
        # Pillow/NumPy are imported by map_formatting only when a live canvas
        # must be decoded, keeping normal session-memory imports lightweight.
        from agent.map_task_as_guide.map_processing.map_formatting import (
            route_grid_from_canvas_data_url,
        )

        grid_text = route_grid_from_canvas_data_url(canvas_data_url)
        if not grid_text:
            return ""

    return grid_text


def build_follower_current_workspace_context(session: Mapping[str, Any]) -> str:
    """Overlay the Follower map landmarks and latest trajectory for the agent.

    The static Follower map and the drawing grid must have the same ``grid_size``.
    A malformed or unavailable drawing is omitted rather than silently merged with
    different coordinates.
    """
    grid_path = follower_grid_path(session)
    if not isinstance(session, Mapping) or not grid_path.is_file():
        return ""
    try:
        follower_map = json.loads(grid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(follower_map, Mapping):
        return ""

    grid_size = follower_map.get("grid_size")
    if not isinstance(grid_size, list) or len(grid_size) != 2:
        return ""
    try:
        rows, cols = int(grid_size[0]), int(grid_size[1])
    except (TypeError, ValueError):
        return ""
    if not (0 < rows <= 120 and 0 < cols <= 120):
        return ""

    trajectory_grid = follower_trajectory_grid_text(session)
    if not trajectory_grid:
        trajectory_grid = "\n".join(["." * cols] * rows)

    try:
        workspace = render_follower_workspace(follower_map, trajectory_grid)
    except ValueError:
        return ""
    return workspace.replace(
        "<FOLLOWER CURRENT WORKSPACE>\n",
        f"<FOLLOWER CURRENT WORKSPACE>\nsource: {display_path(grid_path)}\n",
        1,
    )


GUIDE_BLIND_NOTE = (
    "<VISIBILITY>\n"
    "You cannot see the Follower's map or drawing in this session. You know only what the Follower tells you; "
    "their landmarks may be placed differently from yours, be missing, or go by another name. Do not assume they drew what you asked, "
    "and do not describe or judge their drawing beyond what they have told you.\n"
    "</VISIBILITY>"
)


def guide_sees_follower_drawing(session: Mapping[str, Any]) -> bool:
    """`Session.Interaction.guideSeesFollowerDrawing` when the session sets it, else the server default."""
    from config.grounding_treatment import GUIDE_SEES_FOLLOWER_DRAWING
    interaction = session.get("interaction") if isinstance(session, Mapping) else None
    value = interaction.get("guideSeesFollowerDrawing") if isinstance(interaction, Mapping) else None
    return value if isinstance(value, bool) else bool(GUIDE_SEES_FOLLOWER_DRAWING)


def append_maptask_grid_context(
    prompt: str,
    participant: Mapping[str, Any],
    session: Mapping[str, Any],
) -> str:
    """Append only the Map Task context this agent role is allowed to inspect."""
    role = str(participant.get("role") or "guide").strip().lower()
    if role == "guide" and not guide_sees_follower_drawing(session):
        sections = [build_guide_map_grid_context(participant), GUIDE_BLIND_NOTE]
    else:
        sections = [build_follower_current_workspace_context(session)]
        if role == "follower":
            from agent.map_task_as_follower.trajectory_snapshot import route_points_context
            sections.append(route_points_context(_follower_progress(session), follower_grid_path(session)))
        if role == "guide":
            sections.insert(0, build_guide_map_grid_context(participant))
            sections.append(build_follower_trajectory_change_context(session))
    content = "\n".join(section for section in sections if section)
    return f"{prompt}\n{content}\n" if content else prompt


def _trajectory_cells(grid_text: str) -> set[tuple[int, int]] | None:
    """Parse a normalized text grid without interpreting drawing event types."""
    if not isinstance(grid_text, str) or not grid_text.strip():
        return None
    lines = grid_text.strip().splitlines()
    width = len(lines[0]) if lines else 0
    if not width or any(len(line) != width or set(line) - {".", "x"} for line in lines):
        return None
    return {
        (row, col)
        for row, line in enumerate(lines)
        for col, value in enumerate(line)
        if value == "x"
    }


# Written by the grounding orchestrator when a Guide message is sent.
LAST_GUIDE_MESSAGE_KEY = "grounding_last_guide_message"


def _trajectory_grid_at_last_guide_message(session: Mapping[str, Any]) -> str | None:
    last = session.get(LAST_GUIDE_MESSAGE_KEY) if isinstance(session, Mapping) else None
    grid_text = last.get("trajectory_grid") if isinstance(last, Mapping) else None
    return grid_text if isinstance(grid_text, str) and grid_text.strip() else None


# The drawing and messages the last completed Guide judgement was made on.
LAST_JUDGED_KEY = "grounding_last_judged"

# A Guide message that needs an answer has had a real chance to get one once it
# has waited this long with no visible partner action since.
RESPONSE_WINDOW_SECONDS = 45.0


def _parse_time(value: Any) -> datetime | None:
    """ISO time as aware UTC ("Z" suffix or naive means UTC), or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def turn_changes(
    session: Mapping[str, Any], grid: str | None, messages_seen_until: str, now: datetime | None = None
) -> dict[str, Any]:
    """What is new since the last judged turn, and how long the Guide has waited.

    Only what the Guide can see counts: without the drawing, the partner's
    visible actions are their messages. The response window passes when neither
    side has acted for RESPONSE_WINDOW_SECONDS since the Guide's last message.
    `anything_new` is false when a turn would be judged on the same input as the last one.
    """
    now = now or datetime.now(timezone.utc)
    last = session.get(LAST_JUDGED_KEY) if isinstance(session, Mapping) else None
    last = last if isinstance(last, Mapping) else {}
    sees = guide_sees_follower_drawing(session)
    work_changed = sees and (
        (_trajectory_cells(grid or "") or set()) != (_trajectory_cells(last.get("trajectory_grid") or "") or set())
    )
    work_changed_at = now.isoformat() if work_changed else last.get("work_changed_at")
    new_messages = messages_seen_until != last.get("messages_seen_until", "")

    actions = [t for t in (_parse_time(messages_seen_until), _parse_time(work_changed_at) if sees else None) if t]
    partner_at = max(actions) if actions else None
    guide_message = session.get(LAST_GUIDE_MESSAGE_KEY) if isinstance(session, Mapping) else None
    guide_at = _parse_time(guide_message.get("timestamp")) if isinstance(guide_message, Mapping) else None
    since_guide = (now - guide_at).total_seconds() if guide_at else None
    # Timed from the last visible action of either side: silence after the partner's own reply counts too.
    last_action = max(guide_at, partner_at) if guide_at and partner_at else guide_at
    window_passed = bool(last_action and (now - last_action).total_seconds() >= RESPONSE_WINDOW_SECONDS)
    return {
        "new_partner_messages": new_messages,
        "partner_work_changed": ("yes" if work_changed else "no") if sees else "not_visible",
        "seconds_since_your_last_message": round(since_guide) if since_guide is not None else None,
        "seconds_since_partner_last_visible_action": round((now - partner_at).total_seconds()) if partner_at else None,
        "response_window_passed": window_passed,
        "anything_new": new_messages or work_changed or window_passed != bool(last.get("response_window_passed")),
        "work_changed_at": work_changed_at,
    }


def changes_block(changes: Mapping[str, Any]) -> str:
    def value(key: str) -> str:
        item = changes.get(key)
        return "none" if item is None else ("yes" if item is True else "no" if item is False else str(item))

    keys = ("new_partner_messages", "partner_work_changed", "seconds_since_your_last_message",
            "seconds_since_partner_last_visible_action", "response_window_passed")
    return "<CHANGES>\n" + "\n".join(f"{key}: {value(key)}" for key in keys) + "\n</CHANGES>"


def record_judged_turn(
    session: dict[str, Any], grid: str | None, messages_seen_until: str, changes: Mapping[str, Any] | None = None
) -> None:
    changes = changes or {}
    session[LAST_JUDGED_KEY] = {
        "trajectory_grid": (grid or "").strip(),
        "messages_seen_until": messages_seen_until,
        "work_changed_at": changes.get("work_changed_at"),
        "response_window_passed": bool(changes.get("response_window_passed")),
    }


def build_follower_trajectory_change_context(session: Mapping[str, Any]) -> str:
    """Render changed trajectory cells since the Guide's last sent message.

    The first Guide reply has no preceding snapshot, so its complete workspace
    is already sufficient and no artificial delta is emitted.
    """
    previous = _trajectory_grid_at_last_guide_message(session)
    current = follower_trajectory_grid_text(session)
    previous_cells = _trajectory_cells(previous) if previous else None
    current_cells = _trajectory_cells(current)
    if previous_cells is None or current_cells is None:
        return ""
    added = [list(cell) for cell in sorted(current_cells - previous_cells)]
    removed = [list(cell) for cell in sorted(previous_cells - current_cells)]
    return (
        "<FOLLOWER TRAJECTORY CHANGES SINCE LAST GUIDE MESSAGE>\n"
        f"added_cells: {json.dumps(added)}\n"
        f"removed_cells: {json.dumps(removed)}\n"
        "</FOLLOWER TRAJECTORY CHANGES SINCE LAST GUIDE MESSAGE>"
    )


def _visible_messages_and_names(
    agent_participant_id: str, session: Mapping[str, Any]
) -> tuple[dict[Any, str], list[Mapping[str, Any]]]:
    if not agent_participant_id or not isinstance(session, Mapping):
        return {}, []

    names = {
        participant.get("id"): participant.get("name")
        or participant.get("participant_name")
        or "Unknown"
        for participant in (session.get("participants") or [])
        if isinstance(participant, Mapping) and participant.get("id")
    }
    visible: list[Mapping[str, Any]] = []
    for message in session.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        sender = message.get("sender")
        receiver = message.get("receiver")
        if receiver is None or sender == agent_participant_id or receiver == agent_participant_id:
            visible.append(message)
    visible.sort(key=lambda item: str(item.get("timestamp") or ""))
    return names, visible


def _render_messages(names: Mapping[Any, str], messages: list[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"- {names.get(message.get('sender'), 'Unknown')}: {message['content'].strip()}"
        for message in messages
    )


def build_session_conversation_memory(
    agent_participant_id: str,
    session: Mapping[str, Any],
) -> str:
    """Return all messages visible to the agent in chronological order.

    Context is reconstructed from the persisted session on every LLM call, in
    the same spirit as CollabBench's per-turn trajectory reconstruction. Group
    messages and messages to/from the agent are visible; private exchanges
    between other participants are not.
    """
    names, visible_messages = _visible_messages_and_names(agent_participant_id, session)
    return _render_messages(names, visible_messages)


def latest_visible_message_time(agent_participant_id: str, session: Mapping[str, Any]) -> str:
    """Timestamp of the newest partner message the agent can see now ("" if none).

    The agent's own messages are left out: sending a reply is not new input.
    """
    _, visible_messages = _visible_messages_and_names(agent_participant_id, session)
    partner = [m for m in visible_messages if m.get("sender") != agent_participant_id]
    return str(partner[-1].get("timestamp") or "") if partner else ""


def build_new_visible_messages(
    agent_participant_id: str, session: Mapping[str, Any]
) -> str:
    """Return partner messages the Guide has not yet answered.

    When the last sent Guide message records which messages its turn had seen,
    everything the partner sent after that point is new, including messages
    that arrived while the turn was running and so landed before the reply.
    Only the server-owned Guide writes that record (Map Task has one AI role).
    """
    names, visible_messages = _visible_messages_and_names(agent_participant_id, session)
    last = session.get(LAST_GUIDE_MESSAGE_KEY) if isinstance(session, Mapping) else None
    seen_until = last.get("messages_seen_until") if isinstance(last, Mapping) else None
    if isinstance(seen_until, str):
        return _render_messages(names, [
            message for message in visible_messages
            if message.get("sender") != agent_participant_id and str(message.get("timestamp") or "") > seen_until
        ])
    last_guide_index = max(
        (index for index, message in enumerate(visible_messages) if message.get("sender") == agent_participant_id),
        default=-1,
    )
    return _render_messages(names, visible_messages[last_guide_index + 1 :])


def append_session_conversation_memory(
    prompt: str,
    agent_participant_id: str,
    session: Mapping[str, Any],
) -> str:
    """Append reconstructed visible conversation history to an agent prompt."""
    memory = build_session_conversation_memory(agent_participant_id, session)
    new_messages = build_new_visible_messages(agent_participant_id, session)
    result = f"{prompt}\n<SESSION CONVERSATION MEMORY>\n{memory or 'None'}\n</SESSION CONVERSATION MEMORY>\n"
    result += f"<NEW VISIBLE MESSAGES>\n{new_messages or 'None'}\n</NEW VISIBLE MESSAGES>\n"
    return result
