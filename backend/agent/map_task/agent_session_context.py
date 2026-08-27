"""Session-scoped context construction for the online Map Task agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_GUIDE_GRID_PATH = (
    Path(__file__).resolve().parents[2] / "almanac_maps" / "map_guide.json"
)


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

    The default preset is ``backend/almanac_maps/map_guide.json``.  It is deliberately
    independent of the displayed PNG so the route and landmark coordinates are
    available to the agent in a stable, inspectable format.
    """
    if not isinstance(participant, Mapping) or not _DEFAULT_GUIDE_GRID_PATH.is_file():
        return ""

    try:
        map_data = json.loads(_DEFAULT_GUIDE_GRID_PATH.read_text(encoding="utf-8"))
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
    for raw_cell in route_cells if isinstance(route_cells, list) else []:
        cell = _valid_cell(raw_cell, rows, cols)
        if cell:
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
        "source: backend/almanac_maps/map_guide.json\n"
        f"grid_size: {rows} rows x {cols} columns\n"
        f"start: {start if start else 'unknown'}\n"
        f"end: {end if end else 'unknown'}\n"
        "legend: . = empty, # = landmark, * = target route, S = start, F = end\n"
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


def build_follower_drawing_grid_context(session: Mapping[str, Any]) -> str:
    """Return the follower's latest drawing as an occupancy grid, when available."""
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
        from agent.map_task.map_formatting import route_grid_from_canvas_data_url

        grid_text = route_grid_from_canvas_data_url(canvas_data_url)
        if not grid_text:
            return ""

    return (
        "<FOLLOWER CURRENT DRAWING GRID>\n"
        "legend: x = a follower brush cell, . = no follower brush cell\n"
        f"{grid_text}\n"
        "</FOLLOWER CURRENT DRAWING GRID>"
    )


def append_maptask_grid_context(
    prompt: str,
    participant: Mapping[str, Any],
    session: Mapping[str, Any],
) -> str:
    """Append ground-truth and current drawing grids to a Map Task prompt."""
    sections = [
        build_guide_map_grid_context(participant),
        build_follower_drawing_grid_context(session),
    ]
    content = "\n".join(section for section in sections if section)
    return f"{prompt}\n{content}\n" if content else prompt


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
    if not agent_participant_id or not isinstance(session, Mapping):
        return ""

    names = {
        participant.get("id"): participant.get("name")
        or participant.get("participant_name")
        or "Unknown"
        for participant in (session.get("participants") or [])
        if isinstance(participant, Mapping) and participant.get("id")
    }

    visible_messages: list[Mapping[str, Any]] = []
    for message in session.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        sender = message.get("sender")
        receiver = message.get("receiver")
        if receiver is None or sender == agent_participant_id or receiver == agent_participant_id:
            visible_messages.append(message)

    visible_messages.sort(key=lambda item: str(item.get("timestamp") or ""))
    return "\n".join(
        f"- {names.get(message.get('sender'), 'Unknown')}: {message['content'].strip()}"
        for message in visible_messages
    )


def append_session_conversation_memory(
    prompt: str,
    agent_participant_id: str,
    session: Mapping[str, Any],
) -> str:
    """Append reconstructed visible conversation history to an agent prompt."""
    memory = build_session_conversation_memory(agent_participant_id, session)
    if not memory:
        return prompt
    return f"{prompt}\n<SESSION CONVERSATION MEMORY>\n{memory}\n</SESSION CONVERSATION MEMORY>\n"
