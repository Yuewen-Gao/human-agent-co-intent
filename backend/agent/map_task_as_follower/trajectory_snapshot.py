"""Server-owned Map Task Follower trajectory validation and chat snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_FOLLOWER_MAP_PATH = Path(__file__).resolve().parents[2] / "almanac_maps" / "map.json"


def _map_data() -> dict[str, Any]:
    try:
        data = json.loads(_FOLLOWER_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Follower map definition is unavailable") from error
    if not isinstance(data, dict):
        raise ValueError("Follower map definition is invalid")
    size = data.get("grid_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Follower map has no valid grid size")
    try:
        rows, cols = int(size[0]), int(size[1])
    except (TypeError, ValueError) as error:
        raise ValueError("Follower map grid size is invalid") from error
    if rows <= 0 or cols <= 0:
        raise ValueError("Follower map grid size is invalid")
    data["_rows"] = rows
    data["_cols"] = cols
    return data


def _points_from_grid(progress: Mapping[str, Any], rows: int, cols: int) -> set[tuple[int, int]]:
    grid = progress.get("grid_text")
    if isinstance(grid, str) and grid.strip():
        lines = grid.strip().splitlines()
        if len(lines) != rows or any(len(line) != cols or set(line) - {".", "x"} for line in lines):
            raise ValueError("Stored follower trajectory grid is invalid")
        return {
            (row, col)
            for row, line in enumerate(lines)
            for col, value in enumerate(line)
            if value == "x"
        }

    cells: set[tuple[int, int]] = set()
    raw = progress.get("filledCells")
    if raw is None:
        return cells
    if not isinstance(raw, list):
        raise ValueError("Stored follower cells are invalid")
    for value in raw:
        if not isinstance(value, str) or "," not in value:
            raise ValueError("Stored follower cells are invalid")
        raw_row, raw_col = value.split(",", 1)
        try:
            row, col = int(raw_row), int(raw_col)
        except ValueError as error:
            raise ValueError("Stored follower cells are invalid") from error
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("Stored follower cells are outside the map")
        cells.add((row, col))
    return cells


def _ordered_points(value: Any, rows: int, cols: int) -> list[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Follower path_points must be a non-empty list")
    result: list[tuple[int, int]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("Follower path point must be [row, col]")
        try:
            point = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError) as error:
            raise ValueError("Follower path point must be [row, col]") from error
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            raise ValueError("Follower path point is outside the map")
        if not result or point != result[-1]:
            result.append(point)
    if len(set(result)) != len(result):
        raise ValueError("Follower path cannot revisit a cell")
    if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1 for a, b in zip(result, result[1:])):
        raise ValueError("Follower path points must be 8-neighbor continuous")
    return result


def _landmark_cells(map_data: Mapping[str, Any], rows: int, cols: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    landmarks = map_data.get("landmarks")
    if not isinstance(landmarks, Mapping):
        return cells
    for landmark in landmarks.values():
        if not isinstance(landmark, Mapping):
            continue
        raw_cells = landmark.get("cells")
        if not isinstance(raw_cells, list):
            continue
        for raw in raw_cells:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                continue
            try:
                row, col = int(raw[0]), int(raw[1])
            except (TypeError, ValueError):
                continue
            if 0 <= row < rows and 0 <= col < cols:
                cells.add((row, col))
    return cells


def _dashboard_snapshot(
    map_data: Mapping[str, Any], route_cells: set[tuple[int, int]], new_cells: list[tuple[int, int]]
) -> dict[str, list[list[int]]]:
    """Return the public, full-map data used only by the Guide Dashboard."""
    rows, cols = int(map_data["_rows"]), int(map_data["_cols"])
    return {
        "grid_size": [rows, cols],
        "landmark_cells": [list(cell) for cell in sorted(_landmark_cells(map_data, rows, cols))],
        "route_cells": [list(cell) for cell in sorted(route_cells)],
        "new_cells": [list(cell) for cell in new_cells],
    }


def ensure_dashboard_snapshot(current_progress: Mapping[str, Any] | None) -> dict[str, Any]:
    """Add a dashboard overlay to persisted pre-dashboard AI Follower progress.

    Historical cells become the cumulative route only. No old route is labelled
    as newly added during the current turn.
    """
    map_data = _map_data()
    rows, cols = map_data["_rows"], map_data["_cols"]
    progress = dict(current_progress) if isinstance(current_progress, Mapping) else {}
    snapshot = progress.get("agent_follower_dashboard_snapshot")
    if isinstance(snapshot, Mapping):
        return progress
    route_cells = _points_from_grid(progress, rows, cols)
    progress["agent_follower_dashboard_snapshot"] = _dashboard_snapshot(map_data, route_cells, [])
    return progress


def apply_route_update(
    current_progress: Mapping[str, Any] | None,
    path_points: Any,
) -> tuple[dict[str, Any], str]:
    """Validate a complete replacement route and return progress plus a text snapshot.

    The caller owns persistence. This function is pure with respect to session state,
    so a failed validation cannot partially mutate a participant or chat history.
    """
    map_data = _map_data()
    rows, cols = map_data["_rows"], map_data["_cols"]
    progress = dict(current_progress) if isinstance(current_progress, Mapping) else {}
    existing = _points_from_grid(progress, rows, cols)
    points = _ordered_points(path_points, rows, cols)

    # `path_points` is the Agent's complete current route, not an append-only
    # segment. Replacing the stored cells lets the Agent shorten or redraw a
    # route simply by omitting the cells that should no longer be present.
    all_cells = set(points)
    added_cells = [point for point in points if point not in existing]
    grid_text = "\n".join(
        "".join("x" if (row, col) in all_cells else "." for col in range(cols))
        for row in range(rows)
    )
    updated = {
        **progress,
        "grid_text": grid_text,
        "filledCells": [f"{row},{col}" for row, col in sorted(all_cells)],
        "agent_follower_last_endpoint": [points[-1][0], points[-1][1]],
        "agent_follower_dashboard_snapshot": _dashboard_snapshot(map_data, all_cells, added_cells),
    }

    display = [["." for _ in range(cols)] for _ in range(rows)]
    for row, col in _landmark_cells(map_data, rows, cols):
        display[row][col] = "#"
    for row, col in existing & all_cells:
        display[row][col] = "*"
    for row, col in added_cells:
        display[row][col] = "+"
    snapshot = (
        "<AGENT FOLLOWER MAP SNAPSHOT>\n"
        "legend: . = empty, # = landmark, * = retained AI Follower trajectory, + = added this turn\n"
        + "\n".join("".join(row) for row in display)
        + "\n</AGENT FOLLOWER MAP SNAPSHOT>"
    )
    return updated, snapshot
