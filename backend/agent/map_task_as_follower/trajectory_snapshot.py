"""Server-owned Map Task Follower trajectory validation and chat snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_FOLLOWER_MAP_PATH = Path(__file__).resolve().parents[2] / "almanac_maps" / "map.json"


def _map_data(map_path: Path | None = None) -> dict[str, Any]:
    """The Follower map grid: the session's own map when given, else the ALMANAC default."""
    try:
        data = json.loads((map_path or _FOLLOWER_MAP_PATH).read_text(encoding="utf-8"))
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


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells on the straight line from start to end, both included (Bresenham)."""
    (row, col), (end_row, end_col) = start, end
    d_row, d_col = abs(end_row - row), -abs(end_col - col)
    step_row, step_col = (1 if end_row > row else -1), (1 if end_col > col else -1)
    error = d_row + d_col
    cells = [(row, col)]
    while (row, col) != (end_row, end_col):
        twice = 2 * error
        if twice >= d_col:
            error += d_col
            row += step_row
        if twice <= d_row:
            error += d_row
            col += step_col
        cells.append((row, col))
    return cells


def _ordered_points(value: Any, rows: int, cols: int) -> list[tuple[int, int]]:
    """The drawn path in order: given points joined by straight lines.

    Points may be sparse corner points; the cells between consecutive points are
    filled in. The path may pass a cell more than once (it then appears once).
    """
    if not isinstance(value, list) or not value:
        raise ValueError("Follower path_points must be a non-empty list")
    corners: list[tuple[int, int]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("Follower path point must be [row, col]")
        try:
            point = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError) as error:
            raise ValueError("Follower path point must be [row, col]") from error
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            raise ValueError(f"Follower path point {list(point)} is outside the {rows}x{cols} map")
        corners.append(point)
    path = corners[:1] + [cell for a, b in zip(corners, corners[1:]) for cell in _line_cells(a, b)[1:]]
    return list(dict.fromkeys(path))


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


def ensure_dashboard_snapshot(current_progress: Mapping[str, Any] | None, map_path: Path | None = None) -> dict[str, Any]:
    """Add a dashboard overlay to persisted pre-dashboard AI Follower progress.

    Historical cells become the cumulative route only. No old route is labelled
    as newly added during the current turn.
    """
    map_data = _map_data(map_path)
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
    map_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate a complete replacement route and return progress plus a text snapshot.

    The caller owns persistence. This function is pure with respect to session state,
    so a failed validation cannot partially mutate a participant or chat history.
    """
    map_data = _map_data(map_path)
    rows, cols = map_data["_rows"], map_data["_cols"]
    progress = dict(current_progress) if isinstance(current_progress, Mapping) else {}
    existing = _points_from_grid(progress, rows, cols)
    points = _ordered_points(path_points, rows, cols)
    updated, snapshot = _with_route(progress, map_data, existing, set(points), points,
                                    [int(path_points[-1][0]), int(path_points[-1][1])])
    updated["route_points"] = [[int(p[0]), int(p[1])] for p in path_points]
    return updated, snapshot


def apply_route_edit(
    current_progress: Mapping[str, Any] | None,
    keep_until: Any,
    add: Any,
    map_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Edit the Follower's route: keep its first `keep_until` points, then add new points.

    The route is stored as ordered points (`route_points`), and the drawing is
    those points joined by straight lines. Keeping all points and adding extends
    the route; keeping fewer removes its end; keeping fewer and adding redraws
    the end. The kept part is never re-drawn, so it cannot drift.
    """
    map_data = _map_data(map_path)
    rows, cols = map_data["_rows"], map_data["_cols"]
    progress = dict(current_progress) if isinstance(current_progress, Mapping) else {}
    route = [list(p) for p in progress.get("route_points") or []]
    if isinstance(keep_until, bool) or not isinstance(keep_until, int) or not 0 <= keep_until <= len(route):
        raise ValueError(f"keep_until must be a point number from 0 to {len(route)}")
    if not isinstance(add, list):
        raise ValueError("add must be a list of [row, col] points")
    if not add and keep_until == len(route):
        raise ValueError("The edit changes nothing: keep fewer points or add new ones")
    new_route = route[:keep_until] + [list(p) for p in add]
    cells = _ordered_points(new_route, rows, cols) if new_route else []
    existing = _points_from_grid(progress, rows, cols)
    updated, snapshot = _with_route(progress, map_data, existing, set(cells), cells,
                                    [int(v) for v in new_route[-1]] if new_route else None)
    updated["route_points"] = [[int(r), int(c)] for r, c in new_route]
    return updated, snapshot


def _with_route(
    progress: dict[str, Any], map_data: Mapping[str, Any], existing: set[tuple[int, int]],
    all_cells: set[tuple[int, int]], points: list[tuple[int, int]], endpoint: list[int] | None,
) -> tuple[dict[str, Any], str]:
    rows, cols = map_data["_rows"], map_data["_cols"]
    added_cells = [point for point in points if point not in existing]
    grid_text = "\n".join(
        "".join("x" if (row, col) in all_cells else "." for col in range(cols))
        for row in range(rows)
    )
    updated = {
        **progress,
        "grid_text": grid_text,
        "filledCells": [f"{row},{col}" for row, col in sorted(all_cells)],
        # The last given point, even when the path ends on a cell it passed before.
        "agent_follower_last_endpoint": endpoint,
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


def _relation(point: tuple[int, int], map_data: Mapping[str, Any]) -> str:
    """Where a point lies relative to the nearest landmark on the Follower map."""
    start = map_data.get("start_cell")
    if isinstance(start, list) and len(start) == 2 and max(abs(point[0] - start[0]), abs(point[1] - start[1])) <= 1:
        return "at START"
    best = None
    for name, landmark in (map_data.get("landmarks") or {}).items():
        box = ((landmark or {}).get("summary") or {}).get("bbox")
        if not isinstance(box, Mapping):
            continue
        d_row = box["row_min"] - point[0] if point[0] < box["row_min"] else max(0, point[0] - box["row_max"])
        d_col = box["col_min"] - point[1] if point[1] < box["col_min"] else max(0, point[1] - box["col_max"])
        distance = max(d_row, d_col)
        if best is None or distance < best[0]:
            vertical = "above" if point[0] < box["row_min"] else "below" if point[0] > box["row_max"] else ""
            horizontal = "left" if point[1] < box["col_min"] else "right" if point[1] > box["col_max"] else ""
            if vertical and horizontal:
                where = f"{vertical} and {horizontal} of the {name}"
            elif vertical:
                where = f"{vertical} the {name}"
            elif horizontal:
                where = f"{horizontal} of the {name}"
            else:
                where = f"on the {name}"
            best = (distance, where)
    if best is None:
        return ""
    return best[1] + (f", {best[0]} cells away" if best[0] > 1 else "")


def route_points_context(progress: Mapping[str, Any] | None, map_path: Path | None = None) -> str:
    """The Follower's own route as numbered points, each placed relative to the nearest landmark."""
    route = (progress or {}).get("route_points") if isinstance(progress, Mapping) else None
    try:
        map_data = _map_data(map_path)
    except ValueError:
        return ""
    lines = ["<YOUR CURRENT ROUTE>",
             "Your drawing is these points in order, joined by straight lines (relations computed on your map):"]
    if not route:
        lines.append("- none yet (keep_until 0)")
    for number, point in enumerate(route or [], 1):
        label = " (end)" if number == len(route) else ""
        lines.append(f"{number}. {list(point)} {_relation(tuple(point), map_data)}{label}")
    lines.append("</YOUR CURRENT ROUTE>")
    return "\n".join(lines)
