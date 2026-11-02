"""Map Task map sets: each role's displayed image paired with its grid JSON.

New public sessions use ACTIVE_MAP_SET. An existing session keeps its own maps:
its grid JSON is found from the image filename it was created with, and an
unknown filename (for example an uploaded map) falls back to ALMANAC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_BACKEND = Path(__file__).resolve().parents[1]

MAP_SETS = {
    "ALMANAC": {
        "guide": ("ALMANAC_guide_map.png", _BACKEND / "almanac_maps" / "map_guide.json"),
        "follower": ("ALMANAC_follower_map.png", _BACKEND / "almanac_maps" / "map.json"),
    },
    # Placeholder pictures for the HCRC-style map H1 (route v3); replace the files to use the final drawings.
    "H1": {
        "guide": ("H1_guide_map.png", _BACKEND / "map_grids" / "H1_guide.json"),
        "follower": ("H1_follower_map.png", _BACKEND / "map_grids" / "H1_follower.json"),
    },
    # The user's drawn pair (logs/sim/maps_user_v5): the Follower has a second suspension bridge, a forest where the
    # Guide has the windmill and END, and the pyramid at the top left. Grids from make_map_user_v5.py.
    "H2": {
        "guide": ("H2_guide_map.png", _BACKEND / "map_grids" / "H2_guide.json"),
        "follower": ("H2_follower_map.png", _BACKEND / "map_grids" / "H2_follower.json"),
    },
}
ACTIVE_MAP_SET = "H2"
_FALLBACK = "ALMANAC"


def active_map_files() -> list[tuple[str, str]]:
    """(role, image filename) of the active set, Follower first."""
    maps = MAP_SETS[ACTIVE_MAP_SET]
    return [("follower", maps["follower"][0]), ("guide", maps["guide"][0])]


def _filename(map_entry: Any) -> str:
    if not isinstance(map_entry, Mapping):
        return ""
    name = str(map_entry.get("filename") or "").strip()
    if not name:
        name = str(map_entry.get("file_path") or "").strip().split("?")[0].rstrip("/").split("/")[-1]
    return name


def grid_path_for_map(map_entry: Any, role: str, default: Path | None = None) -> Path:
    """Grid JSON for one displayed map; `default` (else ALMANAC) when the image is not a known preset."""
    name = _filename(map_entry)
    for maps in MAP_SETS.values():
        if maps[role][0] == name:
            return maps[role][1]
    return default or MAP_SETS[_FALLBACK][role][1]


def session_map(session: Any, role: str) -> Mapping[str, Any] | None:
    """The map a session shows to `role`: its params.maps entry, else that participant's map."""
    if not isinstance(session, Mapping):
        return None
    params = session.get("params")
    for entry in (params.get("maps") if isinstance(params, Mapping) else None) or ():
        if isinstance(entry, Mapping) and str(entry.get("role") or "").lower() == role:
            return entry
    for participant in session.get("participants") or ():
        if isinstance(participant, Mapping) and str(participant.get("role") or "").lower() == role:
            experiment_params = participant.get("experiment_params")
            if isinstance(experiment_params, Mapping) and isinstance(experiment_params.get("map"), Mapping):
                return experiment_params["map"]
    return None


def session_grid_path(session: Any, role: str, default: Path | None = None) -> Path:
    return grid_path_for_map(session_map(session, role), role, default)


def display_path(path: Path) -> str:
    """Repository-style path for prompt labels (backend/...)."""
    try:
        return "backend/" + path.relative_to(_BACKEND).as_posix()
    except ValueError:
        return path.as_posix()
