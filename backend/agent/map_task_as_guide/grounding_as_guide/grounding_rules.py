"""Small deterministic labels retained for grounding-decision records."""
from __future__ import annotations


REPAIR_TO_UPDATE = {"R4": "M3", "R6": "M2", "R7": "M4"}


def update_mode(repair_code: str | None, *, adds_missing_detail: bool = False) -> str:
    if repair_code == "R6" and adds_missing_detail:
        return "M1"
    return REPAIR_TO_UPDATE.get(repair_code or "", "M5")
