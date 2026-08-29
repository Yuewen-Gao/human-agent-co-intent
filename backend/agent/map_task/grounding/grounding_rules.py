"""Deterministic constraints for the online grounding pipeline."""
from __future__ import annotations

H_CANDIDATES = {
    "H2": ("R7", "R2"), "H3": ("R4",), "H4": ("R4", "R6"),
    "H5": ("R4", "R6"), "H6": ("R2",), "H1": ("R6",),
}
REPAIR_TO_UPDATE = {"R4": "M3", "R6": "M2", "R7": "M4"}

def repair_candidates(h_code: str | None) -> tuple[str, ...]:
    return H_CANDIDATES.get(h_code or "", ())

def update_mode(repair_code: str | None, *, adds_missing_detail: bool = False) -> str:
    if repair_code == "R6" and adds_missing_detail:
        return "M1"
    return REPAIR_TO_UPDATE.get(repair_code or "", "M5")

def may_update_field(*, evidence_is_explicit: bool, proposed_field: str | None) -> str | None:
    return proposed_field if evidence_is_explicit and proposed_field else None
