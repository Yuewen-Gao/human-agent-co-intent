"""Small deterministic labels retained for grounding-decision records."""
from __future__ import annotations

from typing import Iterable

from .fewshot_repository import repair_association_counts


REPAIR_TO_UPDATE = {"R4": "M3", "R6": "M2", "R7": "M4"}


def associated_repair_codes(
    h_code: str | None, w_codes: Iterable[str] | None
) -> tuple[str, ...]:
    """Rank R labels only to choose historical examples for the planner.

    The ranking remains outside the prompt: the LLM receives selected examples
    but not counts, probabilities, or a prescribed R choice.
    """
    observed = repair_association_counts()
    ranked: dict[str, dict[str, int]] = {}

    def add(r_code: str, priority: int, support: int) -> None:
        entry = ranked.setdefault(r_code, {"priority": priority, "support": 0})
        entry["priority"] = min(entry["priority"], priority)
        entry["support"] += support

    h = h_code or ""
    selected_w = tuple(dict.fromkeys(w_codes or ()))
    for w_code in selected_w:
        for (observed_h, observed_w, r_code), support in observed["H+W"].items():
            if (observed_h, observed_w) == (h, w_code):
                add(r_code, 1, support)
    for (observed_h, r_code), support in observed["H"].items():
        if observed_h == h:
            add(r_code, 2, support)
    for w_code in selected_w:
        for (observed_w, r_code), support in observed["W"].items():
            if observed_w == w_code:
                add(r_code, 3, support)

    return tuple(
        r_code for r_code, _ in sorted(
            ranked.items(), key=lambda item: (item[1]["priority"], -item[1]["support"], item[0])
        )
    )


def update_mode(repair_code: str | None, *, adds_missing_detail: bool = False) -> str:
    if repair_code == "R6" and adds_missing_detail:
        return "M1"
    return REPAIR_TO_UPDATE.get(repair_code or "", "M5")
