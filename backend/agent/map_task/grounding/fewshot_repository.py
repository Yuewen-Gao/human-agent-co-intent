"""Static approved few-shot retrieval; never invokes offline analysis at runtime."""
from __future__ import annotations
import csv
import re
from pathlib import Path

_CSV = Path(__file__).resolve().parents[2] / "prompts" / "grounding_fewshots.csv"

def examples_for(stage: str, *, h_code: str | None = None, limit: int = 3) -> list[dict[str, str]]:
    if not _CSV.is_file(): return []
    with _CSV.open(encoding="utf-8", newline="") as handle:
        rows=[row for row in csv.DictReader(handle) if row.get("status") == "candidate"]
    family = "H" if stage == "h" else "R" if stage == "repair" else ""
    rows=[row for row in rows if row.get("family") == family]
    if h_code and stage == "repair":
        matching=[row for row in rows if re.search(rf"(?:^|[=;,\s]){re.escape(h_code)}(?:$|[;,\s])", row.get("guide_labels", ""))]
        # Rare H labels may have no reviewed R example; retain generic reviewed
        # R examples rather than silently sending an empty repair prompt.
        rows=matching or rows
    return rows[:limit]
