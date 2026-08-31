"""Load the canonical, prompt-safe grounding codebook definitions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re


_CODEBOOK_DIR = (
    Path(__file__).resolve().parents[2] / "prompts" / "mapTaskGuidePrompts" / "codebookDefinition"
)


@lru_cache(maxsize=None)
def definitions_for(family: str) -> str:
    """Return one code family without references or bibliographic metadata."""
    normalized = str(family or "").strip().upper()
    if normalized not in {"H", "W", "R", "M", "O", "C"}:
        raise ValueError(f"unknown codebook family: {family}")
    return (_CODEBOOK_DIR / f"{normalized}_definitions.txt").read_text(encoding="utf-8").strip()


def definitions_for_codes(family: str, codes: tuple[str, ...]) -> str:
    """Return complete definition paragraphs for the selected code(s) only."""
    normalized = str(family or "").strip().upper()
    requested = tuple(dict.fromkeys(
        code for code in codes if isinstance(code, str) and re.fullmatch(rf"{normalized}\d+", code)
    ))
    if not requested:
        return ""
    blocks = re.split(r"(?=^[A-Z]\d+\.\s)", definitions_for(normalized), flags=re.MULTILINE)
    selected = []
    for code in requested:
        prefix = f"{code}. "
        block = next((item.strip() for item in blocks if item.startswith(prefix)), "")
        if block:
            selected.append(block)
    return "\n\n".join(selected)
