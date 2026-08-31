"""Load the canonical, prompt-safe grounding codebook definitions."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


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
