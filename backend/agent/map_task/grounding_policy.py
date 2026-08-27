"""Optional c2 Guide grounding-policy prompt injection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "grounding_policy_as_c2_guide.txt"
)

# Change this flag directly to enable the c2 grounding-treatment prompt.
ENABLE_GROUNDING_POLICY = False


def append_grounding_policy_if_enabled(
    prompt: str,
    participant: Mapping[str, Any],
) -> str:
    """Append the c2 policy only for a Guide when the module flag is enabled."""
    if not isinstance(participant, Mapping):
        return prompt
    if str(participant.get("role") or "").strip().lower() != "guide":
        return prompt
    if not ENABLE_GROUNDING_POLICY or not _POLICY_PATH.is_file():
        return prompt
    try:
        policy = _POLICY_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return prompt
    return f"{prompt}\n\n{policy}\n" if policy else prompt
