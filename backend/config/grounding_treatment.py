"""One server-owned switch for the Map Task grounding treatment.

This controls the participant-visible Shared Mental Model, the H/W/R grounding
pipeline, and corresponding mental-model updates. The public function returns
only a read-only capability flag for clients.
"""
from __future__ import annotations


ENABLE_GROUNDING_TREATMENT = True
MAPTASK_AGENT_ROLE = "guide"



def grounding_treatment_enabled() -> bool:
    return bool(ENABLE_GROUNDING_TREATMENT)


def maptask_agent_role() -> str:
    """Return the only supported server-owned AI role for a Map Task session."""
    role = str(MAPTASK_AGENT_ROLE).strip().lower()
    return role if role in {"guide", "follower"} else "guide"


def public_feature_flags() -> dict[str, bool]:
    return {"grounding_treatment_enabled": grounding_treatment_enabled()}
