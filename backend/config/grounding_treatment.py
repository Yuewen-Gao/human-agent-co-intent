"""One server-owned switch for the Map Task grounding treatment.

This controls the participant-visible Shared Mental Model, the H/W/R grounding
pipeline, and corresponding mental-model updates. The public function returns
only a read-only capability flag for clients.
"""
from __future__ import annotations


ENABLE_GROUNDING_TREATMENT = True



def grounding_treatment_enabled() -> bool:
    return bool(ENABLE_GROUNDING_TREATMENT)


def public_feature_flags() -> dict[str, bool]:
    return {"grounding_treatment_enabled": grounding_treatment_enabled()}
