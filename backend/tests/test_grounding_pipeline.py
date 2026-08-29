from agent.map_task.grounding.grounding_rules import may_update_field, repair_candidates, update_mode
from agent.map_task.grounding.contract import parse_repair

def test_h_candidates_and_repair_to_update_are_deterministic():
    assert repair_candidates("H3") == ("R4",)
    assert update_mode("R4") == "M3"
    assert update_mode("R6", adds_missing_detail=True) == "M1"

def test_uw_requires_explicit_evidence_and_allowed_repair():
    assert may_update_field(evidence_is_explicit=False, proposed_field="W3") is None
    assert parse_repair('{"r_code":"R6","uw_code":"W3"}', ("R4",)) == {}
