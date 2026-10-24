from fractions import Fraction as F
import pytest
from temporal_guard import TemporalGuard
def base(low=9900,high=10300,value=10100,reason="candidate_inside_reachable_budget"):
 return dict(selection="candidate",reason=reason,research_speed_mmps=value,research_speed_mps=value/1000,
 candidate_status="authenticated",anchor_status="authenticated",command_status="authenticated",risk_context="normal",
 low_n=low,low_d=1,high_n=high,high_d=1,conditional_budget_satisfied=True)
def test_nominal_valid_output_is_preserved():
 g=TemporalGuard();d=base();out=g.evaluate(d,1100000)
 assert out["research_speed_mmps"]==10100 and out["selection"]=="candidate"
 assert d==base()
def test_abrupt_joint_offset_latches_abstention():
 g=TemporalGuard();g.evaluate(base(),1100000)
 out=g.evaluate(base(7000,7400,7200),1200000)
 assert out["research_speed_mmps"] is None and out["temporal_latched"]
def test_honest_return_cannot_reset_latch():
 g=TemporalGuard();g.evaluate(base(1000,1400,1200),1100000)
 assert g.evaluate(base(),1200000)["research_speed_mmps"] is None
def test_first_false_report_cannot_override_trusted_initial_set():
 assert TemporalGuard().evaluate(base(7000,7400,7200),1100000)["temporal_latched"]
def test_missing_anchor_preserves_abstention_without_latching():
 d=base();d.update(selection="abstain",reason="anchor_missing",research_speed_mmps=None,research_speed_mps=None,
  anchor_status="missing",low_n=None,low_d=None,high_n=None,high_d=None,conditional_budget_satisfied=False)
 out=TemporalGuard().evaluate(d,1100000)
 assert out["selection"]=="abstain" and not out["temporal_latched"]
def test_lockdown_cannot_become_selection():
 d=base();d.update(selection="abstain",reason="lockdown_context",risk_context="lockdown",research_speed_mmps=None,research_speed_mps=None)
 out=TemporalGuard().evaluate(d,1100000)
 assert out["selection"]=="abstain" and out["reason"]=="lockdown_context"
def test_original_budget_abstention_is_never_overridden():
 d=base(reason="budget_exhausted");d.update(selection="abstain",research_speed_mmps=None,research_speed_mps=None,conditional_budget_satisfied=False)
 assert TemporalGuard().evaluate(d,1100000)["research_speed_mmps"] is None
def test_global_propagation_reflects_at_zero():
 g=TemporalGuard(initial_low_mmps=0,initial_high_mmps=0)
 out=g.evaluate(base(0,100,0),1100000)
 assert F(out["temporal_low_n"],out["temporal_low_d"])==0
def test_touching_endpoint_is_consistent():
 g=TemporalGuard();out=g.evaluate(base(10430,10500,10450),1100000)
 assert not out["temporal_latched"] and F(out["temporal_low_n"],out["temporal_low_d"])==10430
def test_missing_evidence_does_not_shrink_prediction():
 d=base();d.update(reason="anchor_missing",selection="abstain",research_speed_mmps=None)
 out=TemporalGuard().evaluate(d,1100000)
 assert F(out["temporal_low_n"],out["temporal_low_d"])==9390
 assert F(out["temporal_high_n"],out["temporal_high_d"])==10430
@pytest.mark.parametrize("now",[1010000,1000000,True,1.1])
def test_invalid_or_replayed_query_time_latches(now):
 out=TemporalGuard().evaluate(base(),now)
 assert out["selection"]=="abstain" and out["temporal_latched"]
def test_invalid_fraction_cannot_authorize_speed():
 d=base();d["low_d"]=0
 out=TemporalGuard().evaluate(d,1100000)
 assert out["selection"]=="abstain" and out["temporal_latched"]
def test_separate_trial_has_separate_guard_state():
 first=TemporalGuard();first.evaluate(base(1000,1400,1200),1100000)
 assert TemporalGuard().evaluate(base(),1100000)["selection"]=="candidate"
