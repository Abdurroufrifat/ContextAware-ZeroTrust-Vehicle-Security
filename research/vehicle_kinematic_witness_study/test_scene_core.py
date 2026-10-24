import json
from fractions import Fraction as F
import pytest
from scene_core import SceneGuard,sign_scene,witness_bounds
KEY=b"public-test-scene-key-0123456789"
CTX=dict(vehicle_id="v",session_id="s",capture_id="c")
def packet(seq,now,gap=20000000,lead=12000):
 body=dict(schema=1,source_id="scene",key_id="s-key",**CTX,channel="lead_range",seq=seq,measured_us=now,gap_um=gap,lead_mmps=lead)
 return dict(wire=json.dumps(sign_scene(body,KEY)),received_us=now)
def base(value=10000):
 return dict(selection="candidate",reason="candidate_inside_reachable_budget",research_speed_mmps=value,research_speed_mps=value/1000,
  risk_context="normal",conditional_budget_satisfied=True,physical_validation_established=False,promotion_allowed=False)
def test_witness_units_and_conservative_radius():
 lo,hi=witness_bounds(20000000,20200000,12000,100000)
 assert (lo,hi)==(F(9595),F(10405))
def test_first_packet_only_seeds_history():
 out=SceneGuard(KEY,CTX).evaluate(packet(1,100000),100000,base())
 assert out["selection"]=="abstain" and out["scene_status"]=="authenticated" and out["reason"]=="scene_history_unavailable"
def test_nominal_point_is_kept_when_whole_interval_certifies_it():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base())
 out=g.evaluate(packet(2,200000,20200000),200000,base())
 assert out["research_speed_mmps"]==10000 and out["reason"]=="scene_certified_original"
def test_slow_biased_point_is_replaced_without_ego_truth():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base(7000))
 out=g.evaluate(packet(2,200000,20200000),200000,base(7000))
 assert out["research_speed_mmps"]==10000 and out["selection"]=="scene_witness"
def test_original_abstention_cannot_become_selection():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base())
 b=base();b.update(selection="abstain",reason="lockdown_context",risk_context="lockdown",research_speed_mmps=None,research_speed_mps=None)
 out=g.evaluate(packet(2,200000,20200000),200000,b)
 assert out["selection"]=="abstain" and out["reason"]=="lockdown_context"
def test_bad_tag_never_updates_anti_replay_state():
 g=SceneGuard(KEY,CTX);p=packet(1,100000);obj=json.loads(p["wire"]);obj["body"]["lead_mmps"]+=3000;p["wire"]=json.dumps(obj)
 assert g.evaluate(p,100000,base())["scene_status"]=="bad_tag"
 assert g.evaluate(packet(1,100000),100000,base())["scene_status"]=="authenticated"
def test_replay_is_rejected():
 g=SceneGuard(KEY,CTX);p=packet(1,100000);g.evaluate(p,100000,base())
 assert g.evaluate(p,100000,base())["scene_status"]=="replayed"
def test_wrong_context_is_rejected():
 p=packet(1,100000);obj=json.loads(p["wire"]);obj["body"]["vehicle_id"]="other";p["wire"]=json.dumps(sign_scene(obj["body"],KEY))
 assert SceneGuard(KEY,CTX).evaluate(p,100000,base())["scene_status"]=="context_mismatch"
@pytest.mark.parametrize("received,now",[(99999,100000),(100000,100001),(100000,99999)])
def test_non_query_aligned_time_rejected(received,now):
 p=packet(1,100000);p["received_us"]=received
 assert SceneGuard(KEY,CTX).evaluate(p,now,base())["scene_status"]=="time_mismatch"
def test_missing_scene_abstains():
 assert SceneGuard(KEY,CTX).evaluate(None,100000,base())["research_speed_mmps"] is None
def test_age_boundary_and_reseed_after_dropout():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base())
 assert g.evaluate(packet(2,200001,20200000),200001,base())["reason"]=="scene_history_outside_window"
 assert g.evaluate(packet(3,300001,20400000),300001,base())["research_speed_mmps"]==10000
def test_duplicate_fields_rejected():
 p=packet(1,100000);p["wire"]=p["wire"].replace('"schema": 1','"schema": 1, "schema": 1')
 assert SceneGuard(KEY,CTX).evaluate(p,100000,base())["scene_status"]=="malformed"
def test_boolean_schema_rejected():
 p=packet(1,100000);obj=json.loads(p["wire"]);obj["body"]["schema"]=True;p["wire"]=json.dumps(sign_scene(obj["body"],KEY))
 assert SceneGuard(KEY,CTX).evaluate(p,100000,base())["scene_status"]=="malformed"
def test_tiny_time_window_does_not_hide_large_uncertainty():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base())
 assert g.evaluate(packet(2,100001,20000002),100001,base())["reason"]=="scene_budget_exhausted"
def test_empty_nonnegative_witness_abstains():
 g=SceneGuard(KEY,CTX);g.evaluate(packet(1,100000),100000,base())
 assert g.evaluate(packet(2,200000,40000000),200000,base())["reason"]=="scene_inconsistent_interval"
def test_bound_is_conservative_for_discrete_extreme_accelerations():
 for ae in (-4000,0,2000):
  for al in (-4000,0,2000):
   ve=10000;vl=12000;gap=F(20000000);g0=gap
   for k in range(10):
    ve+=F(ae,100);vl+=F(al,100);gap+=(vl-ve)*10
   lo,hi=witness_bounds(g0,gap,vl,100000)
   assert lo<=ve<=hi
