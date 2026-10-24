import importlib.util,sys,math
from pathlib import Path
from fractions import Fraction as F
import pytest
spec=importlib.util.find_spec("learned_core")
def test_implementation_exists():
 assert spec is not None, "learned_core feature is not implemented"
@pytest.fixture
def core():
 if spec is None:pytest.skip("implementation absent")
 import learned_core
 return learned_core
@pytest.mark.parametrize("prediction,expected",[(12000,10095),(0,9905),(10000,10000)])
def test_projection_bounds_full_reference_interval(core,prediction,expected):
 assert core.project_point(prediction,F(9595),F(10405))==expected
def test_noninteger_endpoints_round_inward(core):
 assert core.project_point(-100,F(9595,1)+F(1,3),F(10405,1)+F(1,3))==9906
 assert core.project_point(20000,F(9595,1)+F(1,3),F(10405,1)+F(1,3))==10095
@pytest.mark.parametrize("lo,hi",[(0,1001),(5,4),(-1,10)])
def test_invalid_or_exhausted_interval_has_no_point(core,lo,hi):
 assert core.project_point(100,F(lo),F(hi)) is None
@pytest.mark.parametrize("v",[float("nan"),float("inf"),-float("inf"),True,"10000"])
def test_invalid_model_value_rejected(core,v):
 assert core.project_point(v,F(9595),F(10405)) is None
def test_half_integer_point_does_not_round_outside(core):
 assert core.project_point(10000.5,F(9501),F(10500))==10000
def test_target_and_reference_are_not_model_features(core):
 payload=dict(base={"candidate_status":"authenticated","anchor_status":"authenticated","command_status":"authenticated"},
 packets={"candidate":{"wire":'{"body":{"speed_mmps":10100}}'},"anchor":{"wire":'{"body":{"speed_mmps":10000,"measured_us":910000}}'},"command":{"wire":'{"body":{"commands":[[900000,1000000,1000,1]]}}'}},
 scene_packet={"wire":'{"body":{"lead_mmps":12000,"gap_um":35000000}}'},now_us=1000000)
 a=core.features(payload,F(9595),F(10405))
 payload["base"]["ego_speed_mps"]=999;payload["base"]["selected_disagreement_mps"]=-999
 assert core.features(payload,F(9595),F(10405))==a
 assert len(a)==10 and all(math.isfinite(x) for x in a)
def test_unauthenticated_candidate_not_used(core):
 base={"candidate_status":"bad_tag","anchor_status":"missing","command_status":"missing"}
 payload=dict(base=base,packets={"candidate":{"wire":"not JSON"},"anchor":None,"command":None},
 scene_packet={"wire":'{"body":{"lead_mmps":12000,"gap_um":35000000}}'},now_us=1000000)
 a=core.features(payload,F(9595),F(10405))
 assert a[1]==0 and a[8]==0 and a[9]==0

class ConstantRegressor:
 def __init__(self,value):self.value=value
 def predict(self,x):return [self.value]
def fixture_decision():
 return {"research_speed_mmps":10000,"research_speed_mps":10.,"selection":"candidate","reason":"scene_certified_original",
 "scene_status":"authenticated","scene_budget_satisfied":True,"base_decision":{"candidate_status":"missing","anchor_status":"missing","command_status":"missing"},
 "witness_low_n":9595,"witness_low_d":1,"witness_high_n":10405,"witness_high_d":1}
def payload():
 return {"candidate":None,"anchor":None,"command":None},{"wire":'{"body":{"lead_mmps":12000,"gap_um":35000000}}'}
def test_shield_blocks_arbitrary_model_excursion(core):
 d=fixture_decision();pack,scene=payload()
 out=core.apply_learned(d,pack,scene,1000000,ConstantRegressor(100.),True)
 assert out["research_speed_mmps"]==10095 and out["scene_budget_satisfied"] is True
 assert out["learned_prediction_mmps"]==110000.
def test_unshielded_ablation_exposes_model_prediction(core):
 d=fixture_decision();pack,scene=payload()
 out=core.apply_learned(d,pack,scene,1000000,ConstantRegressor(100.),False)
 assert out["research_speed_mmps"]==110000 and out["scene_budget_satisfied"] is False
def test_base_abstention_cannot_be_overridden(core):
 d=fixture_decision();d.update(research_speed_mmps=None,research_speed_mps=None,selection="abstain",reason="lockdown_context")
 pack,scene=payload()
 out=core.apply_learned(d,pack,scene,1000000,ConstantRegressor(100.),True)
 assert out["research_speed_mmps"] is None and out["reason"]=="lockdown_context"
def test_nonfinite_prediction_falls_back_to_original_scene(core):
 d=fixture_decision();pack,scene=payload()
 out=core.apply_learned(d,pack,scene,1000000,ConstantRegressor(float("nan")),True)
 assert out["research_speed_mmps"]==10000 and out["learned_status"]=="invalid_prediction_fallback"
