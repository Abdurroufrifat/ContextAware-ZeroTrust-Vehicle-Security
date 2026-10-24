import importlib.util
from fractions import Fraction as F
import pytest
spec=importlib.util.find_spec("command_scene_core")
def test_feature_exists():
 assert spec is not None,"command-refined scene interval is not implemented"
@pytest.fixture
def core():
 if spec is None:pytest.skip("feature absent")
 import command_scene_core
 return command_scene_core
def test_constant_lead_and_zero_command_contains_constant_ego(core):
 d=core.refine(35000000,35200000,12000,12000,1000000,1100000,[(1000000,1100000,0,1)],(F(9595),F(10405)))
 assert d["status"]=="available" and d["low"]<=10000<=d["high"]
 assert d["high"]-d["low"]<810
def test_semiimplicit_euler_acceleration_correction(core):
 # ego starts10m/s, a=2m/s2, lead12m/s. Right mean ego10.11, current10.2; gap delta .189m.
 d=core.refine(35000000,35189000,12000,12000,1000000,1100000,[(1000000,1100000,2000,1)],(F(9695),F(10505)))
 assert d["low"]<=10200<=d["high"]
 assert abs(float(d["center"])-10200)<50
@pytest.mark.parametrize("commands",[[],[(1010000,1100000,0,1)],[(1000000,1090000,0,1)],[(1000000,1055000,0,1),(1055000,1100000,0,1)]])
def test_incomplete_or_substep_commands_do_not_refine(core,commands):
 assert core.refine(35000000,35200000,12000,12000,1000000,1100000,commands,(F(9595),F(10405)))["status"]=="unavailable"
def test_near_zero_preserves_original_scene(core):
 d=core.refine(35000000,35000000,0,0,1000000,1100000,[(1000000,1100000,-4000,1)],(F(0),F(405)))
 assert d["status"]=="unavailable" and d["reason"]=="possible_zero_contact"
def test_impossible_lead_endpoint_is_conflict(core):
 d=core.refine(35000000,35200000,12000,20000,1000000,1100000,[(1000000,1100000,0,1)],(F(9595),F(10405)))
 assert d["status"]=="conflict"
def test_exact_discrete_acceleration_extrema_contained(core):
 # Constant lead12, ego initial10, acceleration various allowed by u=0 +/-1.
 for a in (-1000,-500,0,500,1000):
  current=10000+F(a,10);mean=10000+F(a*11,200)
  gap_delta=(12000-mean)*100
  d=core.refine(F(35000000),F(35000000)+gap_delta,12000,12000,1000000,1100000,[(1000000,1100000,0,1)],(current-405,current+405))
  assert d["low"]<=current<=d["high"]

def relative_payload(risk="normal",shift=0,candidate_offset=50):
 import json
 base={"candidate_status":"authenticated","anchor_status":"missing","command_status":"authenticated","risk_context":risk}
 packets={"candidate":{"wire":json.dumps({"body":{"speed_mmps":10000+shift+candidate_offset}})},"anchor":None,
  "command":{"wire":'{"body":{"commands":[[1000000,1100000,0,1]]}}'}}
 refined={"center":F(10000+shift),"low":F(9800+shift),"high":F(10200+shift),"command_correction":F(0),"lead_delta":F(0)}
 return base,packets,{},1100000,refined
def test_elevated_context_excludes_candidate_feature(core):
 x=core.relative_features(*relative_payload("elevated"))
 assert x[0]==0 and x[6]==0
def test_absolute_velocity_shift_cannot_change_model_features(core):
 assert core.relative_features(*relative_payload(shift=10000))==core.relative_features(*relative_payload())
@pytest.mark.parametrize("offset",[-3000,3000])
def test_large_candidate_disagreement_is_sign_symmetric(core,offset):
 x=core.relative_features(*relative_payload(candidate_offset=offset))
 assert x[0]==0 and x[6]==0

@pytest.mark.parametrize("N",range(1,11))
def test_varying_discrete_motion_with_bounded_sensor_errors(core,N):
 import random
 rng=random.Random(700+N)
 for repeat in range(30):
  v=F(10000);lead=F(12000);lead0=lead;gap=F(35000000);commands=[]
  for j in range(N):
   u=rng.randint(-4000,2000);actual=max(-4000,min(2000,u+rng.randint(-1000,1000)))
   v+=F(actual,100);lead+=F(rng.randint(-4000,2000),100);gap+=(lead-v)*10
   commands.append((1000000+j*10000,1010000+j*10000,u,1))
  d=core.refine(35000000+rng.randint(-1500,1500),gap+rng.randint(-1500,1500),lead0+rng.randint(-75,75),lead+rng.randint(-75,75),
   1000000,1000000+N*10000,commands,(v-405,v+405))
  assert d["status"]=="available" and d["low"]<=v<=d["high"]
def test_no_zero_contact_boundary_is_strict(core):
 d=core.refine(35000000,35000000,1000,1000,1000000,1100000,[(1000000,1100000,0,1)],(F(200),F(1000)))
 assert d["status"]=="unavailable"
def test_refined_output_uses_joint_certificate_only(core):
 import sys,json,gzip,hashlib
 from pathlib import Path
 r=Path(__file__).resolve().parent;sys.path.insert(0,str(r))
 from cs_common import KW
 from command_scene_guard import CommandSceneGuard
 trial=next(t for t in json.loads((KW/"results.json").read_text())["trials"] if (t["case"],t["age_us"],t["seed"],t["policy"])==("nominal",90000,17,"kinematic"))
 with gzip.open(KW/trial["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(next(f)) for _ in range(2)]
 guard=CommandSceneGuard(hashlib.sha256(b"COACT-SUMO-PUBLIC-TEST-scene").digest(),rows[0]["context"])
 for row in rows:out=guard.evaluate(row["scene_packet"],row["now_us"],row["decision"]["base_decision"],row["packets"])
 assert out["refined_budget_satisfied"] is True and out["scene_budget_satisfied"] is False
