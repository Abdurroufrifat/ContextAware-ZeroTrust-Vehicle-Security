import gzip,json,uuid,shutil
from pathlib import Path
import pytest
from policy import ComparisonPolicy, control_choice
from cv_common import OLD,P,KEYS
from control_core import seal
@pytest.fixture
def records():
 with gzip.open(OLD/"simulator_outputs"/"trial_004"/"trace.jsonl.gz","rt",encoding="utf-8") as f:
  return [json.loads(next(f)) for _ in range(3)]
@pytest.fixture
def make():
 roots=[]
 def create(ctx,method):
  d=P/"research"/"project_release"/"runtime"/("cv-test-"+uuid.uuid4().hex);roots.append(d)
  return ComparisonPolicy(d,ctx,KEYS,method)
 yield create
 for d in roots:
  if d.exists():shutil.rmtree(d)
def test_evaluator_truth_is_rejected(records,make):
 p=make(records[0]["context"],"full");r=dict(records[0]["input"]);r["truth"]=0
 with pytest.raises(ValueError):p.process(r)
def test_refinement_ablation_keeps_original_scene_estimator(records,make):
 p=make(records[0]["context"],"no_refinement")
 for row in records:o=p.process(row["input"])
 assert "refinement_status" not in o["estimator_decision"]
 assert o["estimator_decision"]["scene_budget_satisfied"] is True
def test_learning_ablation_uses_frozen_refined_midpoint(records,make):
 p=make(records[0]["context"],"no_learning")
 for row in records:o=p.process(row["input"])
 assert o["estimator_decision"]["refinement_status"]=="available"
 assert o["estimator_decision"]["selection"]=="refined_midpoint"
def test_fusion_ablation_does_not_reject_only_radar_disagreement(records,make):
 a=make(records[0]["context"],"full");b=make(records[0]["context"],"no_fusion")
 a.process(records[0]["input"]);b.process(records[0]["input"])
 r=json.loads(json.dumps(records[1]["input"]))
 body=json.loads(r["radar_packet"]["wire"])["body"];body["gap_um"]+=10_000_000
 r["radar_packet"]=seal("radar",body,KEYS["radar"],r["now_us"])
 assert a.process(r)["controller_input"] is None
 assert b.process(r)["controller_input"] is not None
def test_actuator_ablation_accepts_changed_frame_while_full_brakes(records,make):
 a=make(records[0]["context"],"full");b=make(records[0]["context"],"no_actuator_auth")
 now=records[0]["input"]["now_us"];packet=a.command_packet(-2000,now,1)
 obj=json.loads(packet["wire"]);obj["body"]["accel_mmps2"]=2000;packet["wire"]=json.dumps(obj)
 assert a.actuate(packet,now)==(-4000,"bad_tag")
 assert b.actuate(packet,now)==(2000,"unverified_ablation")
def test_filter_ablation_preserves_unsafe_nominal_command():
 bounds={"ego_high":15.5,"lead_low":0,"gap_low":3}
 assert control_choice("no_brake",2000,bounds)[0]==2000
 assert control_choice("full",2000,bounds)[0]==-4000
def test_both_published_adaptations_emit_sets_without_truth(records,make):
 for method in ("s3e_brake","ssf_zocbf"):
  p=make(records[0]["context"],method)
  for row in records:o=p.process(row["input"])
  assert o["estimator_decision"]["plausible_sets_mps"]
  assert o["controller_input"] is not None

def test_anchor_adapter_uses_shared_protected_error_and_clock_bounds():
 from policy import observation_interval
 lo,hi=observation_interval("anchor",20,.09)
 assert abs(lo-19.37)<1e-12 and abs(hi-20.45)<1e-12

def test_refinement_comparison_uses_midpoint_instead_of_preserved_base_point(make):
 from fractions import Fraction as F
 import math
 with gzip.open(OLD/"simulator_outputs"/"trial_004"/"trace.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 p=make(rows[0]["context"],"no_refinement");reference=make(rows[0]["context"],"no_learning");seen=0;fallback=0
 for row in rows:
  out=p.process(row["input"]);ref=reference.process(row["input"]);d=ref["estimator_decision"]
  if d.get("refinement_status")!="available":
   assert out["estimate_mmps"]==ref["estimate_mmps"];fallback+=1;continue
  original=d["scene_decision"]
  if original["reason"]!="scene_certified_original":continue
  lo=F(original["witness_low_n"],original["witness_low_d"]);hi=F(original["witness_high_n"],original["witness_high_d"])
  expected=min(math.floor(lo+500),max(math.ceil(hi-500),round((lo+hi)/2)))
  assert out["estimate_mmps"]==expected;seen+=1
 assert seen>0 and fallback>0
