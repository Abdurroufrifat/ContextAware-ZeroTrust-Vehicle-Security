from pathlib import Path
import sys,json,gzip,copy
import pytest
from dev_common import P,B,F,R,KEYS
from dev_policy import CorroboratedPolicy
def saved_rows(case):
 trials=json.loads((B/"results.json").read_text())["trials"]
 t=next(t for t in trials if t["method"]=="full" and t["group"]=="primary" and t["case"]==case)
 with gzip.open(B/t["trace"],"rt") as f:return [json.loads(x) for x in f]
def test_real_signed_scene_conflict_brakes_and_explains(tmp_path):
 rows=saved_rows("scene_disinformation");policy=CorroboratedPolicy(tmp_path/"state",rows[0]["context"],KEYS)
 witnessed=False
 for row in rows[:60]:
  out=policy.process(row["input"])
  if out["guard"]["status"]=="unresolved_source_conflict":
   witnessed=True
   assert out["estimate_mmps"] is None and out["controller_input"] is None and out["fused_context"] is None
   assert out["requested_mmps2"]==-4000 and out["projected_margin_m"] is None
   assert out["estimator_decision"]["research_speed_mmps"] is None
   packet=policy.command_packet(-4000,row["input"]["now_us"],row["step"]+1)
   accepted,status=policy.actuate(packet,row["input"]["now_us"])
   assert accepted==-4000 and status=="authenticated"
   card=policy.card(out,accepted,status)
   assert card["selection_guard"]==out["guard"] and card["reason"]==out["reason"]
   assert card["controller_input"] is None
   assert not({"truth","attack_case","label"}&set(card))
 assert witnessed
def test_truth_and_attack_labels_are_forbidden(tmp_path):
 row=saved_rows("nominal")[0];policy=CorroboratedPolicy(tmp_path/"state",row["context"],KEYS)
 with pytest.raises(ValueError):policy.process({**row["input"],"truth":12})
def test_input_records_are_not_mutated(tmp_path):
 row=saved_rows("nominal")[0];record=copy.deepcopy(row["input"]);before=copy.deepcopy(record)
 policy=CorroboratedPolicy(tmp_path/"state",row["context"],KEYS);policy.process(record)
 assert record==before
def test_compatible_estimate_preserves_original_control(tmp_path):
 rows=saved_rows("nominal");policy=CorroboratedPolicy(tmp_path/"state",rows[0]["context"],KEYS);seen=False
 for row in rows[:15]:
  out=policy.process(row["input"])
  if out["guard"]["status"]=="corroborated":
   seen=True
   original=out["source_output"]
   for k in ("estimate_mmps","requested_mmps2","desired_mmps2","fused_context","controller_input","projected_margin_m"):
    assert out[k]==original[k]
 assert seen
def test_modified_actuator_frame_is_rejected(tmp_path):
 row=saved_rows("nominal")[0];policy=CorroboratedPolicy(tmp_path/"state",row["context"],KEYS)
 packet=policy.command_packet(-4000,row["input"]["now_us"],1)
 obj=json.loads(packet["wire"]);obj["body"]["accel_mmps2"]=2000;packet["wire"]=json.dumps(obj)
 accepted,status=policy.actuate(packet,row["input"]["now_us"])
 assert accepted==-4000 and status!="authenticated"
