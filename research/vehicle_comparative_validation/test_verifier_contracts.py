import copy,json,hashlib
import pytest
from verification_contracts import expected_public_run,expected_delivery,verify_checkpoint,verify_refinement_metadata

def test_public_split_cannot_self_declare():
 c={"seeds":[1103,1229],"public_input_policy":{"stress_runs":["heldout_medium","heldout_hard"]}}
 assert expected_public_run(c,1103,False)=="heldout_medium"
 assert expected_public_run(c,1229,False)=="heldout_hard"
 with pytest.raises(ValueError):expected_public_run(c,2903,False)
 assert expected_public_run(c,2903,True)=="heldout_medium"

def test_tamper_is_prescribed_and_tag_unchanged():
 p={"wire":json.dumps({"body":{"accel_mmps2":-123,"seq":1},"tag":"old"}),"received_us":9}
 t=expected_delivery(p,"actuator_tamper",True)
 assert json.loads(t["wire"])=={"body":{"accel_mmps2":2000,"seq":1},"tag":"old"}
 assert expected_delivery(p,"nominal",True)==p
 assert expected_delivery(p,"actuator_tamper",False)==p
 assert json.loads(p["wire"])["body"]["accel_mmps2"]==-123

def test_checkpoint_binds_exact_files_and_identity(tmp_path):
 root=tmp_path/"trial_0001"/"attempt_001";root.mkdir(parents=True)
 s={"trial_id":1,"trace":str((root/"trace.jsonl.gz").relative_to(tmp_path)),"trajectory":str((root/"trajectory.jsonl.gz").relative_to(tmp_path))}
 for n,v in [("summary.json",json.dumps(s)),("trace.jsonl.gz","a"),("trajectory.jsonl.gz","b")]: (root/n).write_text(v)
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 marker={"trial_id":1,"lock_digest":"right","files":{n:h(root/n) for n in ["summary.json","trace.jsonl.gz","trajectory.jsonl.gz"]}}
 (root/"complete.json").write_text(json.dumps(marker))
 verify_checkpoint(tmp_path,s,"right")
 with pytest.raises(AssertionError):verify_checkpoint(tmp_path,s,"wrong")
 (root/"trace.jsonl.gz").write_text("changed")
 with pytest.raises(AssertionError):verify_checkpoint(tmp_path,s,"right")
 (root/"trace.jsonl.gz").write_text("a")
 marker["files"].pop("trajectory.jsonl.gz");(root/"complete.json").write_text(json.dumps(marker))
 with pytest.raises(AssertionError):verify_checkpoint(tmp_path,s,"right")

def test_confirmation_refinement_requires_source_metadata():
 ed={"refinement_status":"available","scene_decision":{"research_speed_mmps":7}}
 o={"ablation_refinement_available":True,"estimator_decision":{"selection":"scene_midpoint_ablation"},"unrefined_original_decision":ed["scene_decision"]}
 with pytest.raises(AssertionError):verify_refinement_metadata(o,False)
 o["refinement_ablation_source_decision"]=ed
 assert verify_refinement_metadata(o,False)==ed
 o["ablation_refinement_available"]=False
 with pytest.raises(AssertionError):verify_refinement_metadata(o,False)

def test_history_must_match_accepted_not_requested():
 from verification_contracts import accepted_history
 ledger=[(1010000,1100000,0),(1100000,1200000,-4000)]
 assert accepted_history(ledger,1105000,1200000)==[[1100000,1200000,-4000,1]]
 assert accepted_history(ledger,1005000,1200000)==[]
 assert accepted_history(ledger,None,1200000)==[]
