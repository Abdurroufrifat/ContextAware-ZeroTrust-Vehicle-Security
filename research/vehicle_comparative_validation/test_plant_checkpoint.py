from pathlib import Path
import json,hashlib
import pytest
from plant import step,public_window
from checkpoints import completed_trial,write_checkpoint
def test_right_endpoint_zero_contact_and_position_update():
 s={"ego_v":.01,"lead_v":1.0,"ego_x":60.0,"lead_x":70.0}
 t=step(s,-4,-2)
 assert t["ego_v"]==0 and t["ego_x"]==60
 assert t["lead_v"]==.98 and abs(t["lead_x"]-70.0098)<1e-12
def test_public_window_retains_outages_instead_of_clipping_age():
 src={"age_us":[None,1000,50_000_000],"innovation_mps":[None,0,2]}
 w=public_window(src,3,0)
 assert w["age_us"]==[None,1000,50_000_000]
def test_resume_returns_exact_completed_record_without_rewriting(tmp_path):
 trace=tmp_path/"trace.jsonl.gz";trace.write_bytes(b"trace")
 summary={"trial_id":7,"queries":200}
 write_checkpoint(tmp_path,summary,"lock-abc",[trace])
 before=(tmp_path/"summary.json").read_bytes()
 assert completed_trial(tmp_path,7,"lock-abc")==summary
 assert (tmp_path/"summary.json").read_bytes()==before
def test_resume_rejects_changed_trace_or_wrong_experiment(tmp_path):
 trace=tmp_path/"trace.jsonl.gz";trace.write_bytes(b"trace")
 write_checkpoint(tmp_path,{"trial_id":2},"old-lock",[trace])
 with pytest.raises(ValueError):completed_trial(tmp_path,2,"new-lock")
 trace.write_bytes(b"changed")
 with pytest.raises(ValueError):completed_trial(tmp_path,2,"old-lock")
def test_incomplete_attempt_is_not_counted_as_completed(tmp_path):
 (tmp_path/"trace.jsonl.gz").write_bytes(b"partial")
 assert completed_trial(tmp_path,1,"frozen")==None
