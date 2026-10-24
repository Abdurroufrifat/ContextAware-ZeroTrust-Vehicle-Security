import importlib.util,json,gzip,uuid,sys
from pathlib import Path
import pytest
R=Path(__file__).resolve().parent;RELEASE=R.parent/"project_release";sys.path.insert(0,str(RELEASE))
from research_session import strip_record,study_keys
spec=importlib.util.find_spec("learned_session")
def test_learned_application_exists():
 assert spec is not None,"Signed learned-estimator application is not implemented"
@pytest.fixture
def api():
 if spec is None:pytest.skip("application absent")
 from learned_session import LearnedResearchSession
 return LearnedResearchSession
def rows():
 with gzip.open(R/"simulator_outputs/trial_002/trace.jsonl.gz","rt",encoding="utf-8") as f:return [json.loads(next(f)) for _ in range(2)]
def make(api,tmp_path):
 data=rows();return api(tmp_path/"learned-session",data[0]["context"],study_keys()),data
def test_real_wire_packets_match_frozen_learned_loop(api,tmp_path):
 session,data=make(api,tmp_path)
 assert [session.process(strip_record(row)) for row in data]==[row["decision"] for row in data]
def test_reference_fields_rejected_before_learning(api,tmp_path):
 session,data=make(api,tmp_path);item=strip_record(data[0]);item["ego_speed_mps"]=999
 with pytest.raises(ValueError):session.process(item)
def test_wrong_scene_context_cannot_create_learned_speed(api,tmp_path):
 data=rows();session=api(tmp_path/"wrong-session",{**data[0]["context"],"capture_id":"wrong"},study_keys())
 assert all(session.process(strip_record(row))["research_speed_mmps"] is None for row in data)
def test_unsupported_model_not_exposed(api,tmp_path):
 data=rows()
 with pytest.raises(TypeError):api(tmp_path/"bad-model",data[0]["context"],study_keys(),model_path="external.pkl")
def test_immutable_model_loaded_matches_frozen_selection(api,tmp_path):
 session,data=make(api,tmp_path)
 assert session.model_name=="random_forest" and len(session.model_sha256)==64

def isolated_model_bundle(tmp_path,monkeypatch):
 import shutil,hashlib,learned_session
 bundle=tmp_path/"bundle";(bundle/"models").mkdir(parents=True)
 for name in ("model_selection.json","learned_core.py"):
  shutil.copyfile(R/name,bundle/name)
 for name in ("random_forest","ridge"):shutil.copyfile(R/"models"/(name+".joblib"),bundle/"models"/(name+".joblib"))
 lock={str(p.relative_to(bundle.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob("*") if p.is_file()}
 (bundle/"model_lock.json").write_text(json.dumps(lock),encoding="utf-8")
 monkeypatch.setattr(learned_session,"R",bundle)
 return bundle
def test_modified_model_selection_rejected_before_session_creation(api,tmp_path,monkeypatch):
 bundle=isolated_model_bundle(tmp_path,monkeypatch)
 path=bundle/"model_selection.json";obj=json.loads(path.read_text());obj["selected"]="ridge";path.write_text(json.dumps(obj),encoding="utf-8")
 data=rows()
 with pytest.raises(ValueError,match="integrity"):api(tmp_path/"tampered-selection",data[0]["context"],study_keys())
 assert not (tmp_path/"tampered-selection").exists()
def test_modified_estimator_source_rejected_before_session_creation(api,tmp_path,monkeypatch):
 bundle=isolated_model_bundle(tmp_path,monkeypatch)
 with (bundle/"learned_core.py").open("a",encoding="utf-8") as f:f.write("\\n# tampered source\\n")
 data=rows()
 with pytest.raises(ValueError,match="integrity"):api(tmp_path/"tampered-source",data[0]["context"],study_keys())
 assert not (tmp_path/"tampered-source").exists()
