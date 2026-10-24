import importlib.util,sys,json,gzip
from pathlib import Path
import pytest
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/"project_release"))
from research_session import strip_record,study_keys
spec=importlib.util.find_spec("command_scene_session")
def test_signed_application_exists():
 assert spec is not None,"command-scene signed application is absent"
@pytest.fixture
def api():
 if spec is None:pytest.skip("API absent")
 from command_scene_session import CommandSceneSession
 return CommandSceneSession
def rows():
 with gzip.open(R/"simulator_outputs/trial_002/trace.jsonl.gz","rt",encoding="utf-8") as f:return [json.loads(next(f)) for _ in range(2)]
def test_signed_inputs_match_fresh_loop(api,tmp_path):
 data=rows();session=api(tmp_path/"signed",data[0]["context"],study_keys())
 for row in data:assert session.process(strip_record(row))==row["decision"]
def test_truth_input_rejected(api,tmp_path):
 data=rows();session=api(tmp_path/"truth",data[0]["context"],study_keys());item=strip_record(data[0]);item["ego_speed_mps"]=123
 with pytest.raises(ValueError):session.process(item)
def test_untrusted_model_path_not_exposed(api,tmp_path):
 data=rows()
 with pytest.raises(TypeError):api(tmp_path/"model",data[0]["context"],study_keys(),model_path="external.pkl")
def test_unknown_mode_rejected_before_state_creation(api,tmp_path):
 data=rows()
 with pytest.raises(ValueError):api(tmp_path/"mode",data[0]["context"],study_keys(),mode="physical_drive")
 assert not (tmp_path/"mode").exists()
def test_repeated_time_rejected(api,tmp_path):
 data=rows();session=api(tmp_path/"time",data[0]["context"],study_keys());item=strip_record(data[0]);session.process(item)
 with pytest.raises(ValueError):session.process(item)

def copy_bundle(tmp_path,monkeypatch):
 import shutil,hashlib,command_scene_session
 from cs_common import P
 root=tmp_path/"bundle";(root/"models").mkdir(parents=True)
 for name in ("model_selection.json","command_scene_core.py","command_scene_guard.py"):shutil.copyfile(R/name,root/name)
 for name in ("forest","ridge"):shutil.copyfile(R/"models"/(name+".joblib"),root/"models"/(name+".joblib"))
 lock={str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest() for q in root.rglob("*") if q.is_file()}
 (root/"model_lock.json").write_text(json.dumps(lock),encoding="utf-8");monkeypatch.setattr(command_scene_session,"R",root)
 return root
def test_tampered_selection_rejected_before_state_creation(api,tmp_path,monkeypatch):
 root=copy_bundle(tmp_path,monkeypatch);p=root/"model_selection.json";obj=json.loads(p.read_text());obj["selected"]="ridge";p.write_text(json.dumps(obj),encoding="utf-8")
 with pytest.raises(ValueError,match="integrity"):api(tmp_path/"tampered",rows()[0]["context"],study_keys())
 assert not (tmp_path/"tampered").exists()
def test_tampered_guard_rejected_before_state_creation(api,tmp_path,monkeypatch):
 root=copy_bundle(tmp_path,monkeypatch);p=root/"command_scene_guard.py";p.write_bytes(p.read_bytes()+b"\\n# changed")
 with pytest.raises(ValueError,match="integrity"):api(tmp_path/"source",rows()[0]["context"],study_keys())
 assert not (tmp_path/"source").exists()
