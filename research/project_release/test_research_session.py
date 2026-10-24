from pathlib import Path
import gzip,json,hashlib
import pytest
from research_session import ResearchSession,study_keys,strip_record
ROOT=Path(__file__).resolve().parent;KW=ROOT.parent/"vehicle_kinematic_witness_study"
def sample(mode="kinematic"):
 t=next(x for x in json.loads((KW/"results.json").read_text())["trials"] if (x["case"],x["age_us"],x["seed"],x["policy"])==("nominal",90000,17,mode))
 with gzip.open(KW/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(next(f)) for _ in range(2)]
 return rows
def make(tmp_path,mode="kinematic",context=None,keys=None,name="session"):
 rows=sample(mode);return ResearchSession(tmp_path/name,context or rows[0]["context"],keys or study_keys(),mode=mode),rows
@pytest.mark.parametrize("mode",["kinematic","temporal"])
def test_real_signed_packets_match_existing_gateway_results(tmp_path,mode):
 session,rows=make(tmp_path,mode)
 for row in rows:assert session.process(strip_record(row))==row["decision"]
def test_input_extraction_excludes_reference_and_control_truth():
 row=sample()[0];item=strip_record(row)
 assert set(item)=={"now_us","risk","packets","scene_packet"}
 assert "ego_speed_mps" not in json.dumps(item)
def test_caller_cannot_supply_an_approved_decision(tmp_path):
 session,rows=make(tmp_path);item=strip_record(rows[0]);item["base_decision"]={"research_speed_mmps":12345}
 with pytest.raises(ValueError):session.process(item)
def test_physical_operation_field_is_rejected(tmp_path):
 session,rows=make(tmp_path);item=strip_record(rows[0]);item["operation"]="drive"
 with pytest.raises(ValueError):session.process(item)
def test_duplicate_query_time_is_rejected(tmp_path):
 session,rows=make(tmp_path);item=strip_record(rows[0]);session.process(item)
 with pytest.raises(ValueError):session.process(item)
def test_invalid_packet_shape_cannot_reach_gateway(tmp_path):
 session,rows=make(tmp_path);item=strip_record(rows[0]);item["packets"]["candidate"]={"body":"not a signed packet"}
 with pytest.raises(ValueError):session.process(item)
def test_wrong_context_never_selects(tmp_path):
 rows=sample();context={**rows[0]["context"],"capture_id":"wrong-capture"}
 session,_=make(tmp_path,context=context)
 assert all(session.process(strip_record(row))["research_speed_mmps"] is None for row in rows)
def test_key_material_must_be_distinct(tmp_path):
 keys=study_keys();keys["scene"]=keys["anchor"]
 with pytest.raises(ValueError):make(tmp_path,keys=keys)
def test_short_keys_rejected(tmp_path):
 keys=study_keys();keys["scene"]=b"short"
 with pytest.raises(ValueError):make(tmp_path,keys=keys)
def test_unknown_mode_rejected(tmp_path):
 rows=sample()
 with pytest.raises(ValueError):ResearchSession(tmp_path/"bad",rows[0]["context"],study_keys(),mode="physical")
def test_existing_session_state_is_not_overwritten(tmp_path):
 make(tmp_path)
 with pytest.raises(FileExistsError):make(tmp_path)
def test_state_directory_must_be_inside_release_runtime():
 rows=sample()
 with pytest.raises(PermissionError):ResearchSession(ROOT.parent/"forbidden_release_state",rows[0]["context"],study_keys())
def test_bad_candidate_signature_keeps_only_verified_alternatives(tmp_path):
 session,rows=make(tmp_path);first=strip_record(rows[0]);session.process(first)
 second=strip_record(rows[1]);obj=json.loads(second["packets"]["candidate"]["wire"]);obj["body"]["speed_mmps"]+=3000
 second["packets"]["candidate"]={**second["packets"]["candidate"],"wire":json.dumps(obj)}
 out=session.process(second)
 assert out["base_decision"]["candidate_status"]=="bad_tag" and out["physical_speed_authorized"] is False
def test_public_fixture_keys_match_frozen_study():
 assert study_keys()["scene"]==hashlib.sha256(b"COACT-SUMO-PUBLIC-TEST-scene").digest()
