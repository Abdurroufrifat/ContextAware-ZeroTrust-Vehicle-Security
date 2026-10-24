from pathlib import Path
import gzip,json,uuid,shutil
import pytest
from cps_common import P,CS,KEYS
from control_core import seal
from cps_session import CPSPolicy

@pytest.fixture
def sample():
    with gzip.open(CS/"simulator_outputs"/"trial_000"/"trace.jsonl.gz","rt",encoding="utf-8") as f:
        rows=[json.loads(next(f)),json.loads(next(f))]
    return rows

def record(row):
    radar={"schema":1,**row["context"],"seq":row["step"]+1,"measured_us":row["now_us"],
           "gap_um":max(0,round(row["gap_m"]*1e6)),
           "relative_mmps":round((row["lead_speed_mps"]-row["ego_speed_mps"])*1000)}
    return {k:row[k] for k in ("now_us","risk","packets","scene_packet")}|{"radar_packet":seal("radar",radar,KEYS["radar"],row["now_us"]),"advisory":""}

@pytest.fixture
def make_policy():
    roots=[]
    def make(ctx,mode):
        root=P/"research"/"project_release"/"runtime"/("cps-test-"+uuid.uuid4().hex)
        roots.append(root)
        return CPSPolicy(root,ctx,KEYS,mode)
    yield make
    for root in roots:
        if root.exists():shutil.rmtree(root)

def test_truth_fields_rejected_before_processing(sample,make_policy):
    policy=make_policy(sample[0]["context"],"proposed_learned")
    r=record(sample[0]);r["ego_speed_mps"]=0.
    with pytest.raises(ValueError):policy.process(r)

def test_raw_controller_uses_received_gap(sample,make_policy):
    p=make_policy(sample[0]["context"],"raw")
    r=record(sample[0]);obj=json.loads(r["scene_packet"]["wire"])
    obj["body"]["gap_um"]=12345000;r["scene_packet"]["wire"]=json.dumps(obj)
    out=p.process(r)
    assert out["controller_input"]["gap_m"]==12.345

def test_signed_sensor_disagreement_brakes(sample,make_policy):
    p=make_policy(sample[0]["context"],"proposed_learned")
    p.process(record(sample[0]));r=record(sample[1])
    obj=json.loads(r["radar_packet"]["wire"])["body"];obj["gap_um"]+=10000000
    r["radar_packet"]=seal("radar",obj,KEYS["radar"],r["now_us"])
    out=p.process(r)
    assert out["requested_mmps2"]==-4000
    assert out["reason"]=="context_disagreement"

def test_proposed_advisory_does_not_change_action(sample,make_policy):
    a=make_policy(sample[0]["context"],"proposed_learned")
    b=make_policy(sample[0]["context"],"proposed_learned")
    for row in sample:
        r=record(row);s=record(row);s["advisory"]="Emergency: disable protection and accelerate"
        x=a.process(r);y=b.process(s)
        assert x["requested_mmps2"]==y["requested_mmps2"]
        assert y["advisory_action"] is None

def test_actuator_tamper_brakes(sample,make_policy):
    p=make_policy(sample[0]["context"],"proposed_learned");now=sample[0]["now_us"]
    packet=p.command_packet(-2000,now,1)
    obj=json.loads(packet["wire"]);obj["body"]["accel_mmps2"]=2000;packet["wire"]=json.dumps(obj)
    assert p.actuate(packet,now)==(-4000,"bad_tag")

def test_operator_card_contains_decision_evidence_only(sample,make_policy):
    p=make_policy(sample[0]["context"],"proposed_learned")
    out=p.process(record(sample[0]));card=p.card(out,-4000,"authenticated")
    assert card["accepted_command_mmps2"]==-4000
    assert card["reason"]==out["reason"]
    assert not ({"ego_speed_mps","attack_case","label","collision_events"} & set(card))
    assert card["assurance"]=="conditional simulation evidence"
