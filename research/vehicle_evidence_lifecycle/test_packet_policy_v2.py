import json,gzip
import pytest
from lc_common import *
from packet_policy_v2 import LifecyclePolicy
def row():
 ts=json.loads((BASE/"results.json").read_text())["trials"];t=next(t for t in ts if t["method"]=="full" and t["case"]=="nominal")
 with gzip.open(BASE/t["trace"],"rt") as f:return json.loads(next(f))
def test_packet_only_interface(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint")
 with pytest.raises(ValueError):p.process({**r["input"],"truth":0})
def test_certificate_binds_actual_command_and_card(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint");out=p.process(r["input"])
 packet=p.command_packet(out["requested_mmps2"],r["input"]["now_us"],1)
 assert "certificate" in packet
 accepted,status=p.actuate(packet,r["input"]["now_us"])
 assert accepted==out["requested_mmps2"]
 card=p.card(out,accepted,status)
 assert card["certificate"]["execution_error"]==2.5
 assert card["assurance"]=="conditional simulation evidence"
 assert not({"truth","attack_case","label"}&set(card))
def test_forged_actor_frame_fails_closed(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint");out=p.process(r["input"])
 packet=p.command_packet(out["requested_mmps2"],r["input"]["now_us"],1)
 obj=json.loads(packet["wire"]);obj["body"]["accel_mmps2"]=2000;packet["wire"]=json.dumps(obj)
 accepted,status=p.actuate(packet,r["input"]["now_us"])
 assert accepted==-4000 and status!="authenticated"
def test_missing_sidecar_cannot_authorize_acceleration(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint");out=p.process(r["input"])
 packet=p.command_packet(2000,r["input"]["now_us"],1);packet.pop("certificate")
 accepted,status=p.actuate(packet,r["input"]["now_us"])
 assert accepted==-4000 and status.startswith("certificate_")

def test_deadline_watchdog_brakes_without_new_controller_call(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint")
 out=p.process(r["input"]);packet=p.command_packet(out["requested_mmps2"],r["input"]["now_us"],1)
 value,status=p.actuate(packet,r["input"]["now_us"])
 assert p.watchdog(r["input"]["now_us"]+99999)[0]==value
 assert p.watchdog(r["input"]["now_us"]+100000)==(-4000,"watchdog_expired")
def test_missed_cadence_requires_new_qualification(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint");out=p.process(r["input"])
 p.actuate(p.command_packet(out["requested_mmps2"],r["input"]["now_us"],1),r["input"]["now_us"])
 with pytest.raises(ValueError,match="cadence"):
  p.process({**r["input"],"now_us":r["input"]["now_us"]+200000})
 assert p.watchdog(r["input"]["now_us"]+200000)[0]==-4000

def test_missing_actor_cycle_is_not_propagated_as_old_command(tmp_path):
 r=row();p=LifecyclePolicy(tmp_path/"state",r["context"],KEYS,"joint")
 p.process(r["input"])
 assert p.watchdog(r["input"]["now_us"]+100000)[0]==-4000
 with pytest.raises(ValueError,match="actuation"):
  p.process({**r["input"],"now_us":r["input"]["now_us"]+100000})
