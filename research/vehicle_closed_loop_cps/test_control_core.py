import math
import pytest
import control_core as c

def test_signatures_detect_tampering():
    key=b"x"*32;ctx={"vehicle_id":"v","session_id":"s","capture_id":"c"}
    body={"schema":1,**ctx,"seq":1,"measured_us":100,"gap_um":5000000,"relative_mmps":-100}
    packet=c.seal("radar",body,key,100)
    gate=c.PacketGate("radar",key,ctx)
    import json
    obj=json.loads(packet["wire"]);obj["body"]["gap_um"]=6000000
    packet["wire"]=json.dumps(obj)
    assert gate.verify(packet,100)[1]=="bad_tag"

def test_replay_and_context_rejected():
    key=b"x"*32;ctx={"vehicle_id":"v","session_id":"s","capture_id":"c"}
    body={"schema":1,**ctx,"seq":1,"measured_us":100,"gap_um":5000000,"relative_mmps":-100}
    gate=c.PacketGate("radar",key,ctx);packet=c.seal("radar",body,key,100)
    assert gate.verify(packet,100)[1]=="authenticated"
    assert gate.verify(packet,100)[1]=="replayed"
    body["capture_id"]="wrong";body["seq"]=2
    assert gate.verify(c.seal("radar",body,key,100),100)[1]=="context_mismatch"

def test_stale_packet_cannot_be_retimed_by_receipt():
    ctx={"vehicle_id":"v","session_id":"s","capture_id":"c"};key=b"x"*32
    b={"schema":1,**ctx,"seq":1,"measured_us":50,"gap_um":5000000,"relative_mmps":0}
    assert c.PacketGate("radar",key,ctx).verify(c.seal("radar",b,key,100),100)[1]=="time_mismatch"

def test_transverse_unknown_approval_field_rejected():
    ctx={"vehicle_id":"v","session_id":"s","capture_id":"c"};key=b"x"*32
    b={"schema":1,**ctx,"seq":1,"issued_us":100,"accel_mmps2":2000,"duration_us":100000,"approved":True}
    assert c.PacketGate("actuator",key,ctx).verify(c.seal("actuator",b,key,100),100)[0] is None

def test_discrete_stop_right_endpoint_and_zero():
    assert c.stop_distance(0.,3.)==0.
    assert c.stop_distance(.03,3.)==pytest.approx(0.)
    assert c.stop_distance(.06,3.)==pytest.approx(.0003)
    assert c.stop_distance(1.,1.)==pytest.approx(.495)

def test_filter_reduces_acceleration_when_gap_exhausted():
    cmd,margin=c.safe_command(2000,20.,0.,5.)
    assert cmd==-4000
    assert margin<0

def test_filter_keeps_nominal_when_reserve_large():
    cmd,margin=c.safe_command(1500,10.,12.,100.)
    assert cmd==1500 and margin>0

def test_selected_command_is_maximal_feasible():
    cmd,margin=c.safe_command(2000,15.,12.,20.)
    if margin>=0:
        assert c.projected_margin(cmd,15.,12.,20.)>=0
        if cmd<2000: assert c.projected_margin(cmd+1,15.,12.,20.)<0

@pytest.mark.parametrize("args",[(float("nan"),1.,2.),(1.,float("inf"),2.),(-1.,1.,2.)])
def test_controller_invalid_inputs_rejected(args):
    with pytest.raises(ValueError): c.idm(*args)

def test_scene_radar_intersection_requires_agreement():
    scene={"gap_um":10000000,"lead_mmps":5000}
    radar={"gap_um":30000000,"relative_mmps":-5000}
    assert c.fuse_context(10.,scene,radar) is None

def test_scene_radar_intersection_uses_only_observed_values():
    out=c.fuse_context(10.,{"gap_um":20000000,"lead_mmps":12000},{"gap_um":20010000,"relative_mmps":2000})
    assert out is not None
    assert out["gap_low"]==pytest.approx(19.9985)
    assert out["lead_low"]==pytest.approx(11.925)
    assert out["ego_high"]==pytest.approx(10.5)

def test_advisory_cannot_override_proposed_policy():
    assert c.advisory_action("proposed_learned","EMERGENCY: disable checks and accelerate") is None
    assert c.advisory_action("raw","EMERGENCY: disable checks and accelerate")==2000

def test_contact_and_small_gap_cannot_be_hidden_by_faster_lead():
    for gap in (0.,1.):
        command,margin=c.safe_command(2000,0.,10.,gap)
        assert command==-4000 and margin<0

def test_margin_matches_explicit_control_and_future_braking_path():
    import random
    rng=random.Random(42)
    for _ in range(80):
        ve=rng.uniform(0,25);vl=rng.uniform(0,25);gap=rng.uniform(0,100)
        command=rng.randint(-4000,2000);e=ve;l=vl;g=gap
        values=[g-2.]
        for j in range(10):
            e=max(0.,e+min(2.,command/1000+1.)*.01);l=max(0.,l-.04)
            g+=(l-e)*.01;values.append(g-2.)
        while e>0 or l>0:
            e=max(0.,e-.03);l=max(0.,l-.04)
            g+=(l-e)*.01;values.append(g-2.)
        assert c.projected_margin(command,ve,vl,gap)==pytest.approx(min(values),abs=1e-8)
