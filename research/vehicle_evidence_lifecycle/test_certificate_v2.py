import copy,json
from certificate_v2 import seal_certificate,verify_certificate
KEY=b"development-key-32-bytes-0000000"
WIRE=json.dumps({"body":{"accel_mmps2":2000,"duration_us":100000,"seq":1,"issued_us":1000000}})
CTX={"vehicle_id":"e","session_id":"s","capture_id":"c"}
BOUNDS={"ego_high":15.,"lead_low":5.,"gap_low":150.}
def packet():
 return seal_certificate(WIRE,KEY,CTX,1,1000000,2000,BOUNDS,2.5,10,"obs")
def check(p,now=1000000,seen=-1):
 return verify_certificate(p,WIRE,KEY,CTX,now,BOUNDS,2.5,"obs",seen)
def test_sound_certificate_is_accepted():
 assert check(packet())[0]
def test_fresh_authentication_cannot_extend_expired_evidence():
 assert not check(packet(),1100001)[0]
def test_certificate_cannot_cross_observation_or_command():
 p=packet();assert not verify_certificate(p,"other",KEY,CTX,1000000,BOUNDS,2.5,"obs",-1)[0]
 assert not verify_certificate(p,WIRE,KEY,CTX,1000000,BOUNDS,2.5,"new",-1)[0]
def test_signed_but_false_bounds_are_rejected():
 b={**BOUNDS,"gap_low":999.}
 p=seal_certificate(WIRE,KEY,CTX,1,1000000,2000,b,2.5,10,"obs")
 assert not check(p)[0]
def test_signed_unsafe_command_is_rejected():
 b={**BOUNDS,"gap_low":3.}
 p=seal_certificate(WIRE,KEY,CTX,1,1000000,2000,b,2.5,10,"obs")
 assert not verify_certificate(p,WIRE,KEY,CTX,1000000,b,2.5,"obs",-1)[0]
def test_replay_and_context_changes_rejected():
 assert not check(packet(),seen=1)[0]
 p=packet();assert not verify_certificate(p,WIRE,KEY,{**CTX,"session_id":"other"},1000000,BOUNDS,2.5,"obs",-1)[0]

def test_correctly_signed_certificate_cannot_lie_about_wire_command():
 b={"ego_high":20.,"lead_low":0.,"gap_low":200.}
 p=seal_certificate(WIRE,KEY,CTX,1,1000000,-4000,b,2.5,10,"obs")
 assert not verify_certificate(p,WIRE,KEY,CTX,1000000,b,2.5,"obs",-1)[0]

def test_lease_must_cover_the_wire_command_duration():
 p=seal_certificate(WIRE,KEY,CTX,1,1000000,2000,BOUNDS,2.5,1,"obs")
 assert not check(p)[0]
def test_same_bounds_do_not_authorize_delayed_start():
 assert not check(packet(),1000001)[0]
def test_sequence_is_bound_to_wire_not_only_signature():
 p=seal_certificate(WIRE,KEY,CTX,2,1000000,2000,BOUNDS,2.5,10,"obs")
 assert not check(p)[0]
