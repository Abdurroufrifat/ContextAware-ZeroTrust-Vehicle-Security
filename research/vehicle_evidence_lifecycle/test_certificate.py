import copy,json
from certificate import seal_certificate,verify_certificate
KEY=b"development-key-32-bytes-0000000"
WIRE=json.dumps({"body":{"accel_mmps2":2000}})
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
