import math
import pytest
from lifecycle_core import margin,choose,lease_steps,ContextTube
def test_actuator_envelope_changes_recoverability():
 assert margin(-4000,20,0,100,1)>=0
 assert margin(-4000,20,0,100,2.5)<0
def test_feasible_worst_authority_braking_certificate():
 u,m=choose(2000,20,0,180,2.5)
 assert m>=0 and -4000<=u<=2000
 if u<2000:assert margin(u+1,20,0,180,2.5)<0
def test_no_fabricated_positive_authority():
 with pytest.raises(ValueError):margin(-4000,20,0,100,4)
def test_longer_command_hold_consumes_certificate():
 assert margin(2000,15,5,110,2.5,100)<=margin(2000,15,5,110,2.5,10)
 n=lease_steps(2000,15,5,110,2.5)
 assert 0<=n<=100
 if n:assert margin(2000,15,5,110,2.5,n)>=0
 if n<100:assert margin(2000,15,5,110,2.5,n+1)<0
def test_correlated_range_jump_cannot_refresh_history():
 t=ContextTube((14,15),(15,16),(89,91))
 t.advance(0,.1,2.5)
 before=t.bounds()
 assert t.observe((119,121),(15,16))=="range_conflict"
 assert t.bounds()==before
def test_plausible_unauthenticated_refresh_does_not_shrink_history():
 t=ContextTube((14,15),(15,16),(89,91))
 before=t.bounds()
 assert t.observe((89.5,90.5),(15.2,15.8))=="plausible_unqualified_refresh"
 assert t.bounds()==before
def test_microstep_tube_contains_real_bounded_motion():
 t=ContextTube((14,15),(15,16),(89,91))
 ve,vl,g=14.6,15.3,90
 for _ in range(20):
  t.advance(-1000,.1,2.5)
  for k in range(10):
   ve=max(0,ve+.01*.4);vl=max(0,vl-.01*2.7);g+=.01*(vl-ve)
  b=t.bounds()
  assert b["ego_low"]<=ve<=b["ego_high"]
  assert b["lead_low"]<=vl<=b["lead_high"]
  assert b["gap_low"]<=g<=b["gap_high"]
