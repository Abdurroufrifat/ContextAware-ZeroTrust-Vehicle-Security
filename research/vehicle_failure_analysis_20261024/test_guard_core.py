from fractions import Fraction as F
import pytest
from guard_core import corroborate
def base(lo=14872,hi=15497,value=15121):
 return {"conditional_budget_satisfied":True,"low_n":lo,"low_d":1,"high_n":hi,"high_d":1,"research_speed_mmps":value}
def test_saved_false_scene_cannot_replace_qualified_base():
 assert corroborate(21088,base())==(15121,"base_fallback")
def test_small_learned_improvement_is_preserved():
 assert corroborate(15090,base())==(15090,"corroborated")
def test_missing_estimate_is_not_silently_promoted():
 assert corroborate(None,base())==(None,"unavailable")
@pytest.mark.parametrize("flag",[False,None,"true",1])
def test_unqualified_or_nonboolean_certificate_abstains(flag):
 b=base();b["conditional_budget_satisfied"]=flag
 assert corroborate(21088,b)==(None,"base_unqualified")
def test_empty_admissible_point_set_abstains():
 assert corroborate(15000,base(14000,16000))==(None,"budget_exhausted")
def test_inconsistent_base_point_abstains():
 assert corroborate(21088,base(value=18000))==(None,"inconsistent_base")
def test_fractional_bounds_do_not_round_outward():
 b=base();b.update(low_n=296001,low_d=20,high_n=306001,high_d=20,research_speed_mmps=15001)
 assert corroborate(14800,b)==(15001,"base_fallback")
 assert corroborate(14801,b)==(14801,"corroborated")
def test_returned_points_satisfy_entire_reachable_interval():
 b=base()
 for value in range(14000,22001,3):
  out,_=corroborate(value,b)
  assert out is not None
  assert max(abs(F(out)-14872),abs(F(out)-15497))<=500
def test_boolean_speed_is_rejected():
 with pytest.raises(ValueError):corroborate(True,base())
