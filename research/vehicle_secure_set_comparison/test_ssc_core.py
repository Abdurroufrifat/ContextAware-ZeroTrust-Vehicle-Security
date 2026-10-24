from fractions import Fraction as F
import pytest
import ssc_core as c

def test_agreement_preserves_disconnected_alternatives():
 assert c.agree([(0,100)],[(10,20),(70,80)])==[(F(10),F(20)),(F(70),F(80))]
def test_missing_sensor_is_uninformative_for_unknown_one_attack():
 assert c.agree([(0,100)],[(10,20),None])==[(F(0),F(100))]
def test_empty_prior_never_reinitializes():
 assert c.agree([],[(10,20),(70,80)])==[]
def test_joint_impossible_measurements_exhaust_set():
 assert c.agree([(10000,10500)],[(7000,7500),(7100,7600)])==[]
def test_single_false_measurement_retains_true_branch():
 assert c.agree([(10000,10500)],[(7000,7500),(10100,10400)])==[(F(10100),F(10400))]
def test_command_prediction_reflects_at_zero_and_clips_input():
 assert c.predict([(100,200)],100000,-4000)==[(F(0),F(0))]
 assert c.predict([(10000,10250)],100000,2000)==[(F(10100),F(10450))]
def test_global_prediction_covers_extreme_inputs():
 assert c.predict([(10000,10250)],100000,None)==[(F(9600),F(10450))]
def test_delay_and_clock_bounds_are_inclusive():
 assert c.measurement(10000,95000,5000)==(F(9350),F(10450))
 assert c.measurement(10000,95001,5000) is None
 assert c.measurement(10000,4000,5000) is None
def test_disconnected_output_must_cover_every_branch():
 assert c.output([(0,100),(1000,1100)]) is None
def test_half_unit_rounding_does_not_break_budget():
 assert c.output([(0,1000)])==500
 assert c.output([(F(1,2),F(2001,2))]) is None
def test_context_lockdown_blocks_output():
 assert c.output([(10000,10200)],lockdown=True) is None
@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1])
def test_invalid_measurement_rejected(value):
 with pytest.raises(ValueError):c.measurement(value,50000,5000)
def test_integer_lattice_sensor_subset_oracle():
 prior=[(0,10),(20,30)]
 for a in range(0,31,5):
  for b in range(0,31,5):
   out=c.agree(prior,[(a,a+3),(b,b+3)])
   for x in range(34):
    expected=any(lo<=x<=hi for lo,hi in prior) and (a<=x<=a+3 or b<=x<=b+3)
    assert any(lo<=x<=hi for lo,hi in out)==expected
