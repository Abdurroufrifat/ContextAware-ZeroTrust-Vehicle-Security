from pathlib import Path
import importlib.util
from fractions import Fraction as F
import pytest

def call(**updates):
 p=Path(__file__).with_name("tube_core.py");assert p.exists(),"command-history propagation not implemented"
 spec=importlib.util.spec_from_file_location("tube_tested",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 args=dict(candidate_mmps=10200,anchor_mmps=10000,uncertainty_mmps=250,measured_us=0,now_us=100000,
 clock_uncertainty_us=0,execution_error_mmps2=1000,commands=[(0,50000,F(2000)),(50000,100000,F(2000))],assumptions=True)
 args.update(updates);return m.admit(**args)

def interval(result):return F(result["low_n"],result["low_d"]),F(result["high_n"],result["high_d"])

def test_delayed_anchor_propagates_without_measurement_truth_input():
 r=call();assert r["selection"]=="candidate"
 assert interval(r)==(9850,10450)
 assert r["value_mmps"]==10200 and not r["physical_validation_established"]

def test_exact_execution_interval():
 assert interval(call(execution_error_mmps2=0))==(9950,10450)

def test_bad_candidate_uses_propagated_anchor():
 r=call(candidate_mmps=13000)
 assert r["selection"]=="propagated_anchor" and r["value_mmps"]==10150

def test_missing_candidate_uses_bounded_propagated_value():
 assert call(candidate_mmps=None)["selection"]=="propagated_anchor"

def test_interior_command_boundary_is_needed_for_timestamp_extremum():
 r=call(candidate_mmps=None,uncertainty_mmps=0,execution_error_mmps2=0,
 measured_us=50000,clock_uncertainty_us=25000,
 commands=[(0,50000,F(2000)),(50000,100000,F(-4000))])
 assert interval(r)==(9800,9900)

def test_stopping_floor_keeps_bounds_nonnegative():
 r=call(candidate_mmps=0,anchor_mmps=100,uncertainty_mmps=50,execution_error_mmps2=0,
 commands=[(0,50000,F(-4000)),(50000,100000,F(-4000))])
 assert interval(r)==(0,0) and r["value_mmps"]==0

def test_midpoint_rounding_is_checked_at_exact_budget():
 r=call(candidate_mmps=None,uncertainty_mmps=500,execution_error_mmps2=0,
 commands=[(0,100000,F(0))])
 assert r["selection"]=="propagated_anchor" and r["value_mmps"]==10000

def test_fractional_center_cannot_round_into_false_certificate():
 r=call(candidate_mmps=None,uncertainty_mmps=500,execution_error_mmps2=0,
 commands=[(0,100000,F(5))])
 assert r["selection"]=="abstain" and r["reason"]=="budget_exhausted"

@pytest.mark.parametrize("updates",[
 {"assumptions":False},{"uncertainty_mmps":None},{"execution_error_mmps2":None},
 {"execution_error_mmps2":-1},{"clock_uncertainty_us":True},{"anchor_mmps":True},
 {"commands":None},{"commands":[]},{"commands":[(0,40000,F(0)),(50000,100000,F(0))]},
 {"commands":[(0,50000,F(0)),(40000,100000,F(0))]},{"commands":[(0,150000,F(0))]},
 {"commands":[(50000,100000,F(0))]},{"commands":[(0,100000,F(2001))]}
])
def test_missing_invalid_or_incomplete_history_abstains(updates):
 assert call(**updates)["selection"]=="abstain"

def test_more_than100ms_remains_forbidden_with_clock_uncertainty():
 assert call(clock_uncertainty_us=1)["reason"]=="uncertain_age_outside_window"

def test_future_ambiguous_timestamp_remains_forbidden():
 assert call(measured_us=100000,clock_uncertainty_us=1)["selection"]=="abstain"

def test_no_propagation_needed_at_current_exact_measurement_time():
 r=call(candidate_mmps=10000,measured_us=100000,commands=[])
 assert r["selection"]=="candidate" and interval(r)==(9750,10250)
