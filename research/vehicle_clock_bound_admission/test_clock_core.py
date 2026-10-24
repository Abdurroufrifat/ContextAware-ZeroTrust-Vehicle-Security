from pathlib import Path
import importlib.util
from fractions import Fraction
import pytest

def select(**changes):
    path=Path(__file__).with_name("clock_core.py")
    assert path.exists(),"clock-uncertainty selector not implemented"
    spec=importlib.util.spec_from_file_location("clock_under_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=dict(candidate_mmps=10250,anchor_mmps=10000,uncertainty_mmps=100,measured_us=950000,
                now_us=1000000,clock_uncertainty_us=0,acceleration_mmps2=2000,assumptions=True)
    values.update(changes)
    return module.select(**values)

def test_safe_candidate_is_actually_accepted():
    r=select()
    assert r["selection"]=="candidate" and r["value_mmps"]==10250
    assert Fraction(r["radius_num"],r["radius_den"])==200
    assert r["conditional_budget_satisfied"] is True
    assert r["physical_validation_established"] is False

def test_clock_uncertainty_can_change_candidate_to_anchor():
    base=select(uncertainty_mmps=100,measured_us=925000,acceleration_mmps2=2000,clock_uncertainty_us=0)
    timed=select(uncertainty_mmps=100,measured_us=925000,acceleration_mmps2=2000,clock_uncertainty_us=1)
    assert base["selection"]=="candidate"
    assert timed["selection"]=="anchor"

def test_age_expansion_can_require_abstention():
    r=select(candidate_mmps=10000,uncertainty_mmps=250,measured_us=900000,acceleration_mmps2=8000)
    assert r["selection"]=="abstain" and r["reason"]=="budget_exhausted"

def test_exact_budget_equality_is_inclusive():
    r=select(candidate_mmps=10250,uncertainty_mmps=250,acceleration_mmps2=0)
    assert r["selection"]=="candidate"

def test_one_microsecond_rational_increment_does_not_round_down():
    r=select(candidate_mmps=10250,uncertainty_mmps=250,measured_us=999999,acceleration_mmps2=1)
    assert r["selection"]=="anchor"

def test_exact_age_limit_is_inclusive():
    assert select(candidate_mmps=10000,measured_us=900000,acceleration_mmps2=0)["selection"]=="candidate"

def test_clock_bound_can_make_nominally_fresh_measurement_stale():
    assert select(measured_us=900000,clock_uncertainty_us=1)["reason"]=="uncertain_age_outside_window"

def test_future_ambiguous_interval_abstains():
    assert select(measured_us=1000000,clock_uncertainty_us=1)["reason"]=="uncertain_age_outside_window"

@pytest.mark.parametrize("change",[
    {"assumptions":False},{"assumptions":None},{"uncertainty_mmps":None},{"clock_uncertainty_us":None},
    {"acceleration_mmps2":None},{"uncertainty_mmps":-1},{"clock_uncertainty_us":True},
    {"acceleration_mmps2":-1},{"measured_us":1000001},{"now_us":True}
])
def test_unknown_or_invalid_assumptions_never_certify(change):
    r=select(**change)
    assert r["selection"]=="abstain"
    assert r["conditional_budget_satisfied"] is False

def test_missing_candidate_falls_back_only_if_anchor_radius_fits():
    assert select(candidate_mmps=None)["selection"]=="anchor"
    assert select(candidate_mmps=None,uncertainty_mmps=500,acceleration_mmps2=2000)["selection"]=="abstain"

def test_untrusted_boolean_speed_does_not_become_a_candidate():
    assert select(candidate_mmps=True)["selection"]=="anchor"
