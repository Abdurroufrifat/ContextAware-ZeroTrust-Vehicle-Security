import pytest
def api():
 from sg_evolution import step_bound
 return step_bound
def test_standstill_reflection_respects_negative_command():
 assert api()(0.,0.,-4000,100000)
def test_realized_speed_can_violate_submitted_input_premise():
 assert not api()(10.,10.3,0,100000)
def test_positive_transition_inside_declared_bound():
 assert api()(10.,10.15,1000,100000)
def test_global_acceleration_clip_is_enforced():
 assert not api()(10.,10.25,2000,100000)
def test_invalid_transition_time_rejected():
 with pytest.raises(ValueError):api()(10.,10.,0,0)
