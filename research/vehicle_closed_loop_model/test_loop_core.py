from pathlib import Path
import importlib.util
from fractions import Fraction as F
import pytest

def module():
    path=Path(__file__).with_name("loop_core.py")
    assert path.exists(),"closed-loop model not implemented"
    spec=importlib.util.spec_from_file_location("loop_under_test",path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def test_constant_acceleration_exact_velocity_and_distance():
    speed,position=module().advance(F(10000),F(0),F(2000),1000000)
    assert speed==12000 and position==11000

def test_coasting_preserves_speed():
    speed,position=module().advance(F(5000),F(10),F(0),50000)
    assert speed==5000 and position==260

def test_stop_inside_step_does_not_move_backwards_or_overbrake():
    speed,position=module().advance(F(1000),F(0),F(-4000),1000000)
    assert speed==0 and position==125

def test_stationary_braking_preserves_position():
    assert module().advance(F(0),F(80),F(-2000),50000)==(0,80)

@pytest.mark.parametrize("values",[
    (-1,0,0,50000),(1,-1,0,50000),(1,0,2001,50000),(1,0,-4001,50000),(1,0,0,0)])
def test_invalid_plant_inputs_rejected(values):
    with pytest.raises(ValueError):module().advance(*values)

def test_causal_fractional_history_sample():
    assert module().speed_at([(0,F(10000),F(2000))],25000,50000,F(10100))==10050

def test_history_sample_clamps_after_stop():
    history=[(i*50000,F(max(0,100-200*i)),F(-4000)) for i in range(2)]
    assert module().speed_at(history,40000,100000,F(0))==0

def test_history_cannot_read_future():
    with pytest.raises(ValueError):module().speed_at([],50001,50000,F(10))

def test_history_before_start_is_missing():
    assert module().speed_at([], -1,0,F(10)) is None

def test_current_sample_uses_current_state():
    assert module().speed_at([],0,0,F(10000))==10000

@pytest.mark.parametrize("estimate,expected",[(None,-2000),(0,2000),(10000,0),(20000,-4000)])
def test_controller_bounds_and_abstention(estimate,expected):
    assert module().command(10000,estimate)==expected

def test_controller_has_negative_feedback():
    assert module().command(15000,10000)>0
    assert module().command(10000,15000)<0

def test_exact_command_boundaries():
    m=module()
    assert [m.target(t) for t in (0,7999999,8000000,17999999,18000000)]==[15000,15000,22000,22000,10000]

def test_feedback_changes_next_state_and_reduces_constant_target_error():
    m=module();v=F(10000);x=F(0)
    for _ in range(40):
        v,x=m.advance(v,x,m.command(15000,round(v)),50000)
    assert 10000<v<15000 and x>0
