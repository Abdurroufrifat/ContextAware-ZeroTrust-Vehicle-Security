from pathlib import Path
import importlib.util,numpy as np,pytest
def core():
    p=Path(__file__).with_name("inertial_core.py")
    assert p.exists(),"inertial_core implementation missing"
    s=importlib.util.spec_from_file_location("inertial_tested",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_horizontal_projection_omits_vertical_gravity():
    m=core();q=np.array([[0,0,0,1.]])
    np.testing.assert_array_equal(m.acceleration_world_xy(q,[[2,3,9.81]]),[[2,3]])
def test_orientation_rotation_is_applied():
    m=core();q=np.array([[0,0,np.sqrt(.5),np.sqrt(.5)]])
    np.testing.assert_allclose(m.acceleration_world_xy(q,[[2,0,9.81]]),[[0,2]],atol=1e-14)
def test_sensor_origin_correction_sign():
    m=core()
    v=m.imu_origin_velocity([[1,2]],[[0,0,0,1]],[[0,0,1]],[1,0,0])
    np.testing.assert_array_equal(v,[[1,1]])
def test_integrator_holds_past_sample_without_future_interpolation():
    m=core();t=np.array([0,50_000_000,100_000_000])
    out=m.integrate_causal(t,np.array([[1.,0],[3,0],[999,0]]),25_000_000,75_000_000)
    assert out["supported"];np.testing.assert_allclose(out["delta"],[.1,0],atol=1e-15)
def test_endpoint_age_100ms_is_inclusive():
    m=core();out=m.integrate_causal([0,200_000_000],[[2,0],[999,0]],0,100_000_000)
    assert out["supported"];np.testing.assert_allclose(out["delta"],[.2,0])
def test_stale_endpoint_rejected():
    out=core().integrate_causal([0,200_000_000],[[2,0],[0,0]],0,100_000_001)
    assert not out["supported"]
def test_internal_held_sample_age_cannot_exceed_bound():
    m=core();out=m.integrate_causal([0,150_000_000,160_000_000],[[1,0],[1,0],[1,0]],90_000_000,160_000_000)
    assert not out["supported"]
def test_missing_initial_history_cannot_integrate():
    assert not core().integrate_causal([20,30],[[0,0],[0,0]],0,30)["supported"]
@pytest.mark.parametrize("times",[[1,1],[2,1]])
def test_invalid_time_order_rejected(times):
    with pytest.raises(ValueError):core().integrate_causal(times,[[0,0],[0,0]],1,2)
def test_nonfinite_acceleration_rejected():
    with pytest.raises(ValueError):core().integrate_causal([0,10],[[0,0],[np.nan,0]],0,10)
def test_one_second_history_is_required():
    m=core();t=np.array([0,500_000_000,1_000_000_000]);mt=np.arange(101)*10_000_000
    out=m.innovations(np.zeros((3,2)),t,mt,np.zeros((101,2)),np.ones(3,bool))
    np.testing.assert_array_equal(out["supported"],[False,False,True])
    np.testing.assert_array_equal(out["past_index"],[-1,-1,0])
def test_inclusive_one_mps_statistic_boundary():
    m=core();out=m.compare_changes([[0,0],[1,0]],[0,0],[[0,0],[0,0]],[False,True])
    np.testing.assert_array_equal(out["accepted"],[False,True])
def test_constant_velocity_offset_cancels_exactly():
    m=core();v=np.array([[1.,0],[2,1],[3,2]]);j=np.array([0,0,1]);d=np.array([[0.,0],[1,1],[1,1]]);ok=np.array([False,True,True])
    a=m.compare_changes(v,j,d,ok);b=m.compare_changes(v+[3,0],j,d,ok)
    np.testing.assert_allclose(a["residual"],b["residual"],equal_nan=True,atol=1e-14)
def test_step_detected_then_constant_offset_is_invisible():
    m=core();v=np.array([[0.,0],[3,0],[3,0]]);out=m.compare_changes(v,[0,0,1],np.zeros((3,2)),[False,True,True])
    np.testing.assert_array_equal(out["accepted"],[False,False,True])
def test_slow_ramp_and_forged_imu_are_not_overclaimed():
    m=core();v=np.array([[0.,0],[.3,0],[.6,0]]);j=[0,0,1];support=[False,True,True]
    out=m.compare_changes(v,j,np.zeros((3,2)),support)
    np.testing.assert_array_equal(out["accepted"],[False,True,True])
    step=np.array([[0.,0],[3,0],[3,0]]);spoof=step-step[np.array(j)]
    np.testing.assert_array_equal(m.compare_changes(step,j,spoof,support)["accepted"],[False,True,True])
def test_missing_current_radar_quality_cannot_admit():
    m=core();t=np.array([0,1_000_000_000]);mt=np.arange(101)*10_000_000
    out=m.innovations(np.zeros((2,2)),t,mt,np.zeros((101,2)),[True,False])
    assert not out["supported"].any()
