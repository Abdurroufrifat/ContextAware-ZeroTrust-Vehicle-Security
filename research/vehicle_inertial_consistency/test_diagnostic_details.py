from pathlib import Path
import importlib.util,numpy as np,pytest
def core():
    path=Path(__file__).with_name("diagnostic_details.py")
    assert path.exists(),"diagnostic_details implementation missing"
    spec=importlib.util.spec_from_file_location("diagnostic_details_tested",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_additive_imu_forgery_preserves_nominal_mismatch():
    m=core();d=np.array([[0.,0],[.4,0]]);bias=np.array([[0.,0],[3,0]]);j=np.array([0,0])
    forged=m.additive_forgery(d,bias,j)
    np.testing.assert_allclose(forged,[[0,0],[3.4,0]])
    np.testing.assert_allclose(np.linalg.norm(bias[1]-bias[0]-forged[1]),.4)
@pytest.mark.parametrize("supported,accepted,good,duration,reason",[
    (False,False,True,1_000_000_000,"stale_imu"),
    (True,False,True,1_000_000_000,"innovation_exceeds_1"),
    (True,True,True,1_000_000_000,"accepted"),
    (False,False,False,1_000_000_000,"radar_quality"),
    (False,False,True,1_200_000_000,"long_interval")])
def test_support_reasons_preserve_original_failure_priority(supported,accepted,good,duration,reason):
    m=core();out=m.explain_frames([0,duration],[-1,0],[True,good],[False,supported],[False,accepted])
    assert list(out)==["missing_history",reason]
def test_accepted_without_support_is_invalid():
    with pytest.raises(ValueError):core().explain_frames([0,1_000_000_000],[-1,0],[True,True],[False,False],[False,True])
