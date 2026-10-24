from pathlib import Path
import importlib.util
import numpy as np
import pytest

def fit(a,d,dimensions,robust):
    path=Path(__file__).with_name("equal_input_core.py")
    assert path.exists(), "equal-input fitting component not implemented"
    spec=importlib.util.spec_from_file_location("equal_input_under_test",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.fit_motion(a,d,dimensions,robust)

def rays():
    a=np.random.default_rng(714).normal(size=(80,3))
    return a/np.linalg.norm(a,axis=1)[:,None]

@pytest.mark.parametrize("dimensions,robust",[(2,False),(2,True),(3,False),(3,True)])
def test_known_velocity_recovers_five(dimensions,robust):
    a=rays(); v=np.array([3.,4.,0.]); d=-(a@v)
    result=fit(a,d,dimensions,robust)
    assert result["reason"]=="numeric"
    assert result["speed"]==pytest.approx(5.,abs=1e-10)
    np.testing.assert_allclose(result["velocity"],v[:dimensions],atol=1e-10)

def test_spatial_recovers_vertical_motion_without_planar_assumption():
    a=rays(); v=np.array([1.,2.,3.]); d=-(a@v)
    full=fit(a,d,3,False); planar=fit(a,d,2,False)
    np.testing.assert_allclose(full["velocity"],v,atol=1e-10)
    assert full["speed"]==pytest.approx(np.sqrt(14))
    assert abs(planar["speed"]-full["speed"])>0.1

def test_huber_reduces_sparse_large_radial_outlier_effect():
    a=rays(); v=np.array([3.,4.,0.]); d=-(a@v);d[:8]+=30
    plain=fit(a,d,2,False); robust=fit(a,d,2,True)
    plain_error=np.linalg.norm(np.array(plain["velocity"])-v[:2])
    robust_error=np.linalg.norm(np.array(robust["velocity"])-v[:2])
    assert robust_error<plain_error/5
    assert robust["iterations"]==20

def test_huber_uses_prespecified_twenty_iterations_even_at_exact_solution():
    a=rays(); result=fit(a,-a@np.array([3.,4.,0.]),2,True)
    assert result["iterations"]==20
    assert result["stationarity_inf"]<1e-9

@pytest.mark.parametrize("kind",["nonfinite","shape","short","rank","dimensions"])
def test_invalid_geometry_is_visible(kind):
    a=rays();d=np.zeros(len(a));dim=2
    if kind=="nonfinite":d[0]=np.nan
    if kind=="shape":d=d[:-1]
    if kind=="short":a=a[:5];d=d[:5]
    if kind=="rank":a[:]=[1,0,0]
    if kind=="dimensions":dim=4
    result=fit(a,d,dim,False)
    assert result["speed"] is None
    assert result["reason"]!="numeric"

def test_high_condition_is_diagnostic_without_quality_selection():
    t=np.linspace(-0.001,0.001,80);a=np.column_stack([np.cos(t),np.sin(t),np.zeros(80)])
    result=fit(a,-a@np.array([3.,4.,0.]),2,False)
    assert result["condition"]>100
    assert result["reason"]=="numeric"
    assert result["speed"]==pytest.approx(5,abs=1e-8)

def test_speed_norm_and_fit_do_not_change_under_horizontal_yaw():
    a=rays();v=np.array([3.,4.,0.]);d=-a@v
    angle=.7;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    for robust in (False,True):
        base=fit(a,d,2,robust);turned=fit(a@r.T,d,2,robust)
        assert turned["speed"]==pytest.approx(base["speed"],abs=1e-10)
