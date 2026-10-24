from pathlib import Path
import importlib.util,struct,numpy as np,pytest
def core():
 p=Path(__file__).with_name("crossmodal_core.py")
 assert p.exists(),"optical Doppler and cross-modal agreement adapter missing"
 spec=importlib.util.spec_from_file_location("crossmodal_core",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def scene(v=(8.,1.,.2)):
 rng=np.random.default_rng(42);n=rng.normal(size=(512,3));n[:,0]=abs(n[:,0])+1;n/=np.linalg.norm(n,axis=1)[:,None]
 return n,-n@np.array(v)
def test_optical_29byte_schema_has_offset_and_intensity():
 b=struct.pack("<fffffIBf",1,2,3,4,-5,123456,7,8)
 a=core().decode_optical(b);assert a["velocity"][0]==-5 and a["time_offset"][0]==123456 and a["line"][0]==7 and a["intensity"][0]==8
@pytest.mark.parametrize("b",[b"a",b"\0"*28,b"\0"*30])
def test_truncated_optical_record_rejected(b):
 with pytest.raises(ValueError):core().decode_optical(b)
def test_known_full3D_velocity_recovered():
 n,d=scene();out=core().fit_velocity(n,d);assert out["quality"]
 np.testing.assert_allclose(out["velocity"],[8,1,.2],atol=1e-10)
def test_full3D_forgery_retains_quality():
 n,d=scene();delta=np.array([3,0,0]);a=core().fit_velocity(n,d-n@delta)
 assert a["quality"];np.testing.assert_allclose(a["velocity"],[11,1,.2],atol=1e-10)
def test_unobservable_plane_rejected():
 n=np.tile([1,0,0],(30,1));assert not core().fit_velocity(n,np.ones(30))["quality"]
def test_lever_arm_velocity_uses_cross_product_and_inverse_translation():
 out=core().radar_origin_velocity([5,0,0],np.eye(3),[-2,0,0],np.eye(3),[0,0,1])
 np.testing.assert_allclose(out,[5,2,0])
def test_one_modality_forgery_rejected():
 m=core();assert m.agree(8,8,0,0,True,True)
 assert not m.agree(11,8,0,0,True,True)
 assert not m.agree(8,11,0,0,True,True)
def test_common_mode_shift_remains_unobservable():
 assert core().agree(11,11,0,0,True,True)
def test_fixed_inclusive_age_and_agreement_boundaries():
 m=core();assert m.agree(8,9,100_000_000,100_000_000,True,True)
 assert not m.agree(8,9.000001,0,0,True,True)
 assert not m.agree(8,8,0,100_000_001,True,True)
@pytest.mark.parametrize("speed,age,quality",[(np.nan,0,True),(8,-1,True),(8,0,False)])
def test_nonfinite_future_or_invalid_source_rejected(speed,age,quality):
 assert not core().agree(8,speed,0,age,True,quality)
