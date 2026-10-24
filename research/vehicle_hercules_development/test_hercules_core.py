from pathlib import Path
import importlib.util,struct,numpy as np,pytest
def core():
 p=Path(__file__).with_name("hercules_core.py")
 assert p.exists(),"HeRCULES decoder and causal radar/IMU fit are not implemented"
 spec=importlib.util.spec_from_file_location("hercules_core",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_packed_29byte_schema_not_aligned32():
 b=struct.pack("<fffffBff",1,2,3,-4,5,247,.6,.7)
 a=core().decode_radar(b)
 assert len(a)==1 and a["v"][0]==-4 and a["rcs"][0]==247
@pytest.mark.parametrize("size",[1,28,30,58+1])
def test_partial_records_rejected(size):
 with pytest.raises(ValueError):core().decode_radar(b"\0"*size)
def scene(v=(7.,-1.)):
 theta=np.linspace(-1.,1.,100);el=np.linspace(-.2,.2,100)
 n=np.column_stack([np.cos(el)*np.cos(theta),np.cos(el)*np.sin(theta),np.sin(el)])
 return n,-n[:,:2]@np.array(v)
def test_known_velocity_is_recovered_without_reference():
 n,d=scene();out=core().fit_planar(n,d,np.eye(3))
 assert out["quality"];np.testing.assert_allclose(out["velocity"],[7,-1],atol=1e-10)
def test_huber_reduces_large_minority_outlier_effect():
 n,d=scene();d[::10]+=8
 out=core().fit_planar(n,d,np.eye(3))
 assert out["quality"];assert abs(out["speed"]-np.sqrt(50))<.1
def test_unobservable_geometry_is_rejected():
 n=np.tile([1.,0.,0.],(30,1));d=np.ones(30)
 assert not core().fit_planar(n,d,np.eye(3))["quality"]
def test_orientation_levels_tilted_radar():
 from scipy.spatial.transform import Rotation
 q=Rotation.from_euler("y",20,degrees=True).as_matrix()
 n,d=scene();radar=n@q
 out=core().fit_planar(radar,d,q)
 assert out["quality"];np.testing.assert_allclose(out["speed"],np.sqrt(50),atol=1e-9)
def test_imu_selection_never_uses_future_or_stale():
 m=core();assert m.latest_imu(150,[100,200],100)==0
 assert m.latest_imu(99,[100,200],100) is None
 assert m.latest_imu(301,[100,200],100) is None
def test_bad_quaternion_is_rejected():
 with pytest.raises(ValueError):core().quaternion_matrix([0,0,0,0])
def test_quaternion_xyzw_identity():
 np.testing.assert_array_equal(core().quaternion_matrix([0,0,0,1]),np.eye(3))

def test_stream_archive_keeps_all_timestamp_members(tmp_path):
 import tarfile,io
 p=tmp_path/"frames.tar.gz"
 with tarfile.open(p,"w:gz") as t:
  for stamp,value in [(200,b"b"),(100,b"a")]:
   member=tarfile.TarInfo(f"continental/{stamp}.bin");member.size=len(value);t.addfile(member,io.BytesIO(value))
 assert core().load_archive(p)=={200:b"b",100:b"a"}
def test_stream_archive_rejects_unsafe_member(tmp_path):
 import tarfile,io
 p=tmp_path/"unsafe.tar.gz"
 with tarfile.open(p,"w:gz") as t:
  member=tarfile.TarInfo("../100.bin");member.size=1;t.addfile(member,io.BytesIO(b"a"))
 with pytest.raises(ValueError):core().load_archive(p)
