from pathlib import Path
import sys,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"vehicle_hercules_development"))
from hercules_core import fit_planar,quaternion_matrix,latest_imu
def test_coherent_velocity_shift_preserves_residual_guard():
 theta=np.linspace(-1,1,100);n=np.column_stack([np.cos(theta),np.sin(theta),np.zeros(100)])
 v=np.array([8.,1.]);d=-n[:,:2]@v;delta=3*v/np.linalg.norm(v)
 base=fit_planar(n,d,np.eye(3));forged=fit_planar(n,d-n[:,:2]@delta,np.eye(3))
 assert base["quality"] and forged["quality"]
 np.testing.assert_allclose(forged["speed"]-base["speed"],3.,atol=1e-10)
 np.testing.assert_allclose(forged["inlier_rmse"],base["inlier_rmse"],atol=1e-10)
def test_expired_context_rejected_at_fixed_limit():
 assert latest_imu(100_000_000,[0])==0
 assert latest_imu(100_000_001,[0]) is None
def test_future_context_not_used():
 assert latest_imu(1,[2]) is None
def test_context_rotation_spoof_can_be_a_valid_quaternion():
 from scipy.spatial.transform import Rotation
 q=Rotation.from_euler("y",30,degrees=True).as_quat()
 assert np.all(np.isfinite(quaternion_matrix(q)))
@pytest.mark.parametrize("q",[[0,0,0,0],[0,0,0,2],[np.nan,0,0,1]])
def test_corrupt_quaternion_rejected(q):
 with pytest.raises(ValueError):quaternion_matrix(q)
