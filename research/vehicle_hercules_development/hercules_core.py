"""Radar/IMU-only estimator: no reference inputs, labels or policy promotion."""
import numpy as np
from scipy.spatial.transform import Rotation
RADAR_DTYPE=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("v","<f4"),("range","<f4"),("rcs","u1"),("azimuth","<f4"),("elevation","<f4")],align=False)
def decode_radar(b):
 if len(b)%29:raise ValueError("partial radar record")
 return np.frombuffer(b,dtype=RADAR_DTYPE)
def quaternion_matrix(q):
 q=np.asarray(q,dtype=float)
 if q.shape!=(4,) or not np.all(np.isfinite(q)) or abs(np.linalg.norm(q)-1)>.02:raise ValueError("invalid xyzw quaternion")
 return Rotation.from_quat(q).as_matrix()
def latest_imu(t,times,max_age=100_000_000):
 a=np.asarray(times,dtype=np.int64)
 i=int(np.searchsorted(a,t,side="right"))-1
 return i if i>=0 and 0<=t-int(a[i])<=max_age else None
def fit_planar(directions,doppler,radar_to_level):
 n=np.asarray(directions,dtype=float);d=np.asarray(doppler,dtype=float)
 fail={"quality":False,"speed":None,"reason":"invalid_geometry"}
 if n.ndim!=2 or n.shape[1]!=3 or d.shape!=(len(n),) or len(n)<20:return fail
 if not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)):return fail
 A=(n@np.asarray(radar_to_level,dtype=float).T)[:,:2]
 if np.linalg.cond(A.T@A)>100:return fail
 v=np.linalg.lstsq(A,-d,rcond=None)[0];w=np.ones(len(d))
 for _ in range(20):
  residual=A@v+d
  w=np.minimum(1.,.3/np.maximum(abs(residual),1e-12))
  sw=np.sqrt(w);v=np.linalg.lstsq(A*sw[:,None],-d*sw,rcond=None)[0]
 residual=A@v+d;inside=abs(residual)<=.3
 condition=float(np.linalg.cond(A.T@(w[:,None]*A)))
 frac=float(np.mean(inside));rmse=float(np.sqrt(np.mean(residual[inside]**2))) if np.any(inside) else None
 quality=bool(condition<=100 and frac>=.5 and rmse is not None and rmse<=.2)
 return {"speed":float(np.linalg.norm(v)),"velocity":v.tolist(),"quality":quality,"condition":condition,"inlier_fraction":frac,"inlier_rmse":rmse,"points":len(d),"reason":"ready" if quality else "residual_quality"}

def load_archive(path):
 import tarfile
 from pathlib import PurePosixPath
 result={};total=0
 with tarfile.open(path,"r|gz") as archive:
  for member in archive:
   if member.isdir():continue
   p=PurePosixPath(member.name)
   if not member.isfile() or p.is_absolute() or ".." in p.parts or p.suffix!=".bin" or member.size>=1024*1024:raise ValueError("unsafe radar archive member")
   stamp=int(p.stem)
   if stamp in result:raise ValueError("duplicate radar timestamp")
   total+=member.size
   if total>128*1024*1024:raise ValueError("archive byte cap")
   result[stamp]=archive.extractfile(member).read()
 return result
