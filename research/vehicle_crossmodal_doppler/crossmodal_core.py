"""Independent optical Doppler fit and fixed conditional agreement check."""
import numpy as np
OPTICAL_DTYPE=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("reflectivity","<f4"),("velocity","<f4"),("time_offset","<u4"),("line","u1"),("intensity","<f4")],align=False)
def decode_optical(b):
 if len(b)%29:raise ValueError("partial optical record")
 return np.frombuffer(b,dtype=OPTICAL_DTYPE)
def fit_velocity(directions,doppler):
 A=np.asarray(directions,dtype=float);d=np.asarray(doppler,dtype=float)
 fail={"quality":False,"velocity":None,"reason":"invalid_geometry"}
 if A.ndim!=2 or A.shape[1]!=3 or d.shape!=(len(A),) or len(A)<20:return fail
 if not np.all(np.isfinite(A)) or not np.all(np.isfinite(d)) or np.linalg.cond(A.T@A)>1000:return fail
 v=np.linalg.lstsq(A,-d,rcond=None)[0];w=np.ones(len(d))
 for _ in range(20):
  residual=A@v+d;w=np.minimum(1.,.3/np.maximum(abs(residual),1e-12));sw=np.sqrt(w)
  v=np.linalg.lstsq(A*sw[:,None],-d*sw,rcond=None)[0]
 residual=A@v+d;inside=abs(residual)<=.3
 condition=float(np.linalg.cond(A.T@(w[:,None]*A)));frac=float(np.mean(inside))
 rmse=float(np.sqrt(np.mean(residual[inside]**2))) if np.any(inside) else None
 quality=bool(condition<=1000 and frac>=.5 and rmse is not None and rmse<=.2)
 return {"quality":quality,"velocity":v.tolist(),"condition":condition,"inlier_fraction":frac,"inlier_rmse":rmse,"points":len(d),"reason":"ready" if quality else "residual_quality"}
def radar_origin_velocity(v_l,R_radar_from_lidar,t_radar_from_lidar,R_imu_from_lidar,gyro_imu):
 R=np.asarray(R_radar_from_lidar,dtype=float);t=np.asarray(t_radar_from_lidar,dtype=float)
 arm=-R.T@t;omega=np.asarray(R_imu_from_lidar,dtype=float).T@np.asarray(gyro_imu,dtype=float)
 return R@(np.asarray(v_l,dtype=float)+np.cross(omega,arm))
def agree(radar_speed,optical_speed,radar_age_ns,optical_age_ns,radar_quality,optical_quality):
 values=[radar_speed,optical_speed,radar_age_ns,optical_age_ns]
 if any(x is None for x in values) or not np.all(np.isfinite(values)):return False
 return bool(radar_quality and optical_quality and 0<=radar_age_ns<=100_000_000 and 0<=optical_age_ns<=100_000_000 and abs(radar_speed-optical_speed)<=1.)
def estimate_optical(b,R,t,R_imu,gyro):
 pc=decode_optical(b)
 index=np.linspace(0,len(pc)-1,min(512,len(pc)),dtype=int) if len(pc) else np.empty(0,dtype=int)
 sample=pc[index];xyz=np.column_stack([sample[k] for k in ["x","y","z"]]).astype(float);d=sample["velocity"].astype(float)
 length=np.linalg.norm(xyz,axis=1);valid=np.all(np.isfinite(xyz),axis=1)&np.isfinite(d)&(length>=3)&(length<=100)
 A=xyz[valid]/length[valid,None];d=d[valid];out=fit_velocity(A,d)
 if out["velocity"] is not None:
  vr=radar_origin_velocity(out["velocity"],R,t,R_imu,gyro);out["radar_origin_velocity"]=vr.tolist();out["speed"]=float(np.linalg.norm(vr))
 else:out["radar_origin_velocity"]=None;out["speed"]=None
 out["source_points"]=len(pc);out["sampled_points"]=len(sample)
 return out
