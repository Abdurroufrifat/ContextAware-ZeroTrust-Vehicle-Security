"""Pinned compiled ICP variants with explicit shared research-quality checks."""
import math
import numpy as np
from scipy.spatial import cKDTree
import small_gicp
def estimate(current,previous,dt,rear_from_lidar,method):
    out=dict(accepted=False,reason="invalid_input",speed_mps=None,yaw_rate=None)
    if method not in ("ICP","PLANE_ICP","GICP") or dt is None or not math.isfinite(dt) or not .07<=dt<=.13: return out
    if current.ndim!=2 or previous.ndim!=2 or current.shape[1]!=3 or previous.shape[1]!=3: return out
    if not np.isfinite(current).all() or not np.isfinite(previous).all(): return out
    if min(len(current),len(previous))<150: out["reason"]="insufficient_points";return out
    target,tree=small_gicp.preprocess_points(np.asarray(previous,dtype="float64"),downsampling_resolution=.25,num_neighbors=16,num_threads=1)
    source,_=small_gicp.preprocess_points(np.asarray(current,dtype="float64"),downsampling_resolution=.25,num_neighbors=16,num_threads=1)
    tp=np.asarray(target.points())[:,:3];sp=np.asarray(source.points())[:,:3];tn=np.asarray(target.normals())[:,:3]
    out.update(target_points=len(tp),source_points=len(sp))
    if min(len(tp),len(sp))<150: out["reason"]="insufficient_preprocessed_points";return out
    jac=np.column_stack((tn[:,0],tn[:,1],-tn[:,0]*tp[:,1]+tn[:,1]*tp[:,0]))
    condition=float(np.linalg.cond(jac))
    out["reference_planar_condition"]=condition if math.isfinite(condition) else None
    if not math.isfinite(condition) or condition>1000 or np.linalg.matrix_rank(jac)<3:
        out["reason"]="unobservable_planar_geometry";return out
    result=small_gicp.align(target,source,tree,init_T_target_source=np.eye(4),registration_type=method,
        max_correspondence_distance=1.5,num_threads=1,max_iterations=20,
        rotation_epsilon=.0017453292519943296,translation_epsilon=.001,verbose=False)
    pose=np.asarray(result.T_target_source);hessian=np.asarray(result.H)
    if not np.isfinite(pose).all() or not np.isfinite(hessian).all():
        out["reason"]="nonfinite_backend_result";return out
    moved=sp@pose[:3,:3].T+pose[:3,3]
    dist,_=cKDTree(tp).query(moved,k=1,workers=1);mask=dist<=1.5
    p95=float(np.quantile(dist[mask],.95)) if mask.any() else None
    rear=rear_from_lidar@pose@np.linalg.inv(rear_from_lidar)
    speed=float(np.linalg.norm(rear[:2,3])/dt);yaw=math.atan2(rear[1,0],rear[0,0])/dt
    out.update(converged=bool(result.converged),native_inliers=int(result.num_inliers),iterations=int(result.iterations),
        overlap=float(mask.mean()),inliers=int(mask.sum()),euclidean_inlier_p95_m=p95,
        lidar_pose=pose.tolist(),rear_pose=rear.tolist(),hessian=hessian.tolist())
    if not result.converged: out["reason"]="native_not_converged";return out
    if int(result.num_inliers)<100 or mask.sum()<100 or mask.mean()<.5 or p95 is None or p95>.3:
        out["reason"]="insufficient_registration_quality";return out
    if speed>60 or abs(yaw)>2: out["reason"]="implausible_motion";return out
    out.update(accepted=True,reason="compiled_"+method+"_interval_motion",speed_mps=speed,yaw_rate=yaw)
    return out
