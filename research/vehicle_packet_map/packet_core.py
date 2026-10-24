"""Research-only packet-window odometry; geometric screens are not error certificates."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
CONFIG=dict(out_dir="results",data=dict(max_range=100.,min_range=0.,deskew=True),
 mapping=dict(voxel_size=1.,max_points_per_voxel=20),
 registration=dict(max_num_iterations=500,convergence_criterion=.0001,max_num_threads=4),
 adaptive_threshold=dict(fixed_threshold=None,initial_threshold=2.,min_motion_th=.1))
def scheduled_start(next_due,worker):return max(next_due,worker)
def window_mask(times,now,span):return (times>=now-span)&(times<=now)
def skew(v):
 x,y,z=v
 return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=float)
def left_jac(w):
 theta=float(np.linalg.norm(w));a=skew(w)
 if theta<1e-7:return np.eye(3)+.5*a+(a@a)/6
 return np.eye(3)+(1-np.cos(theta))/theta**2*a+(theta-np.sin(theta))/theta**3*(a@a)
def scale_motion(delta,factor):
 if not np.isfinite(delta).all() or not np.isfinite(factor) or factor<0:raise ValueError("motion")
 w=Rotation.from_matrix(delta[:3,:3]).as_rotvec()
 v=np.linalg.solve(left_jac(w),delta[:3,3]);out=np.eye(4)
 out[:3,:3]=Rotation.from_rotvec(w*factor).as_matrix()
 out[:3,3]=left_jac(w*factor)@(v*factor)
 return out
def interval(previous,current,previous_end,end,mount):
 dt=(end-previous_end)/1e9
 if not .005<=dt<=.2:raise ValueError("unsupported_interval")
 delta=mount@np.linalg.inv(previous)@current@np.linalg.inv(mount)
 speed=float(np.linalg.norm(delta[:3,3])/dt)
 if not np.isfinite(speed) or speed>60:raise ValueError("implausible_motion")
 return dict(speed_mps=speed,dt_seconds=dt,midpoint_ns=(previous_end+end)//2,relative_rear_pose=delta.tolist())
def geometry(source,target,pose):
 out=dict(passed=False,correspondences=0,overlap=0.,residual_p95_m=None,condition=None,uncertainty_certified=False)
 if min(len(source),len(target))<150 or not np.isfinite(pose).all():return out
 moved=source@pose[:3,:3].T+pose[:3,3];dist,ids=cKDTree(target).query(moved,workers=1)
 mask=dist<=1.5;count=int(mask.sum());overlap=float(mask.mean())
 out.update(correspondences=count,overlap=overlap)
 if count<100:return out
 points=moved[mask]
 # Point-to-point local pose Jacobian: observability under fixed correspondences only.
 design=np.zeros((len(points),3,6));design[:,:,:3]=np.eye(3)
 design[:,0,4]=points[:,2];design[:,0,5]=-points[:,1]
 design[:,1,3]=-points[:,2];design[:,1,5]=points[:,0]
 design[:,2,3]=points[:,1];design[:,2,4]=-points[:,0]
 singular=np.linalg.svd(design.reshape(-1,6),compute_uv=False)
 condition=float(singular[0]/singular[-1]) if singular[-1]>1e-12 else None
 p95=float(np.quantile(dist[mask],.95))
 out.update(residual_p95_m=p95,condition=condition)
 out["passed"]=bool(overlap>=.5 and p95<=.3 and condition is not None and condition<=1000)
 return out
class PacketMap:
 def __init__(self):
  self.model=KissICP(KISSConfig(**CONFIG));self.end_ns=None;self.last_dt_ns=None;self.updates=0
 def step(self,points,offsets,end,mount):
  model=self.model;prior=model.last_pose.copy();prior_count=self.updates
  if len(points)<150 or not np.isfinite(points).all():return dict(bootstrap=False,quality_passed=False,reason="insufficient_points",motion=None,prior_updates=prior_count)
  if self.end_ns is None:
   frame=model.preprocessor.preprocess(points,offsets,np.eye(4))
   source,down=model.voxelize(frame);model.local_map.update(down,np.eye(4))
   self.end_ns=end;self.updates+=1
   return dict(bootstrap=True,quality_passed=False,reason="bootstrap",motion=None,prior_updates=prior_count,pose=prior.tolist(),previous_pose=None,previous_end_ns=None,map_points_before=0)
  dt_ns=end-self.end_ns
  if not 5000000<=dt_ns<=200000000:return dict(bootstrap=False,quality_passed=False,reason="unsupported_interval",motion=None,prior_updates=prior_count)
  span_ns=(float(np.max(offsets))-float(np.min(offsets)))*1e9
  deskew=scale_motion(model.last_delta,span_ns/self.last_dt_ns) if self.last_dt_ns else np.eye(4)
  prediction=scale_motion(model.last_delta,dt_ns/self.last_dt_ns) if self.last_dt_ns else np.eye(4)
  frame=model.preprocessor.preprocess(points,offsets,deskew);source,down=model.voxelize(frame)
  target=model.local_map.point_cloud()
  sigma=model.adaptive_threshold.get_threshold();guess=prior@prediction
  pose=model.registration.align_points_to_map(points=source,voxel_map=model.local_map,initial_guess=guess,max_correspondance_distance=3*sigma,kernel=sigma)
  quality=geometry(source,target,pose)
  self.last_source=source;self.last_target=target
  try:motion=interval(prior,pose,self.end_ns,end,mount)
  except ValueError:motion=None
  passed=bool(quality["passed"] and motion is not None)
  result=dict(bootstrap=False,quality_passed=passed,reason="geometry_passed" if passed else "geometry_or_motion_rejected",motion=motion,geometry=quality,
   prior_updates=prior_count,previous_end_ns=self.end_ns,previous_pose=prior.tolist(),pose=pose.tolist(),
   map_points_before=len(target),source_points=len(source),prediction=prediction.tolist(),deskew_motion=deskew.tolist(),
   uncertified_motion_assumption="constant SE3 twist from prior accepted interval; variable-duration scaling")
  if passed:
   model.adaptive_threshold.update_model_deviation(np.linalg.inv(guess)@pose)
   model.local_map.update(down,pose)
   model.last_delta=np.linalg.inv(prior)@pose;model.last_pose=pose
   self.last_dt_ns=dt_ns;self.end_ns=end;self.updates+=1
  return result
