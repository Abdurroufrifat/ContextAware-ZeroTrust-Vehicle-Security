"""Frozen planar robust point-to-plane registration; inputs are LiDAR only."""
import math
import numpy as np
from scipy.spatial import cKDTree
def downsample(points,times,voxel=.25,maximum=2000):
    cells=np.floor(points/voxel).astype("int64")
    _,indices=np.unique(cells,axis=0,return_index=True)
    if len(indices)>maximum: indices=indices[np.linspace(0,len(indices)-1,maximum,dtype=int)]
    return points[indices],times[indices]
def normals(points):
    tree=cKDTree(points);_,indices=tree.query(points,k=min(16,len(points)),workers=1)
    neighbors=points[indices];centered=neighbors-neighbors.mean(axis=1,keepdims=True)
    covariance=np.einsum("nki,nkj->nij",centered,centered)/neighbors.shape[1]
    values,vectors=np.linalg.eigh(covariance);n=vectors[:,:,0]
    planar=(values[:,1]-values[:,0])/np.maximum(values[:,2],1e-12)
    mask=(np.abs(n[:,2])<=.95)&(planar>=.3)
    return n,mask
def estimate(current,previous,dt,rear_from_lidar):
    result=dict(accepted=False,reason="invalid_input",speed_mps=None,yaw_rate=None)
    if not math.isfinite(dt) or not .07<=dt<=.13: return result
    if current.ndim!=2 or previous.ndim!=2 or current.shape[1]!=3 or previous.shape[1]!=3: return result
    if not np.isfinite(current).all() or not np.isfinite(previous).all(): return result
    if min(len(current),len(previous))<150: result["reason"]="insufficient_points";return result
    cn,cm=normals(current);pn,pm=normals(previous)
    source=current[cm];target=previous[pm];target_normals=pn[pm]
    result.update(source_planar_points=len(source),target_planar_points=len(target))
    if min(len(source),len(target))<150: result["reason"]="insufficient_planar_surfaces";return result
    tree=cKDTree(target);rotation=np.eye(3);translation=np.zeros(3);converged=False
    for iteration in range(15):
        moved=source@rotation.T+translation
        distance,index=tree.query(moved,k=1,workers=1)
        mask=distance<=1.5
        ids=np.flatnonzero(mask)
        if len(ids)<100: result["reason"]="insufficient_correspondences";return result
        nn=target_normals[index[ids]];residual=np.sum(nn*(moved[ids]-target[index[ids]]),axis=1)
        keep=np.argsort(np.abs(residual),kind="stable")[:max(100,math.ceil(.7*len(ids)))]
        ids=ids[keep];nn=nn[keep];residual=residual[keep]
        a=np.column_stack((nn[:,0],nn[:,1],-nn[:,0]*moved[ids,1]+nn[:,1]*moved[ids,0]))
        weight=np.sqrt(np.minimum(1.,.1/np.maximum(np.abs(residual),1e-12)))
        aw=a*weight[:,None];condition=float(np.linalg.cond(aw))
        if not math.isfinite(condition) or condition>1000 or np.linalg.matrix_rank(aw)<3:
            result["reason"]="unobservable_planar_geometry";return result
        step=np.linalg.lstsq(aw,-residual*weight,rcond=None)[0]
        theta=float(step[2]);c,s=math.cos(theta),math.sin(theta)
        update=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        rotation=update@rotation;translation=update@translation+np.array([step[0],step[1],0.])
        if np.linalg.norm(step[:2])<1e-4 and abs(theta)<1e-5: converged=True;break
    moved=source@rotation.T+translation;distance,index=tree.query(moved,k=1,workers=1);mask=distance<=1.5
    residual=np.abs(np.sum(target_normals[index[mask]]*(moved[mask]-target[index[mask]]),axis=1))
    if len(residual)<100: result["reason"]="insufficient_correspondences";return result
    chosen=np.sort(residual)[:max(100,math.ceil(.7*len(residual)))]
    overlap=float(mask.mean());p95=float(np.quantile(chosen,.95))
    lidar_pose=np.eye(4);lidar_pose[:3,:3]=rotation;lidar_pose[:3,3]=translation
    pose=rear_from_lidar@lidar_pose@np.linalg.inv(rear_from_lidar)
    speed=float(np.linalg.norm(pose[:2,3])/dt);yaw=math.atan2(pose[1,0],pose[0,0])/dt
    result.update(overlap=overlap,residual_p95_m=p95,condition=condition,iterations=iteration+1,
        converged=converged,lidar_pose=lidar_pose.tolist(),rear_pose=pose.tolist(),correspondences=int(mask.sum()))
    if not converged: result["reason"]="registration_not_converged";return result
    if overlap<.5 or p95>.3: result["reason"]="poor_scene_registration";return result
    if speed>60 or abs(yaw)>2: result["reason"]="implausible_motion";return result
    result.update(accepted=True,reason="lidar_interval_motion",speed_mps=speed,yaw_rate=yaw)
    return result
