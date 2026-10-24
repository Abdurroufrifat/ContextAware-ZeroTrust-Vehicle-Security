"""Empirical quality estimation; no physical uncertainty or attack-resistance certificate."""
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
FEATURES=["log_source_points","log_target_points","overlap","euclidean_q50","euclidean_q90","euclidean_q95",
 "normal_q50","normal_q90","normal_q95","reciprocal_fraction","normal_jacobian_ratio","source_shape_ratio","speed_mps","dt_seconds"]
def upper_quantile(scores,alpha=.05):
 values=np.asarray(scores,dtype=float)
 if not 0<alpha<1 or not np.isfinite(values).all() or (values<0).any():raise ValueError("invalid calibration")
 k=math.ceil((len(values)+1)*(1-alpha))
 return float(np.sort(values)[k-1]) if 1<=k<=len(values) else None
def eligible(scale,q):
 return bool(q is not None and math.isfinite(q) and math.isfinite(scale) and scale>0 and scale*q<=.5)
def split_roles(rows):
 out=[]
 for name in sorted({r["recording"] for r in rows}):
  for role in ("train","calibration"):
   a=sorted([r for r in rows if r["recording"]==name and r["role"]==role],key=lambda x:x["now_us"])
   for i,r in enumerate(a):
    out.append(dict(**{k:r[k] for k in ("recording","role","index","now_us")},
      quality_role="fit" if role=="train" else ("calibrate" if i<len(a)//2 else "check")))
 return out
def block_scores(rows):
 start={}
 for row in rows:start[row["recording"]]=min(start.get(row["recording"],row["now_us"]),row["now_us"])
 blocks={}
 for row in rows:
  key=(row["recording"],(row["now_us"]-start[row["recording"]])//1000000)
  blocks[key]=max(blocks.get(key,0.),row["ratio"])
 return list(blocks.values())
def extract_features(source,target,pose,dt,speed):
 if min(len(source),len(target))<150:raise ValueError("insufficient points")
 moved=source@pose[:3,:3].T+pose[:3,3];tree=cKDTree(target);dist,ids=tree.query(moved)
 mask=dist<=1.5
 if mask.sum()<100:raise ValueError("insufficient correspondences")
 _,neighbors=tree.query(target,k=16);p=target[neighbors];p=p-p.mean(axis=1,keepdims=True)
 vals,vec=np.linalg.eigh(np.einsum("nki,nkj->nij",p,p)/16);nn=vec[ids[mask],:,0]
 residual=np.abs(np.einsum("ij,ij->i",moved[mask]-target[ids[mask]],nn))
 _,back=cKDTree(moved).query(target[ids[mask]])
 mutual=float((back==np.flatnonzero(mask)).mean())
 a=np.column_stack((nn,np.cross(moved[mask],nn)));singular=np.linalg.svd(a,compute_uv=False)
 shape=np.linalg.eigvalsh(np.cov(moved.T))
 features=[math.log1p(len(source)),math.log1p(len(target)),float(mask.mean()),
  *np.quantile(dist[mask],[.5,.9,.95]).tolist(),*np.quantile(residual,[.5,.9,.95]).tolist(),
  mutual,float(singular[-1]/max(singular[0],1e-12)),float(max(0,shape[0])/max(shape[-1],1e-12)),float(speed),float(dt)]
 if not np.isfinite(features).all():raise ValueError("nonfinite features")
 return features
