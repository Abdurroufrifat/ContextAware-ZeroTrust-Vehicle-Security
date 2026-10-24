import numpy as np
from scipy.spatial import cKDTree
def select_anchors(rows):
 out=[]
 for name in sorted({r["recording"] for r in rows}):
  for role in ("train","calibration"):
   group=sorted((r for r in rows if r["recording"]==name and r["role"]==role),key=lambda r:r["now_us"])
   for fraction in (.25,.5,.75):
    i=int((len(group)-1)*fraction);r=group[i]
    out.append({k:r[k] for k in ("recording","role","index","now_us")})
 return out
def warp_to_sensor(world,pose):
 return (world-pose[:3,3])@pose[:3,:3]
def discrepancy(source,target,pose):
 if not len(source) or not len(target):return dict(points=len(source),p95_m=None,overlap=0.,residual_screen_passed=False)
 moved=source@pose[:3,:3].T+pose[:3,3]
 distance,ids=cKDTree(target).query(moved,workers=1)
 good=distance<=1.5;inliers=int(good.sum())
 p95=float(np.quantile(distance[good],.95)) if inliers else None
 return dict(points=len(source),p95_m=p95,overlap=float(good.mean()),inliers=inliers,
  residual_screen_passed=bool(inliers>=100 and good.mean()>=.5 and p95 is not None and p95<=.3))
