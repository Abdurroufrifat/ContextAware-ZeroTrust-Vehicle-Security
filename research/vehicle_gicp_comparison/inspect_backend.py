import json
from pathlib import Path
import numpy as np,small_gicp
ROOT=Path(__file__).resolve().parent
print("PREPROCESS",small_gicp.preprocess_points.__doc__)
points=np.random.default_rng(17).normal(size=(200,3))
p,t=small_gicp.preprocess_points(points,downsampling_resolution=.25,num_threads=1)
print("CLOUD_MEMBERS",[s for s in dir(p) if not s.startswith("_")])
result=small_gicp.align(p,p,t,num_threads=1)
print("RESULT_MEMBERS",[s for s in dir(result) if not s.startswith("_")])
for name in ("converged","num_inliers","iterations","error","H","T_target_source"):
    print(name,str(getattr(result,name))[:1200])
(ROOT/"preprocess_API.txt").write_text(small_gicp.preprocess_points.__doc__)
