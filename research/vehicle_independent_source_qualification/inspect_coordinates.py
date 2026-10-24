from pathlib import Path
import numpy as np,json
from pilot_core import decode_pcd
R=Path(__file__).resolve().parent
pc=decode_pcd((R/"pilot_data"/"000000.pcd").read_bytes())
theta=np.deg2rad(pc["alpha"].astype(float));phi=np.deg2rad(pc["beta"].astype(float))
xyz=np.column_stack([pc[k] for k in ["x","y","z"]]).astype(float)
ranges=pc["range"].astype(float)
directions=np.column_stack([np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),np.sin(phi)])
def med(a):return float(np.nanmedian(a))
out={"points":len(theta),"coordinate_p05_p50_p95":{k:np.nanquantile(pc[k],[.05,.5,.95]).tolist() for k in ["x","y","z","alpha","beta","range","doppler"]},"raw_norm_vs_range_p95_m":float(np.nanquantile(abs(np.linalg.norm(xyz,axis=1)-ranges),.95)),"geometric_hypotheses":{"xyz_as_forward_left_up":med(np.linalg.norm(xyz-ranges[:,None]*directions,axis=1)),"author_remap_z_minusx_minusy":med(np.linalg.norm(xyz[:,[2,0,1]]*np.array([1,-1,-1])-ranges[:,None]*directions,axis=1))},"sample_rows":[{k:float(pc[k][i]) for k in ["x","y","z","alpha","beta","range","doppler"]} for i in range(5)]}
(R/"coordinate_diagnosis.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
