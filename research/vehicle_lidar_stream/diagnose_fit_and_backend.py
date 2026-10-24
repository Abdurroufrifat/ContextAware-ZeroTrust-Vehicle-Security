"""Read-only structural diagnosis and backend availability. No thresholds or policy changes."""
import hashlib,importlib.util,json,urllib.request
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/"stream_benchmark_summary.json").read_text());run=ROOT/s["runtime_relative_path"]
with (run/"estimates.jsonl").open() as f: rows=[json.loads(l) for l in f]
v=json.loads((ROOT/"independent_verification.json").read_text())
bad={(x["recording"],x["index"]) for x in v["geometry_large_errors"]}
result=[]
for r in rows:
    if (r["recording"],r["index"]) not in bad: continue
    with np.load(run/r["input_file"]) as x:
        c,p=x["current"],x["previous"]
        eig=np.linalg.eigvalsh(np.cov(p.T))
        result.append(dict(recording=r["recording"],index=r["index"],current_points=len(c),
            previous_points=len(p),previous_coordinate_spread_m=np.ptp(p,axis=0).tolist(),
            previous_covariance_eigenvalues=eig.tolist(),source_azimuth_span_rad=float(np.ptp(np.arctan2(c[:,1],c[:,0]))),
            registration_reason=r["estimate"]["reason"],condition=r["estimate"]["condition"],
            lower_midpoint_age_before_compute_ms=(r["now_ns"]-r["lower_support_midpoint_ns"])/1e6,
            processing_ms=r["algorithm_ns"]/1e6,queue_wait_ms=r["queue_wait_ns"]/1e6))
print("FAILED_CASE_STRUCTURE",json.dumps(result),flush=True)
print("INSTALLED_BACKENDS",json.dumps({x:bool(importlib.util.find_spec(x)) for x in ("small_gicp","kiss_icp","open3d","numba")}),flush=True)
with urllib.request.urlopen("https://pypi.org/pypi/small-gicp/json",timeout=30) as f: metadata=json.load(f)
files=[x for x in metadata["urls"] if "cp312" in x["filename"] and "win_amd64" in x["filename"]]
print("PYPI",metadata["info"]["version"],json.dumps([dict(filename=x["filename"],url=x["url"],sha256=x["digests"]["sha256"],bytes=x["size"]) for x in files]),flush=True)
(ROOT/"backend_feasibility.json").write_text(json.dumps(dict(structural_diagnosis=result,package=metadata["info"]["name"],
    version=metadata["info"]["version"],windows_python312_wheels=files),indent=2)+"\n")
