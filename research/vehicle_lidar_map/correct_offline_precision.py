"""Repair reference interpolation arithmetic without rerunning or changing online estimates."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/"run_benchmark.py"
old=p.read_text()
start=old.index("def interp(ref,ns):");end=old.index("def main():",start)
assert "t=ns/1000" in old[start:end]
shutil.copyfile(p,R/"RUN_SOURCE_BEFORE_PRECISION_FIX.py.snapshot")
new='''def interp(ref,ns):
 stamps=[int(v)*1000 for v in ref[:,0]]
 at=bisect.bisect_left(stamps,ns)
 if at<len(stamps) and stamps[at]==ns:return float(ref[at,1])
 if not 0<at<len(stamps):return None
 gap=stamps[at]-stamps[at-1]
 if gap>50000000:return None
 return float(ref[at-1,1]+(ref[at,1]-ref[at-1,1])*(ns-stamps[at-1])/gap)
'''
p.write_text(old[:start]+new+old[end:])
q=R/"test_map_sources.py";s=q.read_text().replace('np.ndarray((2,60),dtype="<f4",buffer=msg.data,offset=12,strides=(1208,20))[0,0]=.1','a=np.ndarray((2,60),dtype="<f4",buffer=msg.data,offset=12,strides=(1208,20));a[0,0]=.1;a[0,1]=0')
q.write_text(s)
from run_benchmark import interp,CAN
import numpy as np
summary=json.loads((R/"benchmark_summary.json").read_text());run=R/summary["runtime_relative_path"]
shutil.copyfile(R/"benchmark_summary.json",R/"benchmark_summary_float_epoch.json")
shutil.copyfile(run/"offline_scores.json",run/"offline_scores_float_epoch.json")
scores=json.loads((run/"offline_scores.json").read_text())
cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
refs={k:np.array(v) for k,v in json.loads((cr/"offline_references.json").read_text()).items()}
diffs=[]
for q in scores:
 for kind,key in [("query","now_ns"),("midpoint","midpoint_ns")]:
  val=interp(refs[q["recording"]],q[key]) if q[key] is not None else None
  before=q[kind+"_reference"]
  if val is not None and before is not None:diffs.append(abs(val-before))
  else:assert val==before
  q[kind+"_reference"]=val
  q[kind+"_error_mps"]=abs(q["speed_mps"]-val) if q["speed_mps"] is not None and val is not None else None
for gate,m in summary["metrics"].items():
 group=[q for q in scores if q[gate]]
 for kind in ("query","midpoint"):
  err=[q[kind+"_error_mps"] for q in group if q[kind+"_error_mps"] is not None]
  m[kind]=dict(scored=len(err),unscorable=len(group)-len(err),p95_error_mps=float(np.quantile(err,.95)) if err else None,max_error_mps=max(err) if err else None,errors_above_2_mps=sum(x>2 for x in err))
note=dict(reason="independent verifier exposed epoch-to-float rounding in offline interpolation; subtract integer timestamps before division",maximum_reference_correction_mps=max(diffs),online_poses_queries_and_timing_unchanged=True,eligibility_unchanged=True)
assert max(diffs)<.00001
summary["offline_precision_correction"]=note
(run/"offline_scores.json").write_text(json.dumps(scores,indent=2))
(R/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(R/"precision_correction.json").write_text(json.dumps(note,indent=2)+"\n")
print(json.dumps(note,indent=2))
