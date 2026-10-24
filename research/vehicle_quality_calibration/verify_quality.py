import bisect,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
from run_stream_benchmark import CAN
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref_at(rows,ns):
 if ns is None:return None
 times=[int(r[0])*1000 for r in rows];i=bisect.bisect_left(times,ns)
 if i<len(times) and times[i]==ns:return float(rows[i][1])
 if i==0 or i==len(times) or times[i]-times[i-1]>50000000:return None
 f=(ns-times[i-1])/(times[i]-times[i-1])
 return float((1-f)*rows[i-1][1]+f*rows[i][1])
def main():
 s=json.loads((R/"quality_summary.json").read_text());ps=json.loads((R/"pair_summary.json").read_text());run=R/s["runtime_relative_path"]
 frozen=json.loads((R/"protocol_freeze.json").read_text());protocol=json.loads((R/"protocol_v1.json").read_text())
 assert sha(R/"protocol_v1.json")==s["protocol_sha256"]==frozen["sha256"]
 assert all(sha(R/n)==v for n,v in frozen["implementation_hashes"].items())
 inputs=[json.loads(x) for x in (run/"pairs.jsonl").read_text().splitlines()]
 predictions=json.loads((run/"predictions.json").read_text());forest=json.loads((run/"forest.json").read_text())
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 metadata=json.loads((cr/"online_inputs.json").read_text());refs=json.loads((cr/"offline_references.json").read_text())
 permitted={}
 for name in sorted({r["recording"] for r in metadata}):
  for role in ("train","calibration"):
   group=sorted([r for r in metadata if r["recording"]==name and r["role"]==role],key=lambda r:r["now_us"])
   for i,r in enumerate(group):permitted[name,r["index"]]=(r["now_us"],"fit" if role=="train" else "calibrate" if i<len(group)//2 else "check")
 assert len(inputs)==len(predictions)==len(permitted)==s["pairs"]==ps["pairs"]
 snap=0
 for r,p in zip(inputs,predictions):
  assert (r["now_us"],r["quality_role"])==permitted[r["recording"],r["index"]]
  for key in r:assert r[key]==p[key]
  for kind,ns in [("query",r["now_us"]*1000),("midpoint",r["midpoint_ns"])]:
   value=ref_at(refs[r["recording"]],ns);actual=p[kind+"_reference"]
   assert (value is None and actual is None) or abs(value-actual)<1e-10
   key="error_mps" if kind=="query" else "midpoint_error_mps"
   err=abs(r["speed_mps"]-value) if r["valid"] and value is not None else None
   assert (err is None and p[key] is None) or abs(err-p[key])<1e-10
  if r["valid"]:
   mount=np.array(r["rear_from_lidar"]);pose=np.array(r["pose"]);delta=mount@pose@np.linalg.inv(mount)
   assert abs(r["dt_seconds"]-(r["current_end_ns"]-r["previous_end_ns"])/1e9)<1e-12
   assert abs(r["speed_mps"]-np.linalg.norm(delta[:3,3])/r["dt_seconds"])<1e-10
   assert r["midpoint_ns"]==(r["current_end_ns"]+r["previous_end_ns"])//2
  if "snapshot_file" in r:
   path=run/r["snapshot_file"];assert sha(path)==r["snapshot_sha256"]
   with np.load(path) as z:
    now=r["now_us"]*1000
    assert (z["current_times"]>=now-20000000).all() and (z["current_times"]<=now).all()
    assert (z["previous_times"]>=now-200000000).all() and (z["previous_times"]<=now-100000000).all()
    if r["valid"]:
     source=z["source"];target=z["target"];moved=source@pose[:3,:3].T+pose[:3,3]
     tree=cKDTree(target);distance,ids=tree.query(moved);mask=distance<=1.5
     _,near=tree.query(target,k=16);a=target[near];a=a-a.mean(axis=1,keepdims=True)
     eig,vec=np.linalg.eigh((a.transpose(0,2,1)@a)/16);normal=vec[ids[mask],:,0]
     error=np.abs(np.sum((moved[mask]-target[ids[mask]])*normal,axis=1))
     _,reverse=cKDTree(moved).query(target[ids[mask]])
     singular=np.linalg.svd(np.concatenate((normal,np.cross(moved[mask],normal)),axis=1),compute_uv=False)
     shape=np.linalg.eigvalsh(np.cov(moved.T))
     feature=[np.log1p(len(source)),np.log1p(len(target)),mask.mean(),*np.percentile(distance[mask],[50,90,95]),*np.percentile(error,[50,90,95]),
      (reverse==np.flatnonzero(mask)).mean(),singular[-1]/max(singular[0],1e-12),max(0,shape[0])/max(shape[-1],1e-12),r["speed_mps"],r["dt_seconds"]]
     np.testing.assert_allclose(feature,r["features"],rtol=1e-6,atol=1e-7)
   snap+=1
 # Independently refit using only permitted fitting rows and compare tree arrays.
 fit=[r for r in predictions if r["quality_role"]=="fit" and r["valid"] and r["error_mps"] is not None]
 model=RandomForestRegressor(**protocol["forest_parameters"]);model.fit(np.array([r["features"] for r in fit]),np.log1p([r["error_mps"] for r in fit]))
 assert len(forest["trees"])==len(model.estimators_)==200
 for t,e in zip(forest["trees"],model.estimators_):
  np.testing.assert_array_equal(t["left"],e.tree_.children_left);np.testing.assert_array_equal(t["right"],e.tree_.children_right)
  np.testing.assert_allclose(t["value"],e.tree_.value[:,0,0],atol=0,rtol=0)
 for r in predictions:
  value=None
  if r["valid"]:
   x=np.asarray(r["features"],dtype=np.float32);logs=[]
   for tree in forest["trees"]:
    node=0
    while tree["left"][node]!=-1:node=tree["left"][node] if x[tree["feature"][node]]<=tree["threshold"][node] else tree["right"][node]
    logs.append(tree["value"][node])
   value=max(.05,math.expm1(sum(logs)/len(logs)))
  assert (value is None and r["predicted_scale_mps"] is None) or abs(value-r["predicted_scale_mps"])<1e-10
 calibration=[r for r in predictions if r["quality_role"]=="calibrate" and r["ratio"] is not None]
 starts={name:min(r["now_us"] for r in calibration if r["recording"]==name) for name in {r["recording"] for r in calibration}}
 blocks={}
 for r in calibration:
  ratio=r["error_mps"]/r["predicted_scale_mps"];assert abs(ratio-r["ratio"])<1e-10
  key=(r["recording"],(r["now_us"]-starts[r["recording"]])//1000000)
  blocks[key]=max(blocks.get(key,0),ratio)
 order=math.ceil((len(blocks)+1)*.95);q=sorted(blocks.values())[order-1] if order<=len(blocks) else None
 assert len(blocks)==s["calibration_block_count"] and q==s["block_quantile"] and order==s["finite_sample_rank"]
 for r in predictions:
  upper=r["predicted_scale_mps"]*q if r["valid"] and q is not None else None
  assert (upper is None and r["upper_error_estimate_mps"] is None) or abs(upper-r["upper_error_estimate_mps"])<1e-10
  assert r["quality_eligible"]==bool(upper is not None and np.isfinite(upper) and upper<=.5)
 check=[r for r in predictions if r["quality_role"]=="check"];accepted=[r for r in check if r["quality_eligible"]]
 assert len(check)==s["check_requests"] and len(accepted)==s["guarded_check"]["accepted"]
 assert not s["physical_speed_authorized"] and not s["promotion_allowed"]
 result=dict(all_passed=True,pairs_and_labels_checked=len(predictions),feature_snapshots_independently_checked=snap,
  random_forest_refit_with_fit_roles_only=True,exported_tree_predictions_replayed=True,calibration_blocks_checked=len(blocks),
  evaluation_payloads_decoded=False,protected_holdouts_accessed=False,formal_error_certificate_established=False,
  limitation="all split/labels/prediction/quantile arithmetic and sampled source feature snapshots checked; no new audit of every raw payload or physical timestamp certificate")
 (R/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
