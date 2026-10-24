import bisect,hashlib,json,sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from quality_core import upper_quantile,eligible,block_scores,FEATURES
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
from run_stream_benchmark import CAN
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref_at(ref,now):
 times=[int(v[0])*1000 for v in ref];j=bisect.bisect_left(times,now)
 if j<len(times) and times[j]==now:return float(ref[j][1])
 if not 0<j<len(times) or times[j]-times[j-1]>50000000:return None
 return float(ref[j-1][1]+(ref[j][1]-ref[j-1][1])*(now-times[j-1])/(times[j]-times[j-1]))
def main():
 freeze=json.loads((R/"protocol_freeze.json").read_text());protocol=json.loads((R/"protocol_v1.json").read_text())
 assert sha(R/"protocol_v1.json")==freeze["sha256"]
 assert all(sha(R/n)==h for n,h in freeze["implementation_hashes"].items())
 ps=json.loads((R/"pair_summary.json").read_text());run=R/ps["runtime_relative_path"]
 assert not (R/"quality_summary.json").exists()
 features=[json.loads(x) for x in (run/"pairs.jsonl").read_text().splitlines()]
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 refs=json.loads((cr/"offline_references.json").read_text());rows=[]
 for r in features:
  ref=ref_at(refs[r["recording"]],r["now_us"]*1000);mid=ref_at(refs[r["recording"]],r["midpoint_ns"]) if r["midpoint_ns"] else None
  rows.append(dict(**r,query_reference=ref,midpoint_reference=mid,
    error_mps=abs(r["speed_mps"]-ref) if r["valid"] and ref is not None else None,
    midpoint_error_mps=abs(r["speed_mps"]-mid) if r["valid"] and mid is not None else None))
 (run/"offline_labels.json").write_text(json.dumps(rows,indent=2))
 fit=[r for r in rows if r["quality_role"]=="fit" and r["valid"] and r["error_mps"] is not None]
 model=RandomForestRegressor(**protocol["forest_parameters"])
 model.fit(np.array([r["features"] for r in fit]),np.log1p([r["error_mps"] for r in fit]))
 joblib.dump(model,run/"quality_model.joblib")
 forest=[dict(left=t.tree_.children_left.tolist(),right=t.tree_.children_right.tolist(),feature=t.tree_.feature.tolist(),
   threshold=t.tree_.threshold.tolist(),value=t.tree_.value[:,0,0].tolist()) for t in model.estimators_]
 (run/"forest.json").write_text(json.dumps(dict(features=FEATURES,trees=forest,output="max(0.05,expm1(mean_log_error))")))
 for r in rows:
  r["predicted_scale_mps"]=max(.05,float(np.expm1(model.predict([r["features"]])[0]))) if r["valid"] else None
  r["ratio"]=r["error_mps"]/r["predicted_scale_mps"] if r["predicted_scale_mps"] is not None and r["error_mps"] is not None else None
 calibration=[r for r in rows if r["quality_role"]=="calibrate" and r["ratio"] is not None]
 blocks=block_scores(calibration);q=upper_quantile(blocks,.05)
 # Per-frame diagnostic is shown separately; it never substitutes for block calibration.
 frame_q=upper_quantile([r["ratio"] for r in calibration],.05)
 for r in rows:
  r["upper_error_estimate_mps"]=r["predicted_scale_mps"]*q if q is not None and r["predicted_scale_mps"] is not None else None
  r["quality_eligible"]=bool(r["valid"] and r["predicted_scale_mps"] is not None and eligible(r["predicted_scale_mps"],q))
 (run/"predictions.json").write_text(json.dumps(rows,indent=2))
 check=[r for r in rows if r["quality_role"]=="check"]
 accepted=[r for r in check if r["quality_eligible"]];errors=[r["error_mps"] for r in accepted if r["error_mps"] is not None]
 validcheck=[r for r in check if r["valid"] and r["error_mps"] is not None]
 allerr=[r["error_mps"] for r in validcheck]
 checkblocks={}
 for name in sorted({r["recording"] for r in check}):
  group=[r for r in calibration if r["recording"]==name]
  checkblocks[name]=len(block_scores(group))
 summary=dict(protocol_sha256=freeze["sha256"],runtime_relative_path=str(run.relative_to(R)),pairs=len(rows),
  valid_fit_rows=len(fit),calibration_rows=len(calibration),check_requests=len(check),valid_scorable_check_rows=len(validcheck),
  calibration_block_count=len(blocks),calibration_blocks_by_record=checkblocks,
  finite_sample_rank=int(np.ceil((len(blocks)+1)*.95)),minimum_blocks_for_finite_95pct_quantile=19,
  block_quantile=q,framewise_diagnostic_quantile=frame_q,framewise_diagnostic_not_used=True,
  calibration_status="insufficient_blocks" if q is None else "finite_empirical_block_quantile",
  learned_scale_check_mae_mps=float(np.mean([abs(r["predicted_scale_mps"]-r["error_mps"]) for r in validcheck])) if validcheck else None,
  unguarded_check=dict(scored=len(allerr),p95_error_mps=float(np.quantile(allerr,.95)) if allerr else None,errors_above_2_mps=sum(e>2 for e in allerr)),
  guarded_check=dict(accepted=len(accepted),requests=len(check),coverage_pct=100*len(accepted)/len(check),scored=len(errors),
    unscorable_accepted=len(accepted)-len(errors),p95_error_mps=float(np.quantile(errors,.95)) if errors else None,
    errors_above_2_mps=sum(e>2 for e in errors)),
  blocked_evaluation=True,configuration_variants=1,physical_speed_authorized=False,formal_coverage_guarantee_established=False,
  exchangeability_established=False,calibration_blocks_independence_established=False,production_policy_changed=False,promotion_allowed=False,
  policy_nominal_coverage_unchanged_pct=79.15194346289752,
  limitation="one-second grouping reduces frame pseudo-replication but does not establish independent/exchangeable blocks or attack-conditional validity; no physical or runtime-freshness certificate")
 (R/"quality_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
