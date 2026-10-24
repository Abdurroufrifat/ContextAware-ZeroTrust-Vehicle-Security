from pathlib import Path
import sys,json,hashlib,itertools,numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";T=R.parent/"vehicle_optical_timing";D=R.parent/"vehicle_hercules_reference_diagnosis";C=R.parent/"vehicle_crossmodal_doppler"
sys.path.insert(0,str(C))
from crossmodal_replay import interpolate
from evidence_core import metrics,decisions
for n,h in json.loads((R/"experiment_lock.json").read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
baseline=json.loads((R/"baseline_summary.json").read_text())
for p,h in baseline["unchanged_input_hashes"].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
out={};refs={}
for seq in ["Library01","Library02"]:
 t=np.load(T/(seq+"_replay.npz"),allow_pickle=False);h=np.load(H/(seq+"_replay.npz"),allow_pickle=False);d=np.load(D/(seq+"_reference_diagnostics.npz"),allow_pickle=False)
 q=h["queries"];np.testing.assert_array_equal(q,t["queries"]);r=t["radar_query_speed"];o=t["optical_query_speed"];ra=t["radar_age_ns"];oa=t["optical_age_ns"]
 safe=np.maximum(t["index"],0);oq=(t["index"]>=0)&t["quality"][safe]&(oa>=0)&(oa<=100_000_000);rq=h["ready"]
 rn=np.where(rq,r,np.nan);on=np.where(oq,o,np.nan);support=np.ones(len(q),bool)
 nominal=decisions(rn,on,ra,oa,support,True);np.testing.assert_array_equal(nominal["agreement_guard"],t["accepted"])
 nominal_predictions={"radar_only":r,"optical_only":o,"agreement_guard":r}
 common=np.logical_and.reduce(list(nominal.values()))
 s={"requests":len(q),"nominal_all_grid":{},"nominal_common_admission":{},"common_admitted":int(common.sum()),"cases":{}}
 arrays={"queries":q,"primary":t["primary"],"secondary":t["secondary"],"common":common}
 for method,mask in nominal.items():
  pred=nominal_predictions[method];s["nominal_all_grid"][method]=metrics(pred,t["primary"],mask)
  s["nominal_common_admission"][method]=metrics(pred,t["primary"],common)
  arrays[method+"__nominal_prediction"]=np.where(mask,pred,np.nan);arrays[method+"__nominal_admitted"]=mask
 np.testing.assert_array_equal(arrays["radar_only__nominal_prediction"][common],arrays["agreement_guard__nominal_prediction"][common])
 lag=np.searchsorted(q,q-1_000_000_000,side="right")-1;z=np.maximum(lag,0)
 for case in ["nominal","radar_plus3","optical_plus3","both_plus3","radar_replay_honest","radar_replay_forged","optical_replay_honest","optical_replay_forged","missing_optical","stale_optical"]:
  rr=r.copy();oo=o.copy();rage=ra.copy();oage=oa.copy();present=True;support=t[case+"__supported"]
  if case in ["radar_plus3","both_plus3"]:rr+=3
  if case in ["optical_plus3","both_plus3"]:oo+=3
  if case.startswith("radar_replay"):rr=r[z];rage=ra[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case.startswith("optical_replay"):oo=o[z];oage=oa[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case=="missing_optical":present=False
  if case=="stale_optical":oage=np.full(len(q),100_000_001,dtype=np.int64)
  masks=decisions(rr,oo,rage,oage,support,present);np.testing.assert_array_equal(masks["agreement_guard"],t[case+"__accepted"])
  report={"supported":int(support.sum()),"unsupported":int((~support).sum()),"methods":{}}
  for method,mask in masks.items():
   pred=oo if method=="optical_only" else rr;report["methods"][method]={"primary":metrics(pred,t["primary"],mask),"secondary":metrics(pred,t["secondary"],mask)}
   report["methods"][method]["admission_on_supported"]=float(mask.sum()/support.sum()) if support.any() else None
   arrays[case+"__"+method+"__prediction"]=np.where(mask,pred,np.nan);arrays[case+"__"+method+"__admitted"]=mask
  s["cases"][case]=report
 np.savez_compressed(R/(seq+"_ablations.npz"),**arrays);out[seq]=s
 signals={"radar":rn,"optical":on,"primary_pose_derivative":t["primary"],"direct_INS":t["secondary"],"raw_position_derivative":interpolate(q,d["stamps"],d["raw_position_speed"])}
 bands={"all":slice(None)}
 if seq=="Library01":bands["original_failed_302_requests"]=slice(6000,None)
 ref_report={}
 for band,sl in bands.items():
  rows={}
  for left,right in itertools.combinations(signals,2):
   x=signals[left][sl];y=signals[right][sl];finite=np.isfinite(x)&np.isfinite(y);m=metrics(x,y)
   m["unknown_pair"]=int((~finite).sum());m["signed_median_left_minus_right_mps"]=float(np.median(x[finite]-y[finite])) if finite.any() else None
   rows[left+"__vs__"+right]=m
  ref_report[band]={"requests":len(q[sl]),"start_ns":int(q[sl][0]),"end_ns":int(q[sl][-1]),"pairs":rows}
 np.savez_compressed(R/(seq+"_reference_arrays.npz"),queries=q,**signals);refs[seq]=ref_report
 print(seq,"nominal",json.dumps(s["nominal_all_grid"]),flush=True)
(R/"ablation_summary.json").write_text(json.dumps({"sequences":out,"scope":"same query grid; each-method nominal coverage plus identical supported-pair interface stress cohorts","physical_false_acceptance_rate_established":False},indent=2),encoding="utf-8")
(R/"reference_triangulation.json").write_text(json.dumps({"sequences":refs,"references_independently_certified":False,"optical_is_predictor_not_new_ground_truth":True,"reference_replaced":False,"old_failed_segment_retained":True},indent=2),encoding="utf-8")
print("MATCHED_ANALYSIS_FINISHED",flush=True)
