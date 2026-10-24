"""Independent arithmetic/provenance replay for matched comparison."""
from pathlib import Path
import json,hashlib,math,bisect,itertools,numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";T=R.parent/"vehicle_optical_timing";D=R.parent/"vehicle_hercules_reference_diagnosis"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((R/"baseline_summary.json").read_text());abl=json.loads((R/"ablation_summary.json").read_text());refs=json.loads((R/"reference_triangulation.json").read_text())
for n,h in json.loads((R/"experiment_lock.json").read_text()).items():assert sha(R/n)==h,n
for p,h in base["unchanged_input_hashes"].items():assert sha(Path(p))==h,p
def check(pred,ref,mask,m):
 pred=np.asarray(pred,float);ref=np.asarray(ref,float);mask=np.asarray(mask,bool)
 ok=mask&np.isfinite(pred);scored=ok&np.isfinite(ref);e=[abs(float(a)-float(b)) for a,b in zip(pred[scored],ref[scored])]
 assert len(pred)==m["attempts"] and int(ok.sum())==m["admitted"] and int(scored.sum())==m["scored"]
 assert int((ok&~np.isfinite(ref)).sum())==m["unknown_reference"] and sum(x>2 for x in e)==m["above2"]
 assert int((mask&~np.isfinite(pred)).sum())==m["invalid_prediction"]
 assert abs(float(ok.mean())-m["availability"])<1e-15
 if e:
  assert abs(math.fsum(e)/len(e)-m["mae_mps"])<1e-12 and abs(float(np.percentile(e,95))-m["p95_mps"])<1e-12 and abs(max(e)-m["max_mps"])<1e-12
 else:assert m["p95_mps"] is None and m["mae_mps"] is None
def interpolate(q,t,v):
 out=[];t=list(map(int,t))
 for x in q:
  j=bisect.bisect_left(t,int(x))
  if j<len(t) and t[j]==int(x):out.append(v[j])
  elif j==0 or j==len(t) or t[j]-t[j-1]>200_000_000 or not np.isfinite(v[j-1]) or not np.isfinite(v[j]):out.append(np.nan)
  else:out.append(v[j-1]+(v[j]-v[j-1])*(int(x)-t[j-1])/(t[j]-t[j-1]))
 return np.array(out)
cohort=queries=case_cells=ref_cells=0;totals={}
for seq in ["Library01","Library02"]:
 b=np.load(R/(seq+"_baselines.npz"),allow_pickle=False);h=np.load(H/(seq+"_replay.npz"),allow_pickle=False);s=base["sequences"][seq]
 np.testing.assert_array_equal(b["indices"],h["cohort_indices"]);np.testing.assert_array_equal(b["reference"],h["reference_primary"][b["indices"]])
 for name,expected in [("candidate",h["speed"][b["indices"]]),("author_3d_ransac",h["native_prediction"]),("ols_3d",h["ols_prediction"])]:
  np.testing.assert_allclose(b[name],expected,atol=1e-10,rtol=0)
  check(b[name],b["reference"],np.isfinite(b[name]),s["all_attempt_metrics"][name]);check(b[name],b["reference"],b["common"],s["common_metrics"][name])
 for name,ci in s["paired_mae_intervals"].items():
  e1=abs(b["candidate"][b["common"]]-b["reference"][b["common"]]);e2=abs(b[name][b["common"]]-b["reference"][b["common"]]);delta=e1-e2;n=len(delta)
  rng=np.random.default_rng(ci["seed"]);means=[]
  for _ in range(ci["draws"]):
   starts=rng.integers(0,n,size=(n+ci["block"]-1)//ci["block"]);ix=[]
   for start in starts:ix.extend((int(start)+j)%n for j in range(ci["block"]))
   means.append(float(np.mean(delta[ix[:n]])))
  lo,hi=np.percentile(means,[2.5,97.5]);assert abs(lo-ci["low"])<1e-12 and abs(hi-ci["high"])<1e-12 and abs(float(delta.mean())-ci["difference"])<1e-12
 cohort+=len(b["indices"])
 t=np.load(T/(seq+"_replay.npz"),allow_pickle=False);a=np.load(R/(seq+"_ablations.npz"),allow_pickle=False);q=t["queries"];s=abl["sequences"][seq]
 np.testing.assert_array_equal(q,a["queries"]);np.testing.assert_array_equal(t["primary"],a["primary"]);np.testing.assert_array_equal(t["secondary"],a["secondary"])
 r=t["radar_query_speed"];o=t["optical_query_speed"];ra=t["radar_age_ns"];oa=t["optical_age_ns"];safe=np.maximum(t["index"],0)
 radar=h["ready"]&np.isfinite(r);opt=(t["index"]>=0)&t["quality"][safe]&(oa>=0)&(oa<=100_000_000)&np.isfinite(o);guard=radar&opt&(abs(r-o)<=1)
 for method,mask,pred in [("radar_only",radar,r),("optical_only",opt,o),("agreement_guard",guard,r)]:
  np.testing.assert_array_equal(mask,a[method+"__nominal_admitted"]);check(pred,t["primary"],mask,s["nominal_all_grid"][method]);check(pred,t["primary"],guard,s["nominal_common_admission"][method])
  np.testing.assert_allclose(np.where(mask,pred,np.nan),a[method+"__nominal_prediction"],equal_nan=True)
 np.testing.assert_array_equal(guard,t["accepted"]);np.testing.assert_array_equal(a["common"],guard)
 back=np.searchsorted(q,q-1_000_000_000,side="right")-1;z=np.maximum(back,0)
 for case,c in s["cases"].items():
  rr=r.copy();oo=o.copy();rage=ra.copy();oage=oa.copy();present=True;support=t[case+"__supported"]
  if case in ["radar_plus3","both_plus3"]:rr+=3
  if case in ["optical_plus3","both_plus3"]:oo+=3
  if case.startswith("radar_replay"):rr=r[z];rage=ra[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case.startswith("optical_replay"):oo=o[z];oage=oa[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case=="missing_optical":present=False
  if case=="stale_optical":oage=np.full(len(q),100_000_001,dtype=np.int64)
  radar_ok=support&np.isfinite(rr)&(rage>=0)&(rage<=100_000_000);opt_ok=support&present&np.isfinite(oo)&(oage>=0)&(oage<=100_000_000)
  for method,mask,pred in [("radar_only",radar_ok,rr),("optical_only",opt_ok,oo),("agreement_guard",radar_ok&opt_ok&(abs(rr-oo)<=1),rr)]:
   rep=c["methods"][method];check(pred,t["primary"],mask,rep["primary"]);check(pred,t["secondary"],mask,rep["secondary"])
   np.testing.assert_array_equal(mask,a[case+"__"+method+"__admitted"]);np.testing.assert_allclose(np.where(mask,pred,np.nan),a[case+"__"+method+"__prediction"],equal_nan=True)
   agg=totals.setdefault(case,{}).setdefault(method,{"supported":0,"admitted":0,"scored":0,"above2":0,"unknown_reference":0})
   agg["supported"]+=c["supported"]
   for field in ["admitted","scored","above2","unknown_reference"]:agg[field]+=rep["primary"][field]
   case_cells+=1
   if method=="agreement_guard":np.testing.assert_array_equal(mask,t[case+"__accepted"])
 ref=np.load(R/(seq+"_reference_arrays.npz"),allow_pickle=False);nav=np.load(D/(seq+"_reference_diagnostics.npz"),allow_pickle=False)
 np.testing.assert_allclose(ref["raw_position_derivative"],interpolate(q,nav["stamps"],nav["raw_position_speed"]),atol=1e-12,rtol=0,equal_nan=True)
 np.testing.assert_array_equal(ref["primary_pose_derivative"],t["primary"]);np.testing.assert_array_equal(ref["direct_INS"],t["secondary"])
 np.testing.assert_allclose(ref["radar"],np.where(h["ready"],r,np.nan),equal_nan=True);np.testing.assert_allclose(ref["optical"],np.where(opt,o,np.nan),equal_nan=True)
 for band,c in refs["sequences"][seq].items():
  sl=slice(6000,None) if band=="original_failed_302_requests" else slice(None)
  assert len(q[sl])==c["requests"]
  for pair,m in c["pairs"].items():
   left,right=pair.split("__vs__");x=ref[left][sl];y=ref[right][sl];check(x,y,np.isfinite(x),m)
   assert int((~(np.isfinite(x)&np.isfinite(y))).sum())==m["unknown_pair"];ref_cells+=1
 queries+=len(q)
tests=json.loads((R/"test_validation.json").read_text());assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
assert cohort==564 and queries==11258 and case_cells==60 and ref_cells==30
out={"passed":True,"baseline_frames_reproduced":cohort,"common_scored_baseline_frames":562,"full_queries":queries,"method_case_cells":case_cells,"reference_pair_cells":ref_cells,"bootstrap_intervals_replayed":4,"case_totals":totals,"all_prior_inputs_and_failures_unchanged":True,"reference_gap_resolved":False,"neural_model_reproduced":False,"full_published_AV_defense_reproduced":False,"Library03_payloads_accessed":False,"tests":tests}
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k not in ["tests","case_totals"]},indent=2))
