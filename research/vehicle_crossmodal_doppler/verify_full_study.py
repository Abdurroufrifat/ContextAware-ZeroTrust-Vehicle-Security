"""Independent array/arithmetic and provenance audit; never opens reserved payloads."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";D=R.parent/"vehicle_hercules_reference_diagnosis"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
for lock in ["experiment_lock.json","full_experiment_lock.json"]:
 for n,h in json.loads((R/lock).read_text()).items():assert sha(R/n)==h,(lock,n)
for n,h in json.loads((D/"source_stress_summary.json").read_text())["unchanged_input_hashes"].items():assert sha(H/n)==h,n
for e in json.loads((H/"download_lock.json").read_text())["files"]:
 assert e["sequence"] in ["Library01","Library02"];assert sha(H/e["relative_path"])==e["sha256"]
source_lock=json.loads((R/"full_data_lock.json").read_text());members=json.loads((R/"optical_member_lock.json").read_text())
summary=json.loads((R/"full_summary.json").read_text())["sequences"]
pilot=json.loads((R/"pilot_predictions.json").read_text());pilot_lock=json.loads((R/"pilot_data_lock.json").read_text())
totals={};total_queries=0;total_frames=0;total_pairs=0
def check_metrics(pred,ref,m):
 mask=np.isfinite(pred)&np.isfinite(ref);err=abs(pred[mask]-ref[mask])
 assert int(np.isfinite(pred).sum())==m["predictions"] and int(mask.sum())==m["scored"]
 assert int((np.isfinite(pred)&~np.isfinite(ref)).sum())==m["unknown_reference"] and int((err>2).sum())==m["above2"]
 if len(err):
  assert abs(float(np.quantile(err,.95))-m["p95_mps"])<1e-12 and abs(float(max(err))-m["max_mps"])<1e-12
 else:assert m["p95_mps"] is None and m["max_mps"] is None
for seq,s in summary.items():
 entry=source_lock[seq];assert sha(R/entry["file"])==entry["sha256"] and (R/entry["file"]).stat().st_size==entry["bytes"]
 declared={int(x) for x in (R/(seq+"_aeva_stamp.csv")).read_text().splitlines() if x.strip()}
 assert {x["stamp_ns"] for x in members[seq]}==declared and len(members[seq])==len(declared)
 catalog={x["stamp_ns"]:x for x in members[seq]}
 for x in pilot_lock[seq]["frames"]:
  assert x["sha256"]==catalog[x["stamp_ns"]]["sha256"] and x["bytes"]==catalog[x["stamp_ns"]]["bytes"]
 a=np.load(R/(seq+"_crossmodal_replay.npz"),allow_pickle=False);old=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
 stamps=a["stamps"];assert set(map(int,stamps))==set(catalog);comp=np.empty(len(stamps),dtype=np.int64)
 for i,t in enumerate(stamps):comp[i]=max(int(t),int(comp[i-1]) if i else int(t))+int(a["compute_ns"][i])
 np.testing.assert_array_equal(comp,a["completion_ns"])
 full_q=old["queries"];mask=np.ones(len(full_q),dtype=bool);q=full_q[mask]
 np.testing.assert_array_equal(q,a["queries"]);k=np.searchsorted(comp,q,side="right")-1;safe=np.maximum(k,0)
 oa=q-stamps[safe];ra=old["age_ns"][mask];rv=old["speed"][np.maximum(old["latest"][mask],0)];ov=a["speed"][safe]
 pair=(k>=0)&a["quality"][safe]&(oa>=0)&(oa<=100_000_000)&old["ready"][mask]
 np.testing.assert_array_equal(k,a["optical_index"]);np.testing.assert_array_equal(pair,a["pair"]);np.testing.assert_array_equal(old["request_reference"][mask],a["primary"])
 assert len(q)==s["requests"] and int(pair.sum())==s["paired_fresh_quality"] and int(a["quality"].sum())==s["optical_quality"]
 assert abs(float(np.quantile(a["compute_ns"],.95)*1e-6)-s["compute_p95_ms"])<1e-12
 j=np.searchsorted(q,q-1_000_000_000,side="right")-1;prior=np.maximum(j,0);replay_support=pair&(j>=0)&pair[prior]
 for case,m in s["cases"].items():
  r=rv.copy();o=ov.copy();r_age=ra.copy();o_age=oa.copy();support=pair.copy();valid=True
  if case in ["radar_plus3","both_plus3"]:r+=3
  if case in ["optical_plus3","both_plus3"]:o+=3
  if case.startswith("radar_replay"):
   r=rv[prior];r_age=ra[prior]+q-q[prior] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64);support=replay_support
  if case.startswith("optical_replay"):
   o=ov[prior];o_age=oa[prior]+q-q[prior] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64);support=replay_support
  if case=="missing_optical":valid=False
  if case=="stale_optical":o_age=np.full(len(q),100_000_001,dtype=np.int64)
  accepted=support&valid&np.isfinite(r)&np.isfinite(o)&(r_age>=0)&(o_age>=0)&(r_age<=100_000_000)&(o_age<=100_000_000)&(abs(r-o)<=1)
  prediction=np.where(accepted,r,np.nan)
  np.testing.assert_array_equal(accepted,a[case+"__accepted"]);np.testing.assert_array_equal(support,a[case+"__supported"])
  np.testing.assert_allclose(prediction,a[case+"__prediction"],equal_nan=True)
  assert int(support.sum())==m["supported"] and int(accepted.sum())==m["accepted"] and int((~support).sum())==m["unsupported"]
  check_metrics(prediction,a["primary"],m["primary"]);check_metrics(prediction,a["secondary"],m["secondary_INS"])
  t=totals.setdefault(case,{"supported":0,"accepted":0,"primary_above2":0,"primary_scored":0})
  for field in ["supported","accepted"]:t[field]+=m[field]
  t["primary_above2"]+=m["primary"]["above2"];t["primary_scored"]+=m["primary"]["scored"]
 check_metrics(a["prediction"],a["primary"],s["primary"]);assert abs(float(a["accepted"].mean())-s["availability"])<1e-15
 for segment in s["segments"]:
  sl=slice(segment["start"],segment["start"]+segment["requests"]);assert abs(float(a["accepted"][sl].mean())-segment["availability"])<1e-15
  check_metrics(a["prediction"][sl],a["primary"][sl],segment["primary"])
 total_queries+=len(q);total_frames+=len(stamps);total_pairs+=int(pair.sum())
for row in pilot:
 assert row["quality"] and row["agreement"] and abs(row["radar_speed"]-row["speed"])<=1 and 0<=row["radar_age_ns"]<=100_000_000
proofs=json.loads((R/"raw_optical_forgery_proofs.json").read_text());assert len(proofs)==6
assert all(x["nominal_quality"] and x["attack_quality"] and abs(x["speed_shift"]-3)<1e-6 for x in proofs)
tests=json.loads((R/"test_validation.json").read_text());assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0 and tests["denials"]==[]
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
assert not (H/"data"/"Library03").exists()
out={"passed":True,"scope":"complete two-sequence development replay; artifact verification, not physical certification","optical_frames":total_frames,"pilot_frames":len(pilot),"fixed_interval_requests":total_queries,"nominal_paired_requests":total_pairs,"cases":totals,"encoded_raw_optical_forgery_checks":len(proofs),"original_inputs_and_failed_results_unchanged":True,"full_sequence_study_complete":True,"overall_project_complete":False,"Library03_payloads_accessed":False,"tests":tests}
(R/"full_independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="tests"},indent=2))