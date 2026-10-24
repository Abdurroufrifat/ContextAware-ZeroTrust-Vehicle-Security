from pathlib import Path
import json,hashlib,struct,numpy as np
R=Path(__file__).resolve().parent;C=R.parent/"vehicle_crossmodal_doppler";H=R.parent/"vehicle_hercules_development"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
summary=json.loads((R/"summary.json").read_text());source=json.loads((R/"packet_source_lock.json").read_text())
orig_members=json.loads((C/"optical_member_lock.json").read_text());pilot=json.loads((C/"pilot_data_lock.json").read_text())
for n,h in json.loads((R/"experiment_lock.json").read_text()).items():assert sha(R/n)==h,n
for n,h in summary["unchanged_prior_hashes"].items():assert sha(C/n)==h,n
for n,h in json.loads((C/"full_experiment_lock.json").read_text()).items():assert sha(C/n)==h,n
HEADER=struct.Struct("<8sQQI32s");SLOT=HEADER.size+512*29+32;total_frames=0;total_queries=0;oracles=0;totals={}
def verify_metrics(pred,ref,m):
 mask=np.isfinite(pred)&np.isfinite(ref);e=abs(pred[mask]-ref[mask])
 assert int(np.isfinite(pred).sum())==m["predictions"] and int(mask.sum())==m["scored"]
 assert int(sum(np.isfinite(pred)&~np.isfinite(ref)))==m["unknown_reference"] and int(sum(e>2))==m["above2"]
 if len(e):assert abs(float(np.quantile(e,.95))-m["p95_mps"])<1e-12 and abs(float(max(e))-m["max_mps"])<1e-12
 else:assert m["p95_mps"] is None and m["max_mps"] is None
for seq,s in summary["sequences"].items():
 entry=source[seq];cache=R/entry["cache_file"];assert cache.stat().st_size==entry["cache_bytes"] and sha(cache)==entry["cache_sha256"]
 original={x["stamp_ns"]:x for x in orig_members[seq]};pilot_by_stamp={x["stamp_ns"]:x for x in pilot[seq]["frames"]}
 a=np.load(R/(seq+"_replay.npz"),allow_pickle=False);old=np.load(C/(seq+"_crossmodal_replay.npz"),allow_pickle=False);radar=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
 np.testing.assert_array_equal(a["stamps"],old["stamps"]);np.testing.assert_array_equal(a["quality"],old["quality"])
 np.testing.assert_allclose(a["speed"],old["speed"],atol=1e-10,rtol=0)
 with cache.open("rb") as f:
  for i,row in enumerate(entry["frames"]):
   f.seek(i*SLOT);packet=f.read(row["packet_bytes"]);assert hashlib.sha256(packet).hexdigest()==row["packet_sha256"]
   body=packet[:-32];assert hashlib.sha256(body).digest()==packet[-32:]
   magic,stamp,count,take,identity=HEADER.unpack_from(body);ref=original[stamp]
   assert magic==b"COACTOP1" and stamp==int(a["stamps"][i]) and identity.hex()==ref["sha256"] and count*29==ref["bytes"] and take==min(count,512)
   assert len(body)==HEADER.size+take*29
   if stamp in pilot_by_stamp:
    raw=(C/pilot_by_stamp[stamp]["relative_path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref["sha256"]
    ix=np.linspace(0,count-1,take,dtype=int);want=b"".join(raw[j*29:(j+1)*29] for j in ix)
    assert body[HEADER.size:]==want;oracles+=1
 np.testing.assert_array_equal(a["encode_ns"],[x["encode_ns"] for x in entry["frames"]])
 np.testing.assert_array_equal(a["service_ns"],a["encode_ns"]+a["consumer_ns"])
 done=np.empty(len(a["stamps"]),dtype=np.int64)
 for i,t in enumerate(a["stamps"]):done[i]=max(int(t),int(done[i-1]) if i else int(t))+int(a["service_ns"][i])
 np.testing.assert_array_equal(a["completion_ns"],done);q=radar["queries"];np.testing.assert_array_equal(q,a["queries"])
 j=np.searchsorted(done,q,side="right")-1;k=np.maximum(j,0);oa=q-a["stamps"][k];ov=a["speed"][k]
 rv=radar["speed"][np.maximum(radar["latest"],0)];ra=radar["age_ns"]
 pair=(j>=0)&a["quality"][k]&(oa>=0)&(oa<=100_000_000)&radar["ready"]
 np.testing.assert_array_equal(pair,a["pair"]);np.testing.assert_array_equal(a["primary"],radar["request_reference"])
 back=np.searchsorted(q,q-1_000_000_000,side="right")-1;z=np.maximum(back,0);support_replay=pair&(back>=0)&pair[z]
 for case,m in s["cases"].items():
  r=rv.copy();o=ov.copy();rage=ra.copy();oage=oa.copy();support=pair.copy();valid=True
  if case in ["radar_plus3","both_plus3"]:r+=3
  if case in ["optical_plus3","both_plus3"]:o+=3
  if case.startswith("radar_replay"):
   support=support_replay;r=rv[z];rage=ra[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case.startswith("optical_replay"):
   support=support_replay;o=ov[z];oage=oa[z]+q-q[z] if case.endswith("honest") else np.zeros(len(q),dtype=np.int64)
  if case=="missing_optical":valid=False
  if case=="stale_optical":oage=np.full(len(q),100_000_001,dtype=np.int64)
  accepted=support&valid&np.isfinite(r)&np.isfinite(o)&(rage>=0)&(rage<=100_000_000)&(oage>=0)&(oage<=100_000_000)&(abs(r-o)<=1)
  pred=np.where(accepted,r,np.nan);np.testing.assert_array_equal(accepted,a[case+"__accepted"]);np.testing.assert_array_equal(support,a[case+"__supported"])
  np.testing.assert_allclose(pred,a[case+"__prediction"],equal_nan=True);assert int(support.sum())==m["supported"] and int(accepted.sum())==m["accepted"]
  verify_metrics(pred,a["primary"],m["primary"]);verify_metrics(pred,a["secondary"],m["secondary"])
  t=totals.setdefault(case,{"supported":0,"accepted":0,"primary_scored":0,"primary_above2":0})
  t["supported"]+=m["supported"];t["accepted"]+=m["accepted"];t["primary_scored"]+=m["primary"]["scored"];t["primary_above2"]+=m["primary"]["above2"]
 verify_metrics(a["prediction"],a["primary"],s["primary"]);assert abs(float(a["accepted"].mean())-s["availability"])<1e-15
 for seg in s["segments"]:
  sl=slice(seg["start"],seg["start"]+seg["requests"]);verify_metrics(a["prediction"][sl],a["primary"][sl],seg["primary"])
  assert abs(float(a["accepted"][sl].mean())-seg["availability"])<1e-15
 assert s["whole_numeric_pass"]==bool(s["availability"]>=.95 and s["primary"]["scored"]>0 and s["primary"]["p95_mps"]<=.5 and s["primary"]["above2"]==0)
 total_frames+=len(a["stamps"]);total_queries+=len(q)
tests=json.loads((R/"test_validation.json").read_text());assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
assert total_frames==5630 and total_queries==11258 and oracles==128
out={"passed":True,"exact_prediction_equivalences":total_frames,"query_replays":total_queries,"independent_raw_byte_sample_oracles":oracles,"all_packet_metadata_and_integrity_checks":total_frames,"prior_results_and_core_unchanged":True,"cases":totals,"physical_end_to_end_availability_established":False,"Library03_payloads_accessed":False,"tests":tests}
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="tests"},indent=2))
