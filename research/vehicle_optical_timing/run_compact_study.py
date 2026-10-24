"""Full fixed-data compact-representation replay; no certified transport claim."""
from pathlib import Path
import sys,os,json,hashlib,time,tarfile
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np
R=Path(__file__).resolve().parent;C=R.parent/"vehicle_crossmodal_doppler";H=R.parent/"vehicle_hercules_development"
sys.path[:0]=[str(C),str(H)]
from compact_packet import encode,decode,SLOT
from crossmodal_core import estimate_optical
from crossmodal_replay import make_cases,latest_completed,metrics,interpolate
from hercules_core import latest_imu
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
names=["PROTOCOL.md","compact_packet.py","test_compact_packet.py","run_compact_study.py"]
frozen={n:sha(R/n) for n in names};lock=R/"experiment_lock.json"
if lock.exists():assert json.loads(lock.read_text())==frozen
else:lock.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
for n,h in json.loads((C/"full_experiment_lock.json").read_text()).items():assert sha(C/n)==h,n
unchanged={n:sha(C/n) for n in ["full_summary.json","Library01_crossmodal_replay.npz","Library02_crossmodal_replay.npz","FULL_RESEARCH_DECISION.json","crossmodal_core.py","crossmodal_replay.py"]}
member_lock=json.loads((C/"optical_member_lock.json").read_text());sources=json.loads((C/"full_data_lock.json").read_text())
metadata={};arrays={}
# Preparation: compressed-source read, full byte identity check, fixed encoder.
# Encoder CPU is carried forward into every simulated frame service duration.
for seq in ["Library01","Library02"]:
 old=np.load(C/(seq+"_crossmodal_replay.npz"),allow_pickle=False);stamps=old["stamps"];lookup={int(x):i for i,x in enumerate(stamps)}
 records={x["stamp_ns"]:x for x in member_lock[seq]};n=len(stamps);cache=R/(seq+"_packets.bin");rows=[None]*n
 archive=C/sources[seq]["file"];assert sha(archive)==sources[seq]["sha256"]
 with cache.open("wb") as target,tarfile.open(archive,"r|gz") as ar:
  for member in ar:
   if member.isdir():continue
   name=Path(member.name);assert member.isfile() and not name.is_absolute() and ".." not in name.parts and name.suffix==".bin" and member.size<=32*1024**2
   stamp=int(name.stem);i=lookup[stamp];assert rows[i] is None
   t0=time.perf_counter_ns();b=ar.extractfile(member).read();t1=time.perf_counter_ns()
   origin=records[stamp];assert len(b)==origin["bytes"] and hashlib.sha256(b).hexdigest()==origin["sha256"]
   start=time.perf_counter_ns();packet=encode(b,stamp,origin["sha256"]);encode_ns=time.perf_counter_ns()-start
   target.seek(i*SLOT);target.write(packet);target.write(b"\0"*(SLOT-len(packet)))
   rows[i]={"stamp_ns":stamp,"source_bytes":len(b),"source_sha256":origin["sha256"],"packet_bytes":len(packet),"packet_sha256":hashlib.sha256(packet).hexdigest(),"encode_ns":encode_ns,"archive_read_ns":t1-t0}
 assert all(x is not None for x in rows)
 metadata[seq]={"cache_file":cache.name,"cache_bytes":cache.stat().st_size,"cache_sha256":sha(cache),"frames":rows}
 print(seq,"PREPARED",n,"frames, cache bytes",cache.stat().st_size,flush=True)
(R/"packet_source_lock.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
def tr(p):return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
summaries={}
for seq,entry in metadata.items():
 old=np.load(C/(seq+"_crossmodal_replay.npz"),allow_pickle=False);radar=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
 D=H/"data"/seq;T=tr(D/"Calibration"/"Continental_LiDAR.txt");I=tr(D/"Calibration"/"IMU_LiDAR.txt")[:,:3]
 mr=[x.split(",") for x in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if x.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);gyro=np.array([[float(v) for v in x[8:11]] for x in mr])
 stamps=old["stamps"];n=len(stamps);speed=np.full(n,np.nan);quality=np.zeros(n,bool);consumer=np.zeros(n,dtype=np.int64);construction=np.array([x["encode_ns"] for x in entry["frames"]],dtype=np.int64)
 with (R/entry["cache_file"]).open("rb") as f:
  for i,row in enumerate(entry["frames"]):
   start=time.perf_counter_ns();f.seek(i*SLOT);packet=f.read(row["packet_bytes"]);parsed=decode(packet)
   assert parsed["stamp_ns"]==int(stamps[i]) and parsed["source_sha256"]==row["source_sha256"] and parsed["source_count"]*29==row["source_bytes"]
   j=latest_imu(parsed["stamp_ns"],mt);assert j is not None
   result=estimate_optical(parsed["payload"],T[:,:3],T[:,3],I,gyro[j]);consumer[i]=time.perf_counter_ns()-start
   speed[i]=result["speed"] if result["speed"] is not None else np.nan;quality[i]=result["quality"]
   assert quality[i]==old["quality"][i] and abs(speed[i]-old["speed"][i])<=1e-10,(seq,i,"prediction changed")
 service=construction+consumer;completion=np.empty(n,dtype=np.int64)
 for i,t in enumerate(stamps):completion[i]=max(int(t),int(completion[i-1]) if i else int(t))+int(service[i])
 q=radar["queries"];index=latest_completed(completion,q);safe=np.maximum(index,0);oa=q-stamps[safe]
 optical=speed[safe];rv=radar["speed"][np.maximum(radar["latest"],0)];ra=radar["age_ns"]
 pair=(index>=0)&quality[safe]&(oa>=0)&(oa<=100_000_000)&radar["ready"]
 cases=make_cases(q,rv,optical,pair,ra,oa);accepted=cases["nominal"]["accepted"];prediction=cases["nominal"]["prediction"]
 primary=radar["request_reference"];secondary=interpolate(q,radar["stamps"],radar["reference_ins"])
 s={"frames":n,"requests":len(q),"accepted":int(accepted.sum()),"availability":float(accepted.mean()),"primary":metrics(prediction,primary),"secondary":metrics(prediction,secondary),"old_gzip_path_availability":float(old["accepted"].mean()),"prediction_equivalence_max_mps":float(np.max(abs(speed-old["speed"]))),"encode_p95_ms":float(np.quantile(construction,.95)*1e-6),"consumer_p95_ms":float(np.quantile(consumer,.95)*1e-6),"combined_service_p95_ms":float(np.quantile(service,.95)*1e-6),"combined_service_max_ms":float(max(service)*1e-6),"source_bytes":sum(x["source_bytes"] for x in entry["frames"]),"packet_bytes":entry["cache_bytes"],"segments":[],"cases":{}}
 for begin in range(0,len(q),1200):
  sl=slice(begin,min(begin+1200,len(q)));m=metrics(prediction[sl],primary[sl]);availability=float(accepted[sl].mean())
  s["segments"].append({"start":begin,"requests":len(q[sl]),"availability":availability,"primary":m,"numeric_pass":bool(availability>=.95 and m["scored"]>0 and m["p95_mps"]<=.5 and m["above2"]==0),"complete_reference_support":m["unknown_reference"]==0})
 attack={}
 for case,c in cases.items():
  s["cases"][case]={"supported":int(c["supported"].sum()),"accepted":int(c["accepted"].sum()),"primary":metrics(c["prediction"],primary),"secondary":metrics(c["prediction"],secondary)}
  for field in ["supported","accepted","prediction"]:attack[case+"__"+field]=c[field]
 s["whole_numeric_pass"]=bool(s["availability"]>=.95 and s["primary"]["scored"]>0 and s["primary"]["p95_mps"]<=.5 and s["primary"]["above2"]==0)
 np.savez_compressed(R/(seq+"_replay.npz"),stamps=stamps,speed=speed,quality=quality,encode_ns=construction,consumer_ns=consumer,service_ns=service,completion_ns=completion,queries=q,index=index,optical_age_ns=oa,optical_query_speed=optical,radar_query_speed=rv,radar_age_ns=ra,pair=pair,accepted=accepted,prediction=prediction,primary=primary,secondary=secondary,**attack)
 summaries[seq]=s
 print(seq,json.dumps({k:s[k] for k in ["frames","availability","primary","combined_service_p95_ms","whole_numeric_pass"]}),flush=True)
(R/"summary.json").write_text(json.dumps({"sequences":summaries,"unchanged_prior_hashes":unchanged,"scope":"compact-representation replay from complete source-byte buffers; excludes gzip preparation and vehicle transport; construction plus consumer service counted","physical_witness_admitted":False,"Library03_payloads_accessed":False},indent=2),encoding="utf-8")
print("COMPACT_STUDY_FINISHED",flush=True)
