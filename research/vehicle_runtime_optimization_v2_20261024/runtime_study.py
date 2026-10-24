"""Resumable paired replay benchmark. Never launches a vehicle simulator."""
from runtime_common import *
from pooled_policy import PooledPolicy,PooledFastPolicy
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,sqlite3,datetime
VARIANTS={"original":Original,"pool_native":PooledPolicy,"pool_fast":PooledFastPolicy}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8");os.replace(tmp,path)
def state(stage,n,total,detail=None):
 write(S/"runtime/progress.json",{"stage":stage,"completed":n,"total":total,"detail":detail,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
def checkpoint(folder,t,variant,lock_digest):
 path=S/folder/(f"trial_{t['trial_id']:04d}_{variant}.json");marker=path.with_suffix(".complete.json")
 if marker.exists():
  saved=load(marker);assert saved["sha256"]==sha(path) and saved["lock_digest"]==lock_digest
  result=load(path);assert result["trial_id"]==t["trial_id"] and result["variant"]==variant
  return result
 rows=trace(t);location=fresh_state("paired-"+variant);start=time.perf_counter()
 p=VARIANTS[variant](location,rows[0]["context"],KEYS,"joint");initialization=time.perf_counter()-start
 evidence=[]
 try:
  times,values=run_rows(p,rows,evidence=evidence)
  sql=sqlite3.connect(location/"replay.sqlite")
  mode=sql.execute("PRAGMA journal_mode").fetchone()[0];sync=sql.execute("PRAGMA synchronous").fetchone()[0];sql.close()
  assert sync==2 and mode==("delete" if variant=="original" else "wal")
 finally:
  start=time.perf_counter()
  if hasattr(p,"close"):p.close()
  close_seconds=time.perf_counter()-start
 result={"trial_id":t["trial_id"],"variant":variant,"queries":len(rows),"initialization_seconds":initialization,"close_seconds":close_seconds,"query_wall_seconds":times,
   "equivalent":True,"state_directory":str(location),"journal_mode":mode,"synchronous":sync,"sqlite_version":sqlite3.sqlite_version,
   "values_sha256":hashlib.sha256(canonical(values)).hexdigest(),"evidence_sha256":hashlib.sha256("".join(evidence).encode("ascii")).hexdigest(),"trace_sha256":sha(B/t["trace"]),"lock_digest":lock_digest}
 write(path,result);write(marker,{"sha256":sha(path),"lock_digest":lock_digest});return result
def equivalent_job(t,digest):return checkpoint("equivalence",t,"pool_fast",digest)
def main():
 if (S/"RUNTIME_COMPLETE.json").exists():print("SAVED_RUNTIME_RELEASE_ALREADY_COMPLETE");return
 protocol=load(S/"BENCHMARK_PROTOCOL.json");lock=load(S/"input_lock.json");digest=sha(S/"input_lock.json")
 required={n:h for n,h in lock.items() if n.endswith((".py",".json",".joblib")) or n.startswith(str(S.relative_to(P)))}
 assert all(sha(P/n)==h for n,h in required.items()),"Frozen executable/model/metadata input mismatch"
 alltrials=load(B/"results.json")["trials"];byid={t["trial_id"]:t for t in alltrials};bench=[];done=0
 try:
  state("paired_benchmark",0,36)
  for index,identity in enumerate(protocol["benchmark_trial_ids"]):
   order=protocol["variants"][index%3:]+protocol["variants"][:index%3]
   for variant in order:
    r=checkpoint("benchmark",byid[identity],variant,digest);bench.append(r);done+=1;state("paired_benchmark",done,36)
  assert len(bench)==36
  equivalence=[];pending=[]
  for identity in protocol["equivalence_trial_ids"]:
   existing=next((r for r in bench if r["trial_id"]==identity and r["variant"]=="pool_fast"),None)
   if existing:
    equivalence.append({"trial_id":identity,"record":str((S/"benchmark"/f"trial_{identity:04d}_pool_fast.json").relative_to(S)),"reused_paired_checkpoint":True})
   else:pending.append(identity)
  state("full_equivalence",len(equivalence),216)
  with ProcessPoolExecutor(max_workers=3) as pool:
   futures={pool.submit(equivalent_job,byid[i],digest):i for i in pending}
   for future in as_completed(futures):
    r=future.result();identity=r["trial_id"]
    equivalence.append({"trial_id":identity,"record":str((S/"equivalence"/f"trial_{identity:04d}_pool_fast.json").relative_to(S)),"reused_paired_checkpoint":False})
    state("full_equivalence",len(equivalence),216)
  equivalence.sort(key=lambda x:x["trial_id"])
  write(S/"study_results.json",{"protocol_sha256":sha(S/"BENCHMARK_PROTOCOL.json"),"input_lock_sha256":digest,"benchmark_records":[str((S/"benchmark"/f"trial_{r['trial_id']:04d}_{r['variant']}.json").relative_to(S)) for r in bench],"equivalence_records":equivalence,"benchmark_queries":7200,"equivalence_queries":43200})
  state("evidence_saved",216,216);print("RUNTIME_REPLAY_EVIDENCE_SAVED",flush=True)
 except BaseException as e:
  state("FAILED",done,36,repr(e));raise
if __name__=="__main__":main()
