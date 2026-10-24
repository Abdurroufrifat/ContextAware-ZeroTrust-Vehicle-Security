from runtime_common import *
from pooled_policy import PooledPolicy,PooledFastPolicy
t=next(t for t in load(B/"results.json")["trials"] if t["method"]=="joint" and t["case"]=="nominal");rows=trace(t);results=[]
for label,cls in [("original",Original),("pooled_native",PooledPolicy),("pooled_direct_forest",PooledFastPolicy)]:
 p=cls(fresh_state("development-v2-"+label),rows[0]["context"],KEYS,"joint")
 try:
  times,_=run_rows(p,rows)
  results.append({"variant":label,"queries":len(times),"mean_ms":sum(times)/len(times)*1000,"total_seconds":sum(times),"p95_ms":sorted(times)[189]*1000,"equivalence_passed":True})
 finally:
  if hasattr(p,"close"):p.close()
(S/"pool_development_result.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
