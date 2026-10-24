from ssc_common import *
import gzip,itertools,csv,subprocess
from ssc_replay import inputs,estimate_trial
from score import evolution_prefix,summarize,recorded_summary
assert not (R/"results.json").exists(),"frozen results already exist"
# Verify upstream authentication, reference alignment and source identities before consumption.
for script in ("verify_package.py","verify_study.py"):
 z=subprocess.run([sys.executable,str(OLD/script)],capture_output=True,text=True)
 (R/("upstream_"+script+".log")).write_text(z.stdout+z.stderr,encoding="utf-8")
 assert z.returncode==0,z.stdout+z.stderr
lock=json.loads((R/"input_lock.json").read_text());assert all(sha(P/n)==h for n,h in lock.items())
root=R/"decisions";root.mkdir(exist_ok=False)
summaries=[];total=0
for tid in range(144):
 trial=OLD/"simulator_outputs"/f"trial_{tid:03d}"
 with gzip.open(trial/"trace.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 assert len(rows)==200 and all(x["trial_id"]==tid for x in rows)
 items=[inputs(row) for row in rows];initial,bad=evolution_prefix(trial,rows)
 summaries.append(recorded_summary(rows))
 outputs={mode:list(estimate_trial(items,mode)) for mode in ("s3e_global","s3e_command")}
 for mode,decisions in outputs.items():
  summaries.append(summarize(rows,items,decisions,mode,initial,bad));total+=len(decisions)
 with gzip.open(root/f"trial_{tid:03d}.jsonl.gz","wt",encoding="utf-8") as f:
  for i in range(200):f.write(json.dumps({mode:out[i] for mode,out in outputs.items()},separators=(",",":"))+"\n")
 if (tid+1)%36==0:print("Replayed",tid+1,"of144 trajectories",flush=True)
assert total==57600 and len(summaries)==432
aggregate=[]
for key,group in itertools.groupby(sorted(summaries,key=lambda x:(x["trajectory_policy"],x["case"],x["age_us"],x["estimator"])),key=lambda x:(x["trajectory_policy"],x["case"],x["age_us"],x["estimator"])):
 group=list(group);out=dict(zip(("trajectory_policy","case","age_us","estimator"),key));out["seeds"]=len(group)
 for field in ("queries","selected","selected_error_gt05","selected_error_gt2","active_queries","active_selected","active_error_gt05","active_error_gt2"):
  out[field]=sum(x[field] for x in group)
 out["availability"]=out["selected"]/out["queries"]
 for field in ("empty_state_queries","set_excludes_reference","conditional_set_excludes_reference","conditional_selected_error_gt05","premise_valid_queries"):
  out[field]=None if field not in group[0] else sum(x[field] for x in group)
 aggregate.append(out)
dump(R/"results.json",dict(trajectories=144,queries=28800,new_estimator_decisions=total,summaries=summaries,
 native_full_s3e_reproduction=False,scalar_s3e_adaptation=True,new_closed_loop_results=False,
 physical_vehicle_validation=False,new_algorithm_principle_established=False))
dump(R/"aggregate_results.json",aggregate)
with (R/"aggregate_results.csv").open("w",newline="",encoding="utf-8") as f:
 writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
assert all(sha(P/n)==h for n,h in lock.items())
print("SECURE_SET_REPLAY_COMPLETE",total,flush=True)
