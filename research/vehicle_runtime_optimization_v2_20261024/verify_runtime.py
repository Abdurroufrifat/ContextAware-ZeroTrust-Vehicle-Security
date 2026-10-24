"""Independent replay-evidence and timing arithmetic audit; no simulator launch."""
from runtime_common import S,P,R,B,sha,load
from pathlib import Path
import json,math,statistics,hashlib,random,datetime
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def quantile(values,p):
 a=sorted(values);return a[math.ceil(p*len(a))-1]
def main():
 result=load(S/"study_results.json");protocol=load(S/"BENCHMARK_PROTOCOL.json");lock=load(S/"input_lock.json")
 assert result["protocol_sha256"]==sha(S/"BENCHMARK_PROTOCOL.json") and result["input_lock_sha256"]==sha(S/"input_lock.json")
 for n in lock:
  k=n.casefold();assert not("library03" in k or "rav4" in k or "vehicle_final_evaluation" in k or ("urban02" in k and "night" in k))
 assert len(result["benchmark_records"])==36 and len(result["equivalence_records"])==216
 alltrials=load(B/"results.json")["trials"];trialmap={t["trial_id"]:t for t in alltrials}
 expected={}
 import gzip
 for identity in protocol["equivalence_trial_ids"]:
  t=trialmap[identity];evidence=[];values=[]
  with gzip.open(B/t["trace"],"rt") as f:
   for line in f:
    x=json.loads(line);out={k:v for k,v in x["output"].items() if k!="source_output"}
    body={"policy":out,"packet":x["issued_command"],"accepted":x["accepted_mmps2"],"status":x["actuator_status"],"card":x["operator_card"]}
    evidence.append(hashlib.sha256(canonical(body)).hexdigest())
    values.append([out["estimate_mmps"],out["requested_mmps2"],x["accepted_mmps2"],x["actuator_status"]])
  assert len(values)==200
  expected[identity]={"evidence":hashlib.sha256("".join(evidence).encode("ascii")).hexdigest(),"values":hashlib.sha256(canonical(values)).hexdigest(),"trace":sha(B/t["trace"])}
 def checked(name):
  path=S/name;m=load(path.with_suffix(".complete.json"));assert m["sha256"]==sha(path) and m["lock_digest"]==sha(S/"input_lock.json")
  r=load(path);e=expected[r["trial_id"]]
  assert r["equivalent"] and r["queries"]==200 and len(r["query_wall_seconds"])==200
  assert all(math.isfinite(v) and v>=0 for v in r["query_wall_seconds"])
  assert r["evidence_sha256"]==e["evidence"] and r["values_sha256"]==e["values"] and r["trace_sha256"]==e["trace"]
  assert r["synchronous"]==2 and r["journal_mode"]==("delete" if r["variant"]=="original" else "wal")
  return r
 records=[checked(n) for n in result["benchmark_records"]]
 order=[]
 for index,identity in enumerate(protocol["benchmark_trial_ids"]):
  vs=protocol["variants"][index%3:]+protocol["variants"][:index%3]
  order.extend((identity,v) for v in vs)
 assert [(r["trial_id"],r["variant"]) for r in records]==order
 assert {(r["trial_id"],r["variant"]) for r in records}=={(i,v) for i in protocol["benchmark_trial_ids"] for v in protocol["variants"]}
 assert {r["trial_id"] for r in result["equivalence_records"]}==set(protocol["equivalence_trial_ids"])
 for ref in result["equivalence_records"]:
  r=checked(ref["record"]);assert r["trial_id"]==ref["trial_id"] and r["variant"]=="pool_fast"
 assert result["benchmark_queries"]==7200 and result["equivalence_queries"]==43200
 stats=[]
 for variant in protocol["variants"]:
  rs=[r for r in records if r["variant"]==variant];times=[v for r in rs for v in r["query_wall_seconds"]]
  stats.append({"variant":variant,"paired_trials":12,"queries":len(times),"p50_ms":quantile(times,.5)*1000,"p95_ms":quantile(times,.95)*1000,"p99_ms":quantile(times,.99)*1000,"maximum_ms":max(times)*1000,
    "mean_ms":statistics.mean(times)*1000,"over_100ms":sum(v>.1 for v in times),"initialization_median_ms":statistics.median(r["initialization_seconds"] for r in rs)*1000,"initialization_maximum_ms":max(r["initialization_seconds"] for r in rs)*1000,"close_maximum_ms":max(r["close_seconds"] for r in rs)*1000})
 effects=[]
 bykey={(r["trial_id"],r["variant"]):r for r in records}
 for variant in ["pool_native","pool_fast"]:
  diffs=[1000*(statistics.mean(bykey[i,variant]["query_wall_seconds"])-statistics.mean(bykey[i,"original"]["query_wall_seconds"])) for i in protocol["benchmark_trial_ids"]]
  rng=random.Random(5843);draws=sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(2000))
  effects.append({"left":variant,"right":"original","metric":"mean_loop_ms","paired_trials":12,"mean_difference_ms":statistics.mean(diffs),"descriptive_trial_cluster_interval_95":[draws[49],draws[1950]],"per_trial_differences_ms":diffs})
 summary={"passed":True,"benchmark_trials_per_variant":12,"queries_per_variant":2400,"equivalence_trials":216,"equivalence_queries":43200,"stats":stats,"effects":effects,"source_diagnostic_float_tolerance":1e-12,
   "policy_outputs_certificates_and_cards_exact":True,"sqlite_version":records[0]["sqlite_version"],"initialization_excluded_from_periodic_loop":True,"input_files_verified":len(lock),"new_vehicle_experiments":0,"realtime_deadline_proven":False,"benchmark_seed":protocol["benchmark_seed"],"order_rotation_verified":True}
 (S/"runtime_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 text="# Runtime adapter validation\n\nNo vehicle experiment was repeated and no frozen policy was modified. Saved packet inputs were replayed. This separate adapter uses SQLite WAL with FULL synchronization and a checked single-owner connection reused by the original authentication code; the original per-packet authentication transactions and commits remain. Frozen forest predictions use the same float32 splits and float64 tree accumulation, with the native implementation retained for unsupported/nonfinite inputs. No learned weights or decision thresholds changed.\n\n"
 text+="## Matched sequential timing\n\n12 saved scenarios (all12conditions,3profiles assigned cyclically, predeclared seed2467),200queries each, identical inputs. Original/pooled-native/pooled-forest implementations rotate order across scenarios. Every measured sample and outlier is retained. This is engineering replay, not a new independent research confirmation. The old three-worker wall times are not the paired baseline.\n\n"
 text+="| Implementation | Queries | p50 ms | p95 ms | p99 ms | Maximum ms | Loops >100ms |\n|---|---:|---:|---:|---:|---:|---:|\n"
 for r in stats:text+=f"| {r['variant']} | {r['queries']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['maximum_ms']:.3f} | {r['over_100ms']} |\n"
 text+="\nPer-implementation initialization and checkpoint-close times are retained separately in runtime_summary.json. Fixed-period replay time excludes file loading, post-measurement output comparisons, plotting, initialization and closing. It includes policy processing, command/certificate construction and actuation checks. Warm caches, system scheduling and shared Windows activity limit transferability. Descriptive intervals resample12 matched scenario blocks at one seed; they do not establish a population deadline bound.\n\n"
 text+="## Behavior and persistence\n\nAll216joint-policy saved trials,43200decisions across6seeds,3profiles and12conditions, match the original policy outputs (outside nested upstream diagnostics), issued signed command/certificate bytes, accepted commands/statuses and operator cards exactly. Nested upstream diagnostic floats are compared at1e-12; this tolerance is not applied to issued packets, policy outputs or cards. Full equivalence reuses12completed combined-adapter benchmark checkpoints and checks the remaining204 in3isolated worker processes in fresh state. No simulator, new trajectories, physical measurements or revised collision counts are generated.\n\n"
 text+="Seventeen focused tests cover frozen-tree predictions including observed features and split neighbors, native fallback, FULL/WAL settings, saved command equivalence, forged packets, watchdog expiry, owner-thread enforcement, replay-state reopening and committed/uncommitted recovery after abrupt exits of dedicated test subprocesses. This is process-exit recovery, not a power-loss test or complete controller-state recovery. Restart requires the original qualified-session procedure; this adapter does not invent bootstrap trust.\n\n"
 text+="## SQLite and execution scope\n\nThe device SQLite version is"+records[0]["sqlite_version"]+". SQLite documents a WAL-reset race in older versions when concurrent connections write/checkpoint from separate threads/processes. This adapter requires one owner thread/process and sequential database writes; its cached connection keeps every original FULL commit, revalidates database file identity on every loan, and checkpoints after all loans are released. Concurrent external writers/checkpointers and network filesystems are outside this runtime qualification. This does not certify concurrent operation on the installed SQLite version. https://sqlite.org/wal.html . FULL synchronization is retained, not replaced by NORMAL/OFF; physical durability remains dependent on the storage system. https://sqlite.org/pragma.html#pragma_synchronous .\n\n"
 text+="A finite replay benchmark cannot establish hard real-time scheduling or latency-aware closed-loop safety. The original simulation's exact-time delivery/cadence premises remain. The adapter is opt-in, single-owner and qualified for the joint method only. No manuscript or journal-acceptance claim is produced.\n"
 (S/"RUNTIME_REPORT.md").write_text(text,encoding="utf-8")
 import matplotlib;matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(3);width=.24
 for j,key in enumerate(["p50_ms","p95_ms","p99_ms"]):axes[0].bar(x+(j-1)*width,[r[key] for r in stats],width,label=key.replace("_ms",""))
 axes[0].set_xticks(x,[r["variant"] for r in stats]);axes[0].set(ylabel="Loop time (ms)",title="Matched replay timing");axes[0].axhline(100,color="black",ls="--",lw=1);axes[0].legend()
 for variant in protocol["variants"]:
  values=sorted(v*1000 for r in records if r["variant"]==variant for v in r["query_wall_seconds"]);stride=max(1,len(values)//1500);ix=list(range(0,len(values),stride))+[len(values)-1]
  axes[1].plot([values[i] for i in ix],[(i+1)/len(values) for i in ix],label=variant)
 axes[1].set(xscale="log",xlabel="Recorded loop time (ms)",ylabel="Cumulative fraction",title="All measured outliers retained");axes[1].axvline(100,color="black",ls="--",lw=1);axes[1].legend(fontsize=8)
 fig.suptitle("Sequential matched replay; not a hard real-time guarantee");fig.tight_layout();fig.savefig(S/"runtime_results.png",dpi=180);fig.savefig(S/"runtime_results.pdf");plt.close(fig)
 unit=(S/"runtime/final_unit_tests.log").read_text(encoding="utf-8");assert "Ran 17 tests" in unit and unit.rstrip().endswith("OK")
 complete={"status":"COMPLETE_FOR_RUNTIME_REPLAY_SCOPE","equivalent_joint_trials":216,"equivalent_decisions":43200,"paired_benchmark_trials":12,"benchmarked_variants":3,"unit_tests":17,"new_vehicle_experiments":0,"frozen_sources_unchanged":True,"input_lock_sha256":sha(S/"input_lock.json"),"summary_sha256":sha(S/"runtime_summary.json"),"results_sha256":sha(S/"study_results.json")}
 files=[p for p in S.rglob("*") if p.is_file() and not any(k in p.relative_to(S).parts for k in ["runtime","__pycache__"]) and p.name not in ["package_manifest.json","RUNTIME_COMPLETE.json"]]
 bindings={str(p.relative_to(S)):sha(p) for p in sorted(files)}
 (S/"package_manifest.json").write_text(json.dumps(bindings,indent=2),encoding="utf-8")
 assert all(sha(S/n)==h for n,h in bindings.items())
 assert all(sha(P/n)==h for n,h in lock.items())
 complete["package_manifest_sha256"]=sha(S/"package_manifest.json")
 (S/"RUNTIME_COMPLETE.json").write_text(json.dumps(complete,indent=2),encoding="utf-8")
 print("RUNTIME_AUDIT_AND_RELEASE_PASS",json.dumps(summary),flush=True)
if __name__=="__main__":main()
