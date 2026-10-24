"""Independent saved aggregate, matched-effect, and operator-artifact audit."""
from cv_common import B,P,sha,dump
import json,csv,math,gzip,statistics,random,re,subprocess,itertools
def near(a,b):
 if a is None or b is None:assert a is b
 else:assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)
def main():
 result=json.loads((B/"results.json").read_text());trials=result["trials"];report=json.loads((B/"comparison.json").read_text())
 assert len(trials)==1512
 cfg=json.loads((B/"PROTOCOL.json").read_text())
 expected_aggregates=set()
 for group,cases,methods in [("primary",cfg["primary_cases"],cfg["methods"]),("stress",cfg["stress_cases"],cfg["stress_methods"])]:
  expected_aggregates|={(group,case,method) for case,method in itertools.product(["ALL",*cases],methods)}
 assert {(a["group"],a["case"],a["method"]) for a in report["aggregates"]}==expected_aggregates
 assert report["input_results_sha256"]==sha(B/"results.json")
 assert report["verification_sha256"]==sha(B/"verification.json") and report["policy_replay_sha256"]==sha(B/"policy_replay.json")
 for n,h in json.loads((B/"experiment_lock.json").read_text()).items():assert sha(P/n)==h
 csvrows=list(csv.DictReader((B/"aggregate_results.csv").open(newline="",encoding="utf-8")))
 assert len(csvrows)==len(report["aggregates"])==96
 for aggregate,csvrow in zip(report["aggregates"],csvrows):
  rows=[t for t in trials if t["group"]==aggregate["group"] and t["method"]==aggregate["method"] and (aggregate["case"]=="ALL" or t["case"]==aggregate["case"])]
  assert rows
  expected={"trials":len(rows),"queries":sum(t["queries"] for t in rows),"selected":sum(t["selected"] for t in rows),
   "collision_trials":sum(bool(t["collision_trial"]) for t in rows),"collision_episodes":sum(t["collision_episodes"] for t in rows),
   "over_0_5":sum(t["over_0_5"] for t in rows),"fallback_queries":sum(t["fallback_queries"] for t in rows),
   "unrecoverable_queries":sum(t["unrecoverable_queries"] for t in rows),"changed_actuator_frames":sum(t["changed_actuator_frames"] for t in rows),
   "changed_actuator_frames_accepted":sum(t["changed_actuator_frames_accepted"] for t in rows),
   "mean_progress_m":statistics.mean(t["progress_m"] for t in rows),
   "minimum_gap_m":min(t["minimum_gap_m"] for t in rows),"mean_seconds_gap_below_2m":statistics.mean(t["seconds_gap_below_2m"] for t in rows),
   "maximum_decision_seconds":max(t["maximum_decision_seconds"] for t in rows)}
  expected["availability"]=expected["selected"]/expected["queries"]
  expected["selected_mae_mps"]=sum(t["mae_mps"]*t["selected"] for t in rows if t["selected"])/expected["selected"] if expected["selected"] else None
  for k,v in expected.items():near(aggregate[k],v)
  for k,v in aggregate.items():assert csvrow[k]==("" if v is None else str(v))
 pairfile=json.loads((B/"paired_baseline_results.json").read_text())["pairs"]
 effectfile=json.loads((B/"seed_cluster_summary.json").read_text())
 assert len(pairfile)==10 and len(report["paired_effects"])==60
 primary_pairs={("full","no_learning"),("no_learning","no_refinement"),("full","no_fusion"),("full","no_brake"),("full","no_actuator_auth"),("full","s3e_brake"),("full","ssf_zocbf")}
 expected_pairs={("primary",a,b) for a,b in primary_pairs}|{("stress",a,b) for a,b in primary_pairs if a in cfg["stress_methods"] and b in cfg["stress_methods"]}
 assert {(p["group"],p["left"],p["right"]) for p in pairfile}==expected_pairs
 metrics=("mae_mps","collision_trial","progress_m","seconds_gap_below_2m","fallback_queries","availability")
 assert {(e["group"],e["left"],e["right"],e["metric"]) for e in report["paired_effects"]}=={(*p,m) for p,m in itertools.product(expected_pairs,metrics)}
 assert effectfile["seed_clusters"]==cfg["seeds"]
 lookup={(t["group"],t["profile"],t["case"],t["seed"],t["method"]):t for t in trials}
 for pair in pairfile:
  left=[t for t in trials if t["group"]==pair["group"] and t["method"]==pair["left"]]
  assert len(left)==pair["matched_trials"]==len(pair["effects"])
  assert {(r["profile"],r["case"],r["seed"]) for r in pair["effects"]}=={(r["profile"],r["case"],r["seed"]) for r in left}
  for row in pair["effects"]:
   prefix=(pair["group"],row["profile"],row["case"],row["seed"])
   a,b=lookup[(*prefix,pair["left"])],lookup[(*prefix,pair["right"])]
   for k in ("mae_mps","collision_trial","progress_m","seconds_gap_below_2m","fallback_queries"):near(row[k],None if a[k] is None or b[k] is None else a[k]-b[k])
   near(row["availability"],a["selected"]/a["queries"]-b["selected"]/b["queries"])
 for effect in report["paired_effects"]:
  pair=next(p for p in pairfile if (p["group"],p["left"],p["right"])==(effect["group"],effect["left"],effect["right"]))
  means=[statistics.mean(r[effect["metric"]] for r in pair["effects"] if r["seed"]==seed and r[effect["metric"]] is not None) for seed in effectfile["seed_clusters"]]
  near(effect["mean_left_minus_right"],statistics.mean(means));assert effect["seed_clusters"]==6
  rng=random.Random(2911);draws=[]
  for _ in range(2000):draws.append(statistics.mean([means[rng.randrange(6)] for _ in range(6)]))
  draws.sort()
  for observed,expected in zip(effect["descriptive_cluster_bootstrap_95"],[draws[math.floor(.025*1999)],draws[math.ceil(.975*1999)]]):near(observed,expected)
 assert effectfile["effects"]==report["paired_effects"]
 effects_csv=list(csv.DictReader((B/"method_effects.csv").open(newline="",encoding="utf-8")))
 assert len(effects_csv)==60
 for r,e in zip(effects_csv,report["paired_effects"]):
  for k,v in e.items():assert r[k]==(json.dumps(v) if isinstance(v,(dict,list)) else str(v))
 replays=json.loads((B/"operator_replays.json").read_text())["replays"]
 assert len(replays)==13 and sum(len(r["cards"]) for r in replays)==2600
 assert {r["case"] for r in replays}==set(cfg["primary_cases"]+cfg["stress_cases"])
 assert all(r["profile"]=="urban_cutback" and r["seed"]==cfg["seeds"][0] for r in replays)
 for replay in replays:
  trial=next(t for t in trials if t["trial_id"]==replay["trial_id"])
  assert trial["method"]=="full" and all(trial[k]==replay[k] for k in ("case","profile","seed"))
  with gzip.open(B/trial["trace"],"rt",encoding="utf-8") as f:source=[json.loads(x)["operator_card"] for x in f]
  assert source==replay["cards"]
  assert all(not({"truth","attack_case","label"}&set(c)) for c in source)
 viewer=(B/"operator_viewer.html").read_text(encoding="utf-8")
 match=re.search(r"const REPLAYS=(.*?);\s*const el=",viewer,re.S)
 assert json.loads(match.group(1))==replays
 from PIL import Image
 with Image.open(B/"comparative_results.png") as im:assert im.width>=2000 and im.height>=800
 assert (B/"RESULTS.md").stat().st_size>6000 and (B/"BASELINE_COMPARISON.md").stat().st_size>1000
 rc=subprocess.run(["node",str(B/"viewer_smoke.js"),str(B/"operator_viewer.html")],cwd=B,capture_output=True,text=True)
 assert rc.returncode==0,rc.stdout+rc.stderr
 dump(B/"report_verification.json",{"passed":True,"input_results_sha256":sha(B/"results.json"),"comparison_sha256":sha(B/"comparison.json"),"aggregate_rows":96,"effect_rows":60,"operator_cards":2600,"viewer_smoke":rc.stdout.strip(),"scope":"All aggregate/pair/cluster arithmetic, CSV exports, exact recorded viewer cards and all2600 DOM render states."})
 print("COMPARATIVE_REPORT_VERIFICATION_PASS",rc.stdout.strip(),flush=True)
if __name__=="__main__":main()
