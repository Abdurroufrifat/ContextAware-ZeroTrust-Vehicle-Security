"""Source-bound report for the final lifecycle confirmation; no manuscript."""
from lc_common import *
import json,csv,math,gzip,random,statistics
def summarize(rows):
 n=len(rows);q=sum(x["queries"] for x in rows);s=sum(x["selected"] for x in rows)
 return {"trials":n,"queries":q,"selected":s,"availability":s/q,"selected_mae_mps":math.fsum(x["mae_mps"]*x["selected"] for x in rows if x["selected"])/s if s else None,
  "contact_trials":sum(x["collision_trial"] for x in rows),"mean_progress_m":statistics.mean(x["progress_m"] for x in rows),"minimum_gap_m":min(x["minimum_gap_m"] for x in rows),
  "mean_seconds_below_2m":statistics.mean(x["seconds_gap_below_2m"] for x in rows),"fallback_queries":sum(x["fallback_queries"] for x in rows),
  "maximum_decision_seconds":max(x["maximum_decision_seconds"] for x in rows)}
def main():
 root=R/"confirmation_v2";data=json.loads((root/"results.json").read_text());audit=json.loads((root/"verification.json").read_text());cfg=json.loads((R/"PROTOCOL_V2.json").read_text())
 assert len(data["trials"])==1080 and audit["passed"] and audit["results_sha256"]==sha(root/"results.json")
 assert data["lock_digest"]==audit["lock_digest"]==sha(R/"confirmation_v2_lock.json")
 assert all(sha(P/n)==h for n,h in json.loads((R/"confirmation_v2_lock.json").read_text()).items())
 trials=data["trials"];agg=[]
 for case in ["ALL",*cfg["primary_cases"]]:
  for method in cfg["methods"]:
   rows=[t for t in trials if t["method"]==method and (case=="ALL" or t["case"]==case)]
   agg.append({"case":case,"method":method,**summarize(rows)})
 effects=[]
 for opponent in ("robust_current","nominal_authority","no_learning","always_brake"):
  for metric in ("mae_mps","progress_m","collision_trial"):
   means={}
   for seed in cfg["seeds"]:
    a={(t["profile"],t["case"]):t for t in trials if t["method"]=="joint" and t["seed"]==seed}
    b={(t["profile"],t["case"]):t for t in trials if t["method"]==opponent and t["seed"]==seed}
    assert set(a)==set(b) and len(a)==36
    means[seed]=statistics.mean(a[k][metric]-b[k][metric] for k in a)
   rng=random.Random(8297);values=list(means.values());draws=sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(2000))
   effects.append({"left":"joint","right":opponent,"metric":metric,"matched_trials":216,"seed_clusters":6,"mean_left_minus_right":statistics.mean(values),"seed_effects":means,
      "descriptive_cluster_interval_95":[draws[49],draws[1950]],"scope":"Exploratory six-seed cluster resampling, retaining all fixed profiles/conditions; not independent-query or multiplicity-adjusted inference."})
 dump(root/"comparison.json",{"aggregates":agg,"paired_effects":effects,"results_sha256":sha(root/"results.json"),"verification_sha256":sha(root/"verification.json")})
 for name,rows in [("aggregate_results.csv",agg),("paired_effects.csv",[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in effects])]:
  with (root/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text="# Joint context-history and execution-authority confirmation\n\n"
 text+="1080 new frozen confirmation trials:6fresh seeds,3profiles,12conditions,5methods,216000decisions. Earlier540/1512studies and55/60development studies remain separate. No confirmation result was used to select this policy.\n\n"
 text+="| Method | Trials | Contacts | Selected MAE m/s | Availability | Mean progress m |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
 for row in agg:
  if row["case"]=="ALL":text+=f"| {row['method']} | {row['trials']} | {row['contact_trials']} | {row['selected_mae_mps']:.6f} | {row['availability']:.6f} | {row['mean_progress_m']:.6f} |\n"
 text+="\nInterpret selected error together with availability, uncertainty, progress and fallback. A numeric speed may be reported while its0.5m/s point budget is uncertified; robust control uses the whole retained speed set. These are matched closed-loop scenarios with common exogenous streams, not identical-state estimator queries.\n\n"
 text+="## Contract and limits\n\nInitial gap/lead honesty, bounded uncompromised ego-source subset and execution envelope are premises. Source authentication alone does not establish them. The joint method never narrows safety context from later unqualified source agreement. It conservatively propagates context, uses E=2.5m/s2 worst execution error, and binds each100ms command to an independently checked certificate. Every10ms plant command goes through the actuator watchdog. Missing frames trigger backup. Exact-time delivery and periodic callbacks are modeled; broken controller/actor cadence requires a new qualified session. No network latency tolerance or hard real-time CPU guarantee is claimed.\n\n"
 text+="False bootstrap, authority_exceed and public_motion_innovation are predeclared negative controls. Their contacts, errors and failed state/certificate assumptions remain in the overall and per-condition tables. nominal_authority deliberately uses E=1 and is an assumption ablation. Zero premise-qualified violations is a conditional finite-model result, not immunity to compromised bootstrap, unbounded attacks or a continuous-time vehicle theorem.\n\n"
 text+="## Comparators and ablations\n\nrobust_current uses a scalar S3E/current-context estimate with the same frozen learned nominal proposal, E=2.5stopping filter and certificate gate. This is an enhanced common-task adaptation, not a complete native S3E reproduction. no_learning changes bounded point selection while retaining the same history/authority mechanism. always_brake exposes trivial stopping. The preserved1512-trial study separately compares the secure-safety-filter adaptation. Performance must not be presented as native full-framework superiority.\n\n"
 text+="## Independent audit\n\n"+json.dumps(audit["counts"],indent=2)+"\n\n"
 text+="The audit covers all grid/checkpoints, source/command MACs and attack generation, observer and history updates, raw physical transitions, maximum command/hold boundaries, full-stop margins, certificate/expiry/observation binding, executed watchdog commands, metrics and recorded explanation cards. It fails on any premise-qualified containment or robust-certificate counterexample, including cases whose realized trajectory happens not to collide. Manual frozen forest predictions are checked; the upstream frozen scene adapter retains its separate component evidence. Six-seed intervals are descriptive and not query-independent inference.\n\n"
 text+="## Contribution assessment\n\nSecure estimation, safety filters, provenance, self-triggered control and certificate checking exist in prior work. The bounded candidate contribution is their specific joint authorization contract: unqualified context cannot refresh history, actuator error reduces recoverable authority, command lifetime is derived and bound at the gate, and explanation/audit retains the exact assumptions. PROPERTIES.md gives the discrete derivation and information/authority limits. NOVELTY_SCOPE.md records the primary-source comparison. This is evidence for reviewers to assess; no first-ever originality or journal acceptance certification is made.\n\n"
 text+="The operator viewer replays observation-derived cards and discloses the conditional contract. Technical explanation fidelity is tested; human effectiveness is not evaluated. The simulation scope excludes physical hardware validation by user choice. No manuscript has been written.\n"
 (root/"RESULTS.md").write_text(text,encoding="utf-8")
 replays=[]
 for case in cfg["primary_cases"]:
  t=next(t for t in trials if t["method"]=="joint" and t["case"]==case and t["profile"]=="arterial" and t["seed"]==cfg["seeds"][0])
  with gzip.open(root/t["trace"],"rt") as f:cards=[json.loads(x)["operator_card"] for x in f]
  replays.append({"case":case,"trial_id":t["trial_id"],"cards":cards})
 dump(root/"operator_replays.json",replays)
 payload=json.dumps(replays,separators=(",",":")).replace("<","\\u003c")
 html=(R/"viewer_template.html").read_text().replace("__PAYLOAD__",payload)
 (root/"operator_viewer.html").write_text(html,encoding="utf-8")
 print("LIFECYCLE_REPORT_BUILT",len(agg),len(effects),sum(len(x["cards"]) for x in replays),flush=True)
if __name__=="__main__":main()
