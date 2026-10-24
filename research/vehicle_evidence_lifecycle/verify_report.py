"""Independent report coverage, aggregate/effect arithmetic, viewer and artifact binding."""
from lc_common import *
import json,csv,math,statistics,random,gzip,re,subprocess
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
def main():
 root=R/"confirmation_v2";cfg=json.loads((R/"PROTOCOL_V2.json").read_text());trials=json.loads((root/"results.json").read_text())["trials"];report=json.loads((root/"comparison.json").read_text());audit=json.loads((root/"verification.json").read_text())
 assert report["results_sha256"]==audit["results_sha256"]==sha(root/"results.json")
 assert report["verification_sha256"]==sha(root/"verification.json") and audit["passed"]
 assert all(sha(P/n)==h for n,h in json.loads((R/"confirmation_v2_lock.json").read_text()).items())
 expected={(c,m) for c in ["ALL",*cfg["primary_cases"]] for m in cfg["methods"]}
 assert len(report["aggregates"])==len(expected)==65 and {(x["case"],x["method"]) for x in report["aggregates"]}==expected
 csvrows=list(csv.DictReader((root/"aggregate_results.csv").open(newline="")))
 for a,c in zip(report["aggregates"],csvrows):
  rows=[t for t in trials if t["method"]==a["method"] and (a["case"]=="ALL" or t["case"]==a["case"])]
  n=len(rows);selected=sum(x["selected"] for x in rows)
  assert a["trials"]==n and a["contact_trials"]==sum(x["collision_trial"] for x in rows)
  assert a["queries"]==sum(x["queries"] for x in rows) and a["selected"]==selected
  near(a["selected_mae_mps"],sum(x["mae_mps"]*x["selected"] for x in rows)/selected);near(a["availability"],selected/a["queries"])
  near(a["mean_progress_m"],sum(x["progress_m"] for x in rows)/n);near(a["minimum_gap_m"],min(x["minimum_gap_m"] for x in rows))
  near(a["mean_seconds_below_2m"],sum(x["seconds_gap_below_2m"] for x in rows)/n)
  assert a["fallback_queries"]==sum(x["fallback_queries"] for x in rows)
  near(a["maximum_decision_seconds"],max(x["maximum_decision_seconds"] for x in rows))
  assert all(c[k]==str(v) for k,v in a.items())
 keys={(x["right"],x["metric"]) for x in report["paired_effects"]}
 assert len(report["paired_effects"])==12 and keys=={(m,k) for m in cfg["methods"] if m!="joint" for k in ["mae_mps","progress_m","collision_trial"]}
 lookup={(t["seed"],t["profile"],t["case"],t["method"]):t for t in trials}
 for e in report["paired_effects"]:
  means=[]
  for seed in cfg["seeds"]:
   diffs=[lookup[(seed,p,c,"joint")][e["metric"]]-lookup[(seed,p,c,e["right"])][e["metric"]] for p in cfg["profiles"] for c in cfg["primary_cases"]]
   means.append(statistics.mean(diffs));near(e["seed_effects"][str(seed)],means[-1])
  assert e["seed_clusters"]==6 and e["matched_trials"]==216
  near(e["mean_left_minus_right"],statistics.mean(means))
  rng=random.Random(8297);draws=sorted(sum(means[rng.randrange(6)] for _ in range(6))/6 for _ in range(2000))
  near(e["descriptive_cluster_interval_95"][0],draws[49]);near(e["descriptive_cluster_interval_95"][1],draws[1950])
 effects=list(csv.DictReader((root/"paired_effects.csv").open(newline="")));assert len(effects)==12
 for row,e in zip(effects,report["paired_effects"]):
  assert all(row[k]==(json.dumps(v) if isinstance(v,(list,dict)) else str(v)) for k,v in e.items())
 replays=json.loads((root/"operator_replays.json").read_text());assert len(replays)==12 and {r["case"] for r in replays}==set(cfg["primary_cases"])
 for r in replays:
  t=next(t for t in trials if t["trial_id"]==r["trial_id"]);assert t["method"]=="joint" and t["profile"]=="arterial" and t["seed"]==cfg["seeds"][0]
  with gzip.open(root/t["trace"],"rt") as f:cards=[json.loads(x)["operator_card"] for x in f]
  assert cards==r["cards"] and len(cards)==200
  assert all(not({"truth","attack_case","label"}&set(c)) for c in cards)
 html=(root/"operator_viewer.html").read_text();match=re.search(r"const REPLAYS=(.*?);\s*const \$=",html,re.S);assert json.loads(match.group(1))==replays
 code="""const fs=require('fs'),vm=require('vm'),a=require('assert');const html=fs.readFileSync(process.argv[2],'utf8');let nodes={};function node(){return{value:0,textContent:'',children:[],appendChild(x){this.children.push(x)}}}const document={getElementById(id){return nodes[id]||(nodes[id]=node())},createElement(){return node()}};let ctx=vm.createContext({document});vm.runInContext(html.match(/<script>([\\s\\S]*?)<\\/script>/)[1],ctx);let n=0;for(let s=0;s<12;s++)for(let i=0;i<200;i++){vm.runInContext('scenario='+s+';index='+i+';render()',ctx);let card=JSON.parse(vm.runInContext('JSON.stringify(REPLAYS[scenario].cards[index])',ctx));a.deepStrictEqual(JSON.parse(nodes.card.textContent),card);a.equal(Number(nodes.step.value),i);a(nodes.details.textContent.includes(card.actuator_status));n++}nodes.scenario.value=0;nodes.scenario.onchange();nodes.previous.onclick();a.equal(nodes.step.value,0);nodes.next.onclick();a.equal(nodes.step.value,1);nodes.step.value=199;nodes.step.oninput();nodes.next.onclick();a.equal(nodes.step.value,199);console.log('LIFECYCLE_VIEWER_PASS '+n);"""
 script=R/"runtime/viewer_smoke_generated.js";script.write_text(code)
 r=subprocess.run(["node",str(script),str(root/"operator_viewer.html")],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
 dump(root/"report_verification.json",{"passed":True,"results_sha256":sha(root/"results.json"),"comparison_sha256":sha(root/"comparison.json"),"aggregate_rows":65,"effects":12,"viewer_cards":2400,"viewer_smoke":r.stdout.strip()})
 print("LIFECYCLE_REPORT_VERIFIED",r.stdout.strip(),flush=True)
if __name__=="__main__":main()
