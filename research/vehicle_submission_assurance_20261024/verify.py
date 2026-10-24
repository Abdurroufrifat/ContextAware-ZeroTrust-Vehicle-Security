"""Independent supplement checks, browser exercise and preservation seal."""
from pathlib import Path
import json,hashlib,gzip,math,statistics,re,subprocess,datetime
S=Path(__file__).resolve().parent;P=S.parent.parent;R=P/"research/vehicle_evidence_lifecycle";B=R/"confirmation_v2"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
def q(a,p):
 a=sorted(a);return a[math.ceil(len(a)*p)-1]
def main():
 assert not (S/"SUPPLEMENT_COMPLETE.json").exists(),"Already verified; no repeat."
 a=load(S/"analysis.json");rows=load(S/"per_trial.json");original=load(B/"results.json")["trials"];inputs=load(S/"input_manifest.json")
 assert len(rows)==len(original)==1080 and a["queries"]==216000
 assert all(sha(P/n)==h for n,h in inputs.items())
 times={m:[] for m in {r["method"] for r in rows}};cold={m:[] for m in times};warm={m:[] for m in times}
 for t,r in zip(original,rows):
  assert all(r[k]==v for k,v in t.items())
  n=point=accepted=uncertified=over=0
  with gzip.open(B/t["trace"],"rt") as f:
   for line in f:
    x=json.loads(line);c=x["operator_card"];d=x["decision_wall_seconds"]
    point+=c["point_budget_satisfied"] is True
    accepted+=x["actuator_status"]=="authenticated"
    uncertified+=(x["actuator_status"]=="authenticated" and c["point_budget_satisfied"] is not True)
    over+=d>.1;n+=1;times[t["method"]].append(d);(cold if x["step"]==0 else warm)[t["method"]].append(d)
  assert [r[k] for k in ["queries","point_certified","command_accepted","accepted_with_uncertified_point","over_100ms"]]==[n,point,accepted,uncertified,over]
 for timing in a["timing"]:
  values={"all":times,"first_query":cold,"later_queries":warm}[timing["scope"]][timing["method"]]
  assert timing["queries"]==len(values) and timing["over_100ms"]==sum(x>.1 for x in values)
  for key,p in [("p50_ms",.5),("p95_ms",.95),("p99_ms",.99)]:near(timing[key],q(values,p)*1000)
  near(timing["max_ms"],max(values)*1000)
 negative={"initial_false_context","authority_exceed","public_motion_innovation"}
 assert len(a["aggregates"])==60
 for x in a["aggregates"]:
  chosen=[r for r in rows if r["method"]==x["method"] and (x["profile"]=="ALL" or r["profile"]==x["profile"]) and (x["scope"]=="all" or (r["case"] not in negative if x["scope"]=="main" else r["case"] in negative))]
  assert len(chosen)==x["trials"]
  assert sum(r["collision_trial"] for r in chosen)==x["contacts"]
  near(x["mean_progress_m"],sum(r["progress_m"] for r in chosen)/len(chosen))
  near(x["point_budget_coverage"],sum(r["point_certified"] for r in chosen)/sum(r["queries"] for r in chosen))
  near(x["contact_fraction"],sum(r["collision_trial"] for r in chosen)/len(chosen))
 cfg=load(R/"PROTOCOL_V2.json")
 assert len(a["effects"])==24
 for e in a["effects"]:
  means=[]
  for seed in cfg["seeds"]:
   chosen=[t for t in original if t["seed"]==seed and (e["scope"]=="all" or (t["case"] not in negative if e["scope"]=="main" else t["case"] in negative))]
   groups=[[t for t in chosen if t["method"]==m] for m in ["joint",e["right"]]]
   metric="collision_trial" if e["metric"]=="contact_fraction" else "progress_m"
   means.append(statistics.mean(t[metric] for t in groups[0])-statistics.mean(t[metric] for t in groups[1]))
  for x,y in zip(e["seed_effects"],means):near(x,y)
  near(e["mean_left_minus_right"],statistics.mean(means))
  import random
  rng=random.Random(8297);draws=sorted(sum(means[rng.randrange(6)] for _ in range(6))/6 for _ in range(2000))
  near(e["descriptive_95_interval"][0],draws[49]);near(e["descriptive_95_interval"][1],draws[1950])
 source=load(B/"operator_replays.json");viewer=load(S/"operator_evidence.json");cards=0;uncertified_cards=0
 for old,new in zip(source,viewer):
  assert old["case"]==new["case"] and old["trial_id"]==new["trial_id"] and len(old["cards"])==len(new["cards"])==200
  for c,wrapped in zip(old["cards"],new["cards"]):
   assert c==wrapped["source_card"];e=wrapped["explanation"];cards+=1
   assert e["speed_budget_certified"]==(c["point_budget_satisfied"] is True)
   assert e["command_gate_accepted"]==(c["actuator_status"]=="authenticated")
   if c["point_budget_satisfied"] is not True:
    uncertified_cards+=1;assert "no certified" in e["speed_text"]
   else:assert "Conditional" in e["speed_text"]
   near(e["requested_mps2"],c["requested_command_mmps2"]/1000);near(e["accepted_mps2"],c["accepted_command_mmps2"]/1000)
   assert e["hold_ms"]==c["certificate"]["issued_hold_steps"]*10
   assert e["maximum_hold_ms"]==c["certificate"]["max_hold_steps"]*10
   assert e["execution_error_mps2"]==c["certificate"]["execution_error"]
   assert not e["human_effectiveness_evaluated"] and "does not establish" in e["measurement_truth_text"]
 assert cards==2400
 html=(S/"operator_assurance_viewer.html").read_text(encoding="utf-8")
 assert json.loads(re.search(r"const REPLAYS=(.*?);\s*const \$=",html,re.S).group(1))==viewer
 runtime=S/"runtime";browser=runtime/("browser-"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S"));browser.mkdir()
 script="""<script>try{let checked=0,warnings=0;
 for(let s=0;s<REPLAYS.length;s++)for(let i=0;i<200;i++){scenario=s;index=i;render();let r=REPLAYS[s].cards[i];
 if(JSON.stringify(JSON.parse(document.getElementById('card').textContent))!==JSON.stringify(r.source_card))throw Error('card');
 if(document.getElementById('command').textContent!==r.explanation.command_text)throw Error('command');
 if(document.getElementById('speed').textContent!==r.explanation.speed_text)throw Error('speed');
 if(!r.source_card.point_budget_satisfied){if(!document.getElementById('speed').textContent.includes('no certified'))throw Error('missing warning');warnings++}
 checked++;}
 document.getElementById('scenario').value=0;document.getElementById('scenario').onchange();document.getElementById('previous').click();if(Number(document.getElementById('step').value)!==0)throw Error('previous');
 document.getElementById('next').click();if(Number(document.getElementById('step').value)!==1)throw Error('next');
 document.getElementById('step').value=199;document.getElementById('step').oninput();document.getElementById('next').click();if(Number(document.getElementById('step').value)!==199)throw Error('last');
 document.getElementById('scenario').value=0;document.getElementById('scenario').onchange();document.documentElement.setAttribute('data-assurance-check','PASS_'+checked+'_'+warnings);
 }catch(e){document.documentElement.setAttribute('data-assurance-check','FAIL_'+e.message)}</script>"""
 target=browser/"viewer_test.html";target.write_text(html.replace("</html>",script+"</html>"),encoding="utf-8")
 cmd=[r"C:\Program Files\Google\Chrome\Application\chrome.exe","--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check","--disable-background-networking","--disable-extensions","--disable-sync","--user-data-dir="+str(browser/"profile"),"--window-size=1280,1100","--virtual-time-budget=5000","--screenshot="+str(browser/"viewer.png"),"--dump-dom",target.as_uri()]
 with (browser/"dom.html").open("w",encoding="utf-8") as out,(browser/"browser.log").open("w",encoding="utf-8") as err:
  run=subprocess.run(cmd,stdout=out,stderr=err,timeout=60)
 assert run.returncode==0
 assert f'data-assurance-check="PASS_{cards}_{uncertified_cards}"' in (browser/"dom.html").read_text(encoding="utf-8")
 assert (browser/"viewer.png").stat().st_size>1000
 # Verify the old release and every frozen input remain untouched.
 old_manifest=load(R/"package_manifest.json")["files"];old_lock=load(R/"confirmation_v2_lock.json")
 for names in [old_manifest,old_lock]:
  for name in names:
   low=name.casefold();assert not("library03" in low or "rav4" in low or "vehicle_final_evaluation" in low or ("urban02" in low and "night" in low))
  assert all(sha(P/n)==h for n,h in names.items())
 assert load(R/"final_seal.json")["package_manifest_sha256"]==sha(R/"package_manifest.json")
 # Record the meaningful eight checks against the final unchanged source.
 import sys
 unit=subprocess.run([sys.executable,"-m","unittest","-v","test_assurance"],cwd=S,capture_output=True,text=True)
 (runtime/"tests.log").write_text(unit.stdout+unit.stderr,encoding="utf-8");assert unit.returncode==0 and "Ran 8 tests" in unit.stderr
 record={"passed":True,"trials":1080,"queries":216000,"aggregate_rows":60,"timing_rows":15,"paired_effects":24,"viewer_cards":cards,"uncertified_speed_warnings_checked":uncertified_cards,"unit_tests":8,"browser_returncode":0,"browser_screenshot":str(browser/"viewer.png"),"original_release_files_rehashed":len(old_manifest),"original_frozen_inputs_rehashed":len(old_lock),"original_results_sha256":sha(B/"results.json"),"input_manifest_sha256":sha(S/"input_manifest.json"),"timing_is_deployment_qualification":False,"human_effectiveness_evaluated":False}
 (S/"verification.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
 files=[p for p in S.rglob("*") if p.is_file() and not any(k in p.relative_to(S).parts for k in ["runtime","__pycache__"]) and p.name not in ["package_manifest.json","SUPPLEMENT_COMPLETE.json"]]
 bindings={str(p.relative_to(S)):sha(p) for p in sorted(files)}
 (S/"package_manifest.json").write_text(json.dumps(bindings,indent=2),encoding="utf-8")
 assert all(sha(S/n)==h for n,h in bindings.items())
 (S/"SUPPLEMENT_COMPLETE.json").write_text(json.dumps({"status":"COMPLETE_FOR_SECONDARY_ASSURANCE_SCOPE","verification_sha256":sha(S/"verification.json"),"package_manifest_sha256":sha(S/"package_manifest.json"),"original_release_unchanged":True,"new_vehicle_experiments":0,"policy_changes":0,"files":len(bindings)},indent=2),encoding="utf-8")
 print("SUPPLEMENT_VERIFIED",json.dumps(record),flush=True)
if __name__=="__main__":main()
