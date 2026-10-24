from dev_common import *
import json,math
def main():
 data=json.loads((R/"pilot_results.json").read_text());v=json.loads((R/"pilot_verification.json").read_text())
 assert v["passed"] and v["results_sha256"]==sha(R/"pilot_results.json")
 rows=data["trials"];methods=json.loads((R/"PILOT_PROTOCOL.json").read_text())["methods"]
 text="# Corroborated policy: closed-loop development pilot\n\n"
 text+="52 fresh development trials;13conditions,4policies,one new longitudinal profile,seed2069. This is a small engineering pilot, not independent confirmation. Original540 and1512 studies are preserved.\n\n"
 text+="| Policy | Trials | Contact trials | Selected MAE m/s | Availability | Mean progress m |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
 aggregates=[]
 for m in methods:
  a=[t for t in rows if t["method"]==m];n=sum(t["selected"] for t in a)
  o={"method":m,"trials":len(a),"contact_trials":sum(t["collision_trial"] for t in a),"mae_mps":math.fsum(t["mae_mps"]*t["selected"] for t in a if t["selected"])/n if n else None,"availability":n/sum(t["queries"] for t in a),"mean_progress_m":sum(t["progress_m"] for t in a)/len(a)}
  aggregates.append(o);text+=f"| {m} | {o['trials']} | {o['contact_trials']} | {o['mae_mps']} | {o['availability']:.6f} | {o['mean_progress_m']:.6f} |\n"
 text+="\nThe guard can abstain rather than supply a disputed estimate. Interpret selected MAE with availability, braking/progress and contacts. Shared false-range failures may remain; no global safety or superiority claim follows from this pilot.\n\nThe39 reference-policy trials passed the frozen independent arithmetic audit. All13 integrated guard trials were checked for raw transitions, guard interval membership, controller margins, command authentication, packet-only replay and operator-card fidelity. Verification counts: "+json.dumps(v["counts"])+".\n\nNo evaluator truth or attack label enters the guard. Qualification still depends on honest bounded anchor evidence, accepted-command history, clock and execution premises. The earlier base-fallback prototype's54 new offline errors remain documented; the conservative policy uses conflict abstention instead. Explanation consistency is tested technically; human effectiveness is not evaluated.\n\nNext: address shared context/range corruption separately, then freeze a separate revised-method confirmation protocol. Existing540/1512 results and this pilot must not be relabelled as untouched new-method confirmation.\n"
 (R/"RESULTS.md").write_text(text)
 dump(R/"comparison.json",{"aggregates":aggregates,"pilot_results_sha256":sha(R/"pilot_results.json"),"verification_sha256":sha(R/"pilot_verification.json")})
 assert all(sha(P/n)==h for n,h in json.loads((R/"development_lock.json").read_text()).items())
 dump(R/"development_status.json",{"status":"INTEGRATED_POLICY_DEVELOPMENT_PILOT_VERIFIED","trials":52,"queries":10400,"independent_confirmation":False,"original_sources_unchanged":True,"production_policy_changed":False,"verification_sha256":sha(R/"pilot_verification.json")})
 files={p.relative_to(R).as_posix():sha(p) for p in R.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and "runtime" not in p.relative_to(R).parts and p.name!="manifest.json"}
 dump(R/"manifest.json",files)
 assert all(sha(R/n)==h for n,h in files.items())
 print("DEVELOPMENT_PILOT_REPORTED",json.dumps(aggregates),flush=True)
if __name__=="__main__":main()
