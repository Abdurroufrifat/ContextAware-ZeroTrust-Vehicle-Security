from common import *
install_guard()
import csv,gzip,itertools,random,math
from fractions import Fraction as F
L=R.parent/"vehicle_closed_loop_model"
sys.path.insert(0,str(L))
from loop_core import advance,speed_at,command,target,DT_US
sys.path.insert(0,str(R.parent/"vehicle_conditional_source"))
from conditional_core import choose
sys.path.insert(0,str(R.parent/"vehicle_clock_bound_admission"))
from clock_core import select
from tube_core import admit
assert not (R/"results.json").exists(),"Frozen results exist; use verify_tube.py"
prior_config=json.loads((L/"configuration.json").read_text())
config={**prior_config,"cases":prior_config["cases"]+["actuation_bound_violated","command_history_missing","command_history_tamper"],
"policies":prior_config["policies"]+["command_tube"],"actuation_variants":["exact","bounded_noise"],"execution_error_bound_mmps2":1000,
"execution_noise_interval_mmps2":[-500,500],"execution_noise_seed_offset":10000}
write("configuration.json",config)
paths=[P/"CURRENT_PROJECT_STATUS.md",P/"FINAL_PROJECT_README.md",L/"loop_core.py",L/"results.json",L/"configuration.json",L/"file_manifest.json",
R.parent/"vehicle_clock_bound_admission"/"clock_core.py",R.parent/"vehicle_conditional_source"/"conditional_core.py",
R.parent/"vehicle_can_attack_audit"/"attack_core.py"]
paths += [R/name for name in ("PROTOCOL.md","configuration.json","common.py","tube_core.py","test_tube_core.py","run_study.py")]
lock={str(path.relative_to(P)):sha(path) for path in paths};write("input_lock.json",lock)
noise={};actuation_noise={}
for seed in config["seeds"]:
 rng=random.Random(seed);noise[seed]=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(600)]
 rng=random.Random(seed+10000);actuation_noise[seed]=[rng.randint(-500,500) for _ in range(600)]
def q95(a):
 a=sorted(a);z=(len(a)-1)*.95;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def rat(prefix,value):return {prefix+"_n":value.numerator,prefix+"_d":value.denominator}
prior={tuple(row[k] for k in ("case","nominal_age_us","clock_us","seed","policy")):row for row in json.loads((L/"results.json").read_text())["trials"]}
trials=[];old_checks=0;trace_rows=0
with gzip.open(R/"trace.csv.gz","wt",newline="",encoding="utf-8",compresslevel=3) as f:
 writer=None
 design=itertools.product(config["cases"],config["nominal_age_us"],config["clock_us"],config["seeds"],config["actuation_variants"],config["policies"])
 for trial_id,(case,age,clock,seed,actuation,policy) in enumerate(design):
  v=F(10000);x=F(0);history=[];command_log=[];errors=[];tracking=[];overshoot=[]
  counts=dict(candidate=0,anchor=0,abstain=0)
  if policy=="command_tube":counts["propagated_anchor"]=0
  conditional_bad=outside=unavailable=0
  for step in range(600):
   now=step*DT_US;active=10000000<=now<20000000
   extra=50000 if active and case=="clock_bound_violated" else 0
   sampled=now-age-clock-extra;raw_anchor=speed_at(history,sampled,now,v)
   nc,na=noise[seed][step];candidate=max(0,round(v)+nc);anchor=None if raw_anchor is None else max(0,round(raw_anchor)+na)
   if active:
    if case in ("candidate_plus3","both_plus3"):candidate+=3000
    if case=="candidate_minus3":candidate=max(0,candidate-3000)
    if case=="candidate_dropout":candidate=None
    if case=="anchor_dropout":anchor=None
    if case=="both_plus3" and anchor is not None:anchor+=3000
   assumed=not(active and case=="unknown_bounds")
   anchor_ok=raw_anchor is not None and anchor is not None and abs(F(anchor)-raw_anchor)<=250 and extra==0
   start=max(0,(now-age-clock)//DT_US)
   genuine=command_log[start:]
   supplied=None if active and case=="command_history_missing" else list(genuine)
   if active and case=="command_history_tamper":
    supplied=[(a,b,min(F(2000),c+2000)) for a,b,c in genuine]
   log_intact=supplied is not None and supplied==genuine
   execution_ok=raw_anchor is not None and all(abs(history[k][2]-command_log[k][2])<=1000 for k in range(max(0,sampled//DT_US),len(history)))
   holds=anchor_ok and (log_intact and execution_ok if policy=="command_tube" else True)
   unavailable+=not holds;low=high=None
   if policy=="unguarded_candidate":
    estimate=candidate;selection="candidate" if estimate is not None else "abstain";reason="unguarded_scalar_control"
   elif policy=="static_native":
    value,code=choose(None if candidate is None else candidate/1000,now,None if anchor is None else anchor/1000,.25,now-age,now,assumed)
    selection={0:"abstain",1:"candidate",2:"anchor"}[code];estimate=candidate if code==1 else anchor if code==2 else None
    reason="static_native_under_measurement_time_contract";assert (value is None)==(estimate is None)
   elif policy=="clock_aware":
    decision=select(candidate_mmps=candidate,anchor_mmps=anchor,uncertainty_mmps=250,measured_us=now-age,now_us=now,
       clock_uncertainty_us=clock,acceleration_mmps2=4000,assumptions=assumed)
    estimate=decision["value_mmps"];selection=decision["selection"];reason=decision["reason"]
   else:
    decision=admit(candidate_mmps=candidate,anchor_mmps=anchor,uncertainty_mmps=250,measured_us=now-age,now_us=now,
       clock_uncertainty_us=clock,execution_error_mmps2=1000,commands=supplied,assumptions=assumed)
    estimate=decision["value_mmps"];selection=decision["selection"];reason=decision["reason"]
    if decision["low_n"] is not None:low=F(decision["low_n"],decision["low_d"]);high=F(decision["high_n"],decision["high_d"])
   goal=target(now);requested=command(goal,estimate)
   disturbance=F(actuation_noise[seed][step] if actuation=="bounded_noise" else 0)
   if active and case=="actuation_bound_violated":disturbance=F(2000)
   actual=max(F(-4000),min(F(2000),requested+disturbance))
   newv,newx=advance(v,x,actual,DT_US)
   error=None if estimate is None else abs(F(estimate)-v);counts[selection]+=1
   if error is not None:
    errors.append(float(error)/1000);conditional_bad+=bool(error>500 and assumed and holds);outside+=not(assumed and holds)
   tracking.append(float(abs(v-goal))/1000);overshoot.append(float(max(F(0),v-goal))/1000)
   row=dict(trial_id=trial_id,step=step,case=case,nominal_age_us=age,clock_us=clock,seed=seed,actuation=actuation,policy=policy,
   time_us=now,target_mmps=goal,candidate_mmps=candidate,anchor_mmps=anchor,sampled_us=sampled,reported_us=now-age,
   assumptions=assumed,anchor_contract_holds=anchor_ok,command_history_intact=log_intact,execution_bound_holds=execution_ok,model_holds=holds,
   selection=selection,output_mmps=estimate,reason=reason,**rat("v",v),**rat("x",x),**rat("command",requested),**rat("a",actual),
   **rat("post_v",newv),**rat("post_x",newx),
   anchor_truth_n=raw_anchor.numerator if raw_anchor is not None else None,anchor_truth_d=raw_anchor.denominator if raw_anchor is not None else None,
   interval_low_n=low.numerator if low is not None else None,interval_low_d=low.denominator if low is not None else None,
   interval_high_n=high.numerator if high is not None else None,interval_high_d=high.denominator if high is not None else None)
   if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
   writer.writerow(row);trace_rows+=1
   history.append((now,v,actual));command_log.append((now,now+DT_US,requested));v,x=newv,newx
  saved=dict(trial_id=trial_id,case=case,nominal_age_us=age,clock_us=clock,seed=seed,actuation=actuation,policy=policy,queries=600,
   selections=counts,fallback_steps=counts["abstain"],tracking_mae_mps=math.fsum(tracking)/600,tracking_p95_mps=q95(tracking),
   maximum_overshoot_mps=max(overshoot),final_speed_mps=float(v)/1000,final_position_m=float(x)/1000,selected_queries=len(errors),
   selected_error_p95_mps=q95(errors) if errors else None,selected_error_above500_count=sum(e>.5 for e in errors),selected_error_above2000_count=sum(e>2 for e in errors),
   selected_error_above500_with_model_and_assumptions=conditional_bad,selected_outside_model_or_assumptions=outside,model_unavailable_or_violated_steps=unavailable)
  if actuation=="exact" and (case,age,clock,seed,policy) in prior:
   original=prior[(case,age,clock,seed,policy)]
   for key,value in original.items():
    if key!="trial_id":assert saved[key]==value,(trial_id,key,saved[key],value)
   old_checks+=1
  trials.append(saved)
  if (trial_id+1)%96==0:print("completed",trial_id+1,"trials",flush=True)
assert len(trials)==1056 and trace_rows==633600 and old_checks==288
assert all(t["selected_error_above500_with_model_and_assumptions"]==0 for t in trials if t["policy"] in ("clock_aware","command_tube"))
assert all(sha(P/name)==digest for name,digest in lock.items())
write("results.json",{"trials":trials,"trial_count":1056,"query_rows":trace_rows,"original_trials_reproduced":old_checks,
"scope":"Command-history extension of fixed longitudinal component model","physical_validation_established":False,
"specific_novelty_established":False,"promotion_allowed":False})
print(json.dumps({"trials":1056,"query_rows":trace_rows,"original_trials_reproduced":old_checks},indent=2),flush=True)
