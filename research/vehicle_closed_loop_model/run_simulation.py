from common import *
install_guard()
import csv,gzip,itertools,random,math
from fractions import Fraction as F
from loop_core import advance,speed_at,command,target,DT_US
sys.path.insert(0,str(R.parent/"vehicle_conditional_source"))
from conditional_core import choose
sys.path.insert(0,str(R.parent/"vehicle_clock_bound_admission"))
from clock_core import select
assert not (R/"results.json").exists(),"Results frozen; use verify_loop.py"
config={"steps":600,"dt_us":50000,"seeds":[17,29,43],"nominal_age_us":[50000,100000],"clock_us":[0,1000],
"cases":["normal","candidate_plus3","candidate_minus3","candidate_dropout","anchor_dropout","both_plus3","unknown_bounds","clock_bound_violated"],
"policies":["unguarded_candidate","static_native","clock_aware"],"attack_start_us":10000000,"attack_end_us":20000000,
"uncertainty_mmps":250,"acceleration_bound_mmps2":4000,"bias_mmps":3000,"fallback_acceleration_mmps2":-2000}
write("configuration.json",config)
paths=[P/"CURRENT_PROJECT_STATUS.md",P/"FINAL_PROJECT_README.md",
R.parent/"vehicle_clock_bound_admission"/"clock_core.py",R.parent/"vehicle_clock_bound_admission"/"file_manifest.json",
R.parent/"vehicle_conditional_source"/"conditional_core.py",R.parent/"vehicle_can_attack_audit"/"attack_core.py",
R.parent/"vehicle_equal_input_ablation"/"file_manifest.json"]
paths += [R/name for name in ("PROTOCOL.md","configuration.json","common.py","loop_core.py","test_loop_core.py","run_simulation.py")]
lock={str(path.relative_to(P)):sha(path) for path in paths};write("input_lock.json",lock)
noise={}
for seed in config["seeds"]:
 rng=random.Random(seed);noise[seed]=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(600)]
def q95(values):
 ordered=sorted(values);at=(len(ordered)-1)*.95;i=int(at)
 return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(at-i)
def rational(prefix,value):
 return {prefix+"_n":value.numerator,prefix+"_d":value.denominator}
trials=[];trace_rows=0
with gzip.open(R/"trace.csv.gz","wt",newline="",encoding="utf-8",compresslevel=3) as stream:
 writer=None
 for trial_id,(case,age,clock,seed,policy) in enumerate(itertools.product(config["cases"],config["nominal_age_us"],config["clock_us"],config["seeds"],config["policies"])):
  v=F(10000);x=F(0);history=[];tracking=[];selected_errors=[];overshoot=[];selection_counts=dict(candidate=0,anchor=0,abstain=0)
  conditional_bad=0;outside_model_selected=0;model_violation_steps=0;fallback=0
  for step in range(600):
   now=step*DT_US;active=10000000<=now<20000000
   extra=50000 if active and case=="clock_bound_violated" else 0
   sampled=now-age-clock-extra
   original_anchor=speed_at(history,sampled,now,v)
   nc,na=noise[seed][step]
   candidate=max(0,round(v)+nc)
   anchor=None if original_anchor is None else max(0,round(original_anchor)+na)
   if active:
    if case in ("candidate_plus3","both_plus3"):candidate+=3000
    if case=="candidate_minus3":candidate=max(0,candidate-3000)
    if case=="candidate_dropout":candidate=None
    if case=="anchor_dropout":anchor=None
    if case=="both_plus3" and anchor is not None:anchor+=3000
   assumed=not (active and case=="unknown_bounds")
   model_holds=original_anchor is not None and anchor is not None and abs(F(anchor)-original_anchor)<=250 and extra==0
   model_violation_steps+=not model_holds
   if policy=="unguarded_candidate":
    estimate=candidate;selection="candidate" if estimate is not None else "abstain";reason="unguarded_scalar_control"
   elif policy=="static_native":
    value,code=choose(None if candidate is None else candidate/1000,now,None if anchor is None else anchor/1000,.25,now-age,now,assumed)
    selection={0:"abstain",1:"candidate",2:"anchor"}[code]
    estimate=candidate if code==1 else anchor if code==2 else None;reason="static_native_under_measurement_time_contract"
    assert (value is None)==(estimate is None)
   else:
    decision=select(candidate_mmps=candidate,anchor_mmps=anchor,uncertainty_mmps=250,measured_us=now-age,now_us=now,
       clock_uncertainty_us=clock,acceleration_mmps2=4000,assumptions=assumed)
    estimate=decision["value_mmps"];selection=decision["selection"];reason=decision["reason"]
   goal=target(now);a=command(goal,estimate);newv,newx=advance(v,x,a,DT_US)
   error=None if estimate is None else abs(F(estimate)-v)
   selection_counts[selection]+=1;fallback+=estimate is None
   if error is not None:
    selected_errors.append(float(error)/1000)
    conditional_bad+=bool(error>500 and assumed and model_holds)
    outside_model_selected+=not (assumed and model_holds)
   tracking.append(float(abs(v-goal))/1000);overshoot.append(float(max(F(0),v-goal))/1000)
   row=dict(trial_id=trial_id,step=step,case=case,nominal_age_us=age,clock_us=clock,seed=seed,policy=policy,
    time_us=now,target_mmps=goal,candidate_mmps=candidate,anchor_mmps=anchor,sampled_us=sampled,reported_us=now-age,
    assumptions=assumed,model_holds=model_holds,selection=selection,output_mmps=estimate,reason=reason,
    **rational("v",v),**rational("x",x),**rational("a",a),**rational("post_v",newv),**rational("post_x",newx),
    anchor_truth_n=original_anchor.numerator if original_anchor is not None else None,
    anchor_truth_d=original_anchor.denominator if original_anchor is not None else None)
   if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
   writer.writerow(row);trace_rows+=1
   history.append((now,v,a));v,x=newv,newx
  trials.append(dict(trial_id=trial_id,case=case,nominal_age_us=age,clock_us=clock,seed=seed,policy=policy,queries=600,
    selections=selection_counts,fallback_steps=fallback,tracking_mae_mps=math.fsum(tracking)/600,
    tracking_p95_mps=q95(tracking),maximum_overshoot_mps=max(overshoot),final_speed_mps=float(v)/1000,
    final_position_m=float(x)/1000,selected_queries=len(selected_errors),
    selected_error_p95_mps=q95(selected_errors) if selected_errors else None,
    selected_error_above500_count=sum(e>.5 for e in selected_errors),selected_error_above2000_count=sum(e>2 for e in selected_errors),
    selected_error_above500_with_model_and_assumptions=conditional_bad,selected_outside_model_or_assumptions=outside_model_selected,
    model_unavailable_or_violated_steps=model_violation_steps))
  if (trial_id+1)%36==0:print("completed",trial_id+1,"trials",flush=True)
assert len(trials)==288 and trace_rows==172800
assert all(t["selected_error_above500_with_model_and_assumptions"]==0 for t in trials if t["policy"]=="clock_aware")
assert all(sha(P/name)==digest for name,digest in lock.items())
write("results.json",{"trials":trials,"trial_count":288,"query_rows":trace_rows,"scope":"Fixed simple longitudinal model; numerical component controls",
"physical_validation_established":False,"full_authenticated_gateway_reproduced":False,"specific_novelty_established":False,"promotion_allowed":False})
print(json.dumps({"trials":288,"query_rows":trace_rows,"physical_validation_established":False},indent=2),flush=True)
