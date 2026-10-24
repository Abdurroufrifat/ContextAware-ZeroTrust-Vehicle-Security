"""Independent recurrence, source and selection replay; no model/selector imports."""
from common import *
install_guard()
import csv,gzip,itertools,math,random
from fractions import Fraction as F
lock=json.loads((R/"input_lock.json").read_text())
assert all(sha(P/name)==digest for name,digest in lock.items())
result=json.loads((R/"results.json").read_text())
config=json.loads((R/"configuration.json").read_text())
assert config["steps"]==600 and config["dt_us"]==50000 and result["trial_count"]==288
def value(row,name):return F(int(row[name+"_n"]),int(row[name+"_d"]))
def optional(row,name):return None if row[name]=="" else int(row[name])
def p95(vals):
 ordered=sorted(vals);pos=F(95,100)*(len(ordered)-1);i=pos.numerator//pos.denominator
 return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(pos-i)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12),(a,b)
checks=0;clock_conditional_bad=0;clock_zero_output_trials=0
with gzip.open(R/"trace.csv.gz","rt",newline="",encoding="utf-8") as f:
 reader=iter(csv.DictReader(f))
 design=itertools.product(config["cases"],config["nominal_age_us"],config["clock_us"],config["seeds"],config["policies"])
 for trial_id,(case,age,clock,seed,policy) in enumerate(design):
  saved=result["trials"][trial_id]
  for key,expected in (("trial_id",trial_id),("case",case),("nominal_age_us",age),("clock_us",clock),("seed",seed),("policy",policy)):
   assert saved[key]==expected
  v=F(10000);x=F(0);history=[];rng=random.Random(seed)
  tracking=[];errors=[];overshoot=[];counts=dict(candidate=0,anchor=0,abstain=0)
  model_bad=outside=conditional_bad=0
  for step in range(600):
   row=next(reader);now=step*50000;active=10000000<=now<20000000
   assert int(row["trial_id"])==trial_id and int(row["step"])==step and int(row["time_us"])==now
   for key,expected in (("case",case),("policy",policy),("seed",str(seed)),("nominal_age_us",str(age)),("clock_us",str(clock))):assert row[key]==expected
   assert value(row,"v")==v and value(row,"x")==x
   extra=50000 if active and case=="clock_bound_violated" else 0
   sampled=now-age-clock-extra
   if sampled<0:truth=None
   else:
    index=sampled//50000;start,prior,action=history[index]
    truth=max(F(0),prior+action*F(sampled-start,1000000))
   assert int(row["sampled_us"])==sampled and int(row["reported_us"])==now-age
   if truth is None:assert row["anchor_truth_n"]==row["anchor_truth_d"]==""
   else:assert F(int(row["anchor_truth_n"]),int(row["anchor_truth_d"]))==truth
   nc,na=rng.randint(-100,100),rng.randint(-100,100)
   candidate=max(0,round(v)+nc);anchor=None if truth is None else max(0,round(truth)+na)
   if active:
    if case in ("candidate_plus3","both_plus3"):candidate+=3000
    if case=="candidate_minus3":candidate=max(0,candidate-3000)
    if case=="candidate_dropout":candidate=None
    if case=="anchor_dropout":anchor=None
    if case=="both_plus3" and anchor is not None:anchor+=3000
   assert optional(row,"candidate_mmps")==candidate and optional(row,"anchor_mmps")==anchor
   assumed=not(active and case=="unknown_bounds")
   holds=truth is not None and anchor is not None and abs(F(anchor)-truth)<=250 and extra==0
   assert row["assumptions"]==str(assumed) and row["model_holds"]==str(holds)
   if policy=="unguarded_candidate":
    selected=candidate;selection="candidate" if candidate is not None else "abstain"
   elif policy=="static_native":
    if not assumed or anchor is None or now-age<0:selected=None;selection="abstain"
    elif candidate is not None and abs(candidate/1000-anchor/1000)+.25<=.5:selected=candidate;selection="candidate"
    else:selected=anchor;selection="anchor"
   else:
    radius_n=250*1000000+4000*(age+clock)
    if not assumed or anchor is None or now-age<0 or age-clock<0 or age+clock>100000 or radius_n>500000000:
     selected=None;selection="abstain"
    elif candidate is not None and abs(candidate-anchor)*1000000+radius_n<=500000000:
     selected=candidate;selection="candidate"
    else:selected=anchor;selection="anchor"
   assert optional(row,"output_mmps")==selected and row["selection"]==selection
   goal=15000 if now<8000000 else 22000 if now<18000000 else 10000
   assert int(row["target_mmps"])==goal
   acceleration=F(-2000) if selected is None else min(F(2000),max(F(-4000),F(4*(goal-selected),5)))
   assert value(row,"a")==acceleration
   next_v=max(F(0),v+acceleration/20)
   # Equivalent trapezoid or stopping-distance formulation, independent of advance().
   dx=(v+next_v)/40 if v+acceleration/20>=0 else -v*v/(2*acceleration)
   next_x=x+dx
   assert value(row,"post_v")==next_v and value(row,"post_x")==next_x
   assert next_x>=x and next_v>=0
   tracking.append(abs(v-goal)/1000);overshoot.append(max(F(0),v-goal)/1000)
   counts[selection]+=1;model_bad+=not holds
   if selected is not None:
    error=abs(F(selected)-v);errors.append(error/1000)
    conditional_bad+=bool(error>500 and assumed and holds);outside+=not(assumed and holds)
   history.append((now,v,acceleration));v,x=next_v,next_x;checks+=1
  assert saved["queries"]==600 and saved["selections"]==counts
  assert saved["fallback_steps"]==counts["abstain"] and saved["selected_queries"]==len(errors)
  close(sum(tracking,F(0))/600,saved["tracking_mae_mps"]);close(p95(tracking),saved["tracking_p95_mps"])
  close(max(overshoot),saved["maximum_overshoot_mps"]);close(v/1000,saved["final_speed_mps"]);close(x/1000,saved["final_position_m"])
  if errors:close(p95(errors),saved["selected_error_p95_mps"])
  else:assert saved["selected_error_p95_mps"] is None
  assert saved["selected_error_above500_count"]==sum(e>F(1,2) for e in errors)
  assert saved["selected_error_above2000_count"]==sum(e>2 for e in errors)
  assert saved["selected_error_above500_with_model_and_assumptions"]==conditional_bad
  assert saved["selected_outside_model_or_assumptions"]==outside
  assert saved["model_unavailable_or_violated_steps"]==model_bad
  if policy=="clock_aware":
   clock_conditional_bad+=conditional_bad;clock_zero_output_trials+=not errors
 assert next(reader,None) is None
assert checks==172800 and clock_conditional_bad==0
assert clock_zero_output_trials>0
assert all(sha(P/name)==digest for name,digest in lock.items())
assert all(result[name] is False for name in ("physical_validation_established","full_authenticated_gateway_reproduced","specific_novelty_established","promotion_allowed"))
print(json.dumps({"passed":True,"trials":288,"feedback_steps_replayed":checks,"source_and_selection_rows_checked":checks,
"clock_conditional_error_above500_count":clock_conditional_bad,"clock_trials_with_no_selected_values":clock_zero_output_trials,
"input_identities_checked":len(lock),"physical_validation_established":False},indent=2))
