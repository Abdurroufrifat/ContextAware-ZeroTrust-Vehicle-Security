"""Independent trace replay. Does not import the plant, controller or admission modules."""
from common import *
install_guard()
import csv,gzip,itertools,math,random
from fractions import Fraction as F
lock=json.loads((R/"input_lock.json").read_text())
assert all(sha(P/name)==digest for name,digest in lock.items())
result=json.loads((R/"results.json").read_text());config=json.loads((R/"configuration.json").read_text())
assert config["steps"]==600 and config["dt_us"]==50000 and result["trial_count"]==1056
def value(row,name):return F(int(row[name+"_n"]),int(row[name+"_d"]))
def optional(row,name):return None if row[name]=="" else int(row[name])
def p95(vals):
 a=sorted(vals);pos=F(95,100)*(len(a)-1);i=pos.numerator//pos.denominator
 return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(pos-i)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12),(a,b)
def reflected(initial,increments):
 # Closed form for a composition of nonnegative-floor affine maps.
 # max(initial+sum(d), every proper suffix sum, 0).
 suffix=F(0);floor=F(0)
 for d in reversed(increments):floor=max(floor,suffix);suffix+=d
 return max(initial+suffix,floor)
assert reflected(F(50),[F(-100),F(30)])==30
assert reflected(F(50),[F(30),F(-100)])==0
assert reflected(F(50),[])==50
def oracle(candidate,anchor,stamp,now,clock,records,assumed):
 if not assumed:return None,"abstain","assumptions_unknown",None,None
 if anchor is None or stamp<0:return None,"abstain","invalid_or_missing_bound",None,None
 if now-stamp-clock<0 or now-stamp+clock>100000:
  return None,"abstain","uncertain_age_outside_window",None,None
 if records is None:return None,"abstain","invalid_or_missing_history",None,None
 early,late=stamp-clock,stamp+clock
 assert early>=0 and len(records)<=16
 assert early==now or (records and records[0][0]<=early and records[-1][1]==now)
 cuts={early,late}|{b for record in records for b in record[:2] if early<b<late}
 bounds=[]
 for s in cuts:
  ds=[(F(max(0,end-max(start,s)),1000000),cmd) for start,end,cmd in records]
  lower=reflected(F(max(0,anchor-250)),[max(F(-4000),cmd-1000)*dt for dt,cmd in ds])
  upper=reflected(F(anchor+250),[min(F(2000),cmd+1000)*dt for dt,cmd in ds])
  bounds.append((lower,upper))
 low=min(a for a,b in bounds);high=max(b for a,b in bounds)
 acceptable=lambda e:high-500<=e<=low+500
 if candidate is not None and acceptable(candidate):return candidate,"candidate","candidate_inside_reachable_budget",low,high
 center=round((low+high)/2)
 if acceptable(center):return center,"propagated_anchor","propagated_anchor_inside_budget",low,high
 return None,"abstain","budget_exhausted",low,high
checks=conditional_total=0;old_checks=0
prior={tuple(t[k] for k in ("case","nominal_age_us","clock_us","seed","policy")):t for t in json.loads((R.parent/"vehicle_closed_loop_model"/"results.json").read_text())["trials"]}
with gzip.open(R/"trace.csv.gz","rt",newline="",encoding="utf-8") as f:
 reader=iter(csv.DictReader(f))
 design=itertools.product(config["cases"],config["nominal_age_us"],config["clock_us"],config["seeds"],config["actuation_variants"],config["policies"])
 for tid,(case,age,clock,seed,actuation,policy) in enumerate(design):
  saved=result["trials"][tid]
  for key,expected in (("trial_id",tid),("case",case),("nominal_age_us",age),("clock_us",clock),("seed",seed),("actuation",actuation),("policy",policy)):assert saved[key]==expected
  v=F(10000);x=F(0);history=[];commands=[];rng=random.Random(seed);arng=random.Random(seed+10000)
  tracking=[];errors=[];overshoot=[];counts=dict(candidate=0,anchor=0,abstain=0)
  if policy=="command_tube":counts["propagated_anchor"]=0
  model_bad=outside=conditional_bad=0
  for step in range(600):
   row=next(reader);now=step*50000;active=10000000<=now<20000000
   for key,expected in (("trial_id",str(tid)),("step",str(step)),("time_us",str(now)),("case",case),("policy",policy),("actuation",actuation),("seed",str(seed)),("nominal_age_us",str(age)),("clock_us",str(clock))):assert row[key]==expected
   assert value(row,"v")==v and value(row,"x")==x
   extra=50000 if active and case=="clock_bound_violated" else 0
   sampled=now-age-clock-extra
   if sampled<0:truth=None
   else:
    index=sampled//50000;start,oldv,a=history[index]
    truth=max(F(0),oldv+a*F(sampled-start,1000000))
   assert int(row["sampled_us"])==sampled and int(row["reported_us"])==now-age
   if truth is None:assert row["anchor_truth_n"]==row["anchor_truth_d"]==""
   else:assert value(row,"anchor_truth")==truth
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
   anchor_ok=truth is not None and anchor is not None and abs(F(anchor)-truth)<=250 and extra==0
   genuine=commands[max(0,(now-age-clock)//50000):]
   supplied=None if active and case=="command_history_missing" else list(genuine)
   if active and case=="command_history_tamper":supplied=[(a,b,min(F(2000),c+2000)) for a,b,c in genuine]
   intact=supplied is not None and supplied==genuine
   execution_ok=truth is not None and all(abs(history[k][2]-commands[k][2])<=1000 for k in range(max(0,sampled//50000),len(history)))
   holds=anchor_ok and (intact and execution_ok if policy=="command_tube" else True)
   for key,expected in (("assumptions",assumed),("anchor_contract_holds",anchor_ok),("command_history_intact",intact),("execution_bound_holds",execution_ok),("model_holds",holds)):assert row[key]==str(expected)
   low=high=None
   if policy=="unguarded_candidate":selected=candidate;selection="candidate" if candidate is not None else "abstain"
   elif policy=="static_native":
    if not assumed or anchor is None or now-age<0:selected=None;selection="abstain"
    elif candidate is not None and abs(candidate/1000-anchor/1000)+.25<=.5:selected=candidate;selection="candidate"
    else:selected=anchor;selection="anchor"
   elif policy=="clock_aware":
    radius=250*1000000+4000*(age+clock)
    if not assumed or anchor is None or now-age<0 or age-clock<0 or age+clock>100000 or radius>500000000:selected=None;selection="abstain"
    elif candidate is not None and abs(candidate-anchor)*1000000+radius<=500000000:selected=candidate;selection="candidate"
    else:selected=anchor;selection="anchor"
   else:
    selected,selection,reason,low,high=oracle(candidate,anchor,now-age,now,clock,supplied,assumed)
    assert row["reason"]==reason
   for name,expected in (("interval_low",low),("interval_high",high)):
    if expected is None:assert row[name+"_n"]==row[name+"_d"]==""
    else:assert value(row,name)==expected
   assert optional(row,"output_mmps")==selected and row["selection"]==selection
   goal=15000 if now<8000000 else 22000 if now<18000000 else 10000
   requested=F(-2000) if selected is None else min(F(2000),max(F(-4000),F(4*(goal-selected),5)))
   disturbance=F(arng.randint(-500,500)) # Advance independent RNG identically for both variants.
   if actuation=="exact":disturbance=F(0)
   if active and case=="actuation_bound_violated":disturbance=F(2000)
   a=max(F(-4000),min(F(2000),requested+disturbance))
   assert int(row["target_mmps"])==goal and value(row,"command")==requested and value(row,"a")==a
   nv=max(F(0),v+a/20);dx=(v+nv)/40 if v+a/20>=0 else -v*v/(2*a);nx=x+dx
   assert value(row,"post_v")==nv and value(row,"post_x")==nx and nx>=x and nv>=0
   tracking.append(abs(v-goal)/1000);overshoot.append(max(F(0),v-goal)/1000);counts[selection]+=1;model_bad+=not holds
   if selected is not None:
    error=abs(F(selected)-v);errors.append(error/1000);conditional_bad+=bool(error>500 and assumed and holds);outside+=not(assumed and holds)
    if policy=="command_tube" and holds and assumed:assert low<=v<=high
   history.append((now,v,a));commands.append((now,now+50000,requested));v,x=nv,nx;checks+=1
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
  if policy in ("clock_aware","command_tube"):assert conditional_bad==0;conditional_total+=conditional_bad
  key=(case,age,clock,seed,policy)
  if actuation=="exact" and key in prior:
   for k,before in prior[key].items():
    if k!="trial_id":assert saved[k]==before
   old_checks+=1
  if (tid+1)%96==0:print("verified",tid+1,"trials",flush=True)
 assert next(reader,None) is None
assert checks==633600 and old_checks==288 and conditional_total==0
assert all(sha(P/name)==digest for name,digest in lock.items())
assert all(result[k] is False for k in ("physical_validation_established","specific_novelty_established","promotion_allowed"))
validation={"passed":True,"trials":1056,"feedback_steps_replayed":checks,"source_and_selection_rows_checked":checks,
"original_trials_reproduced":old_checks,"independent_interval_oracle":"reverse cumulative suffix closed form; no tube_core imports",
"conditional_error_above500_count":conditional_total,"input_identities_checked":len(lock),"physical_validation_established":False}
write("replay_validation.json",validation);print(json.dumps(validation,indent=2),flush=True)
