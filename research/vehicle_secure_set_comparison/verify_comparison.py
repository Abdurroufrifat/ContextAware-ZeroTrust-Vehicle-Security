"""Independent endpoint-cell oracle, reference scoring and cohort reconstruction.
Does not import estimator, replay adapter, score.py or any production gateway.
"""
from ssc_common import *
from fractions import Fraction as F
import gzip,json,bisect,xml.etree.ElementTree as ET,itertools
result=json.loads((R/"results.json").read_text())
assert result["trajectories"]==144 and result["queries"]==28800 and result["new_estimator_decisions"]==57600
assert result["scalar_s3e_adaptation"] is True
assert all(result[k] is False for k in ("native_full_s3e_reproduction","new_closed_loop_results","physical_vehicle_validation","new_algorithm_principle_established"))
lock=json.loads((R/"input_lock.json").read_text());assert len(lock)==298 and all(sha(P/n)==h for n,h in lock.items())
def union(points):
 out=[]
 for a,b in sorted(points):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return out
def member(segments,x):return any(a<=x<=b for a,b in segments)
def sensor_bounds(row):
 out=[]
 for role in ("candidate","anchor"):
  obj=row["packets"][role]
  if obj is None or row["decision"][role+"_status"]!="authenticated" or (role=="candidate" and row["risk"]!="normal"):
   out.append(None);continue
  body=json.loads(obj["wire"])["body"];y=body["speed_mmps"]
  if role=="candidate":out.append((F(max(0,y-250)),F(y+250)));continue
  age=row["now_us"]-body["measured_us"]
  if age<5000 or age+5000>100000:out.append(None);continue
  elapsed=F(age+5000,1000000)
  out.append((max(F(0),y-250-4000*elapsed),y+250+2000*elapsed))
 return out
def oracle(state,delta,previous_command,obs,mode):
 dt=F(delta,1000000)
 lows,highs=(-4000,2000) if mode=="s3e_global" else (max(-4000,previous_command-1000),min(2000,previous_command+1000))
 predicted=union([(max(F(0),a+lows*dt),max(F(0),b+highs*dt)) for a,b in state])
 if any(x is None for x in obs):return predicted
 # Decompose at every boundary, then test each endpoint and each open cell.
 boundaries=sorted({x for pair in predicted+obs for x in pair})
 valid=lambda x:member(predicted,x) and member(obs,x)
 pieces=[(x,x) for x in boundaries if valid(x)]
 pieces += [(a,b) for a,b in zip(boundaries,boundaries[1:]) if valid((a+b)/2)]
 return union(pieces)
expected=[];count=0;transitions=0;truth_queries=0
for tid in range(144):
 trial=OLD/"simulator_outputs"/f"trial_{tid:03d}"
 with gzip.open(trial/"trace.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 with gzip.open(R/"decisions"/f"trial_{tid:03d}.jsonl.gz","rt",encoding="utf-8") as f:outs=[json.loads(x) for x in f]
 assert len(rows)==len(outs)==200 and all(x["trial_id"]==tid for x in rows)
 with gzip.open(trial/"fcd.xml.gz","rt",encoding="utf-8") as f:root=ET.parse(f).getroot()
 states=[]
 for t in root:
  for v in t:
   if v.attrib.get("id")=="ego":states.append((1000000+round(float(t.attrib["time"])*1000000)+10000,float(v.attrib["speed"])*1000))
 assert states[0][0]==1010000
 first=rows[0];values=[x for x in rows if x["decision"]["research_speed_mmps"] is not None]
 recorded=dict(trial_id=tid,case=first["case"],age_us=first["age_us"],seed=first["seed"],trajectory_policy=first["policy"],estimator="recorded_"+first["policy"],
 queries=200,selected=len(values),availability=len(values)/200,
 selected_error_gt05=sum(abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>500 for x in values),
 selected_error_gt2=sum(abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>2000 for x in values),
 active_queries=sum(x["active"] for x in rows),active_selected=sum(x["active"] for x in values),
 active_error_gt05=sum(x["active"] and abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>500 for x in values),
 active_error_gt2=sum(x["active"] and abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>2000 for x in values))
 expected.append(recorded)
 starts=[x["now_us"] for x in rows];commands=[x["requested_mmps2"] for x in rows]
 truth_map=dict(states)
 for row in rows:
  assert abs(truth_map[row["now_us"]]-row["ego_speed_mps"]*1000)<=.002
  truth_queries+=1
 for mode in ("s3e_global","s3e_command"):
  violations=[]
  for (s,v),(e,w) in zip(states,states[1:]):
   j=bisect.bisect_right(starts,s)-1;u=0 if j<0 else commands[j]
   low,high=(-4000,2000) if mode=="s3e_global" else (max(-4000,u-1000),min(2000,u+1000))
   dt=(e-s)/1000000
   if w<max(0,v+low*dt)-.002 or w>max(0,v+high*dt)+.002:violations.append(e)
   transitions+=1
  state=[(F(9750),F(10250))];last=1010000;previous_command=0
  premise=9750<=states[0][1]<=10250;counts={k:0 for k in ("selected","selected_error_gt05","selected_error_gt2","empty_state_queries",
    "set_excludes_reference","conditional_set_excludes_reference","conditional_selected_error_gt05","premise_valid_queries",
    "active_selected","active_error_gt05","active_error_gt2")}
  for row,out in zip(rows,outs):
   obs=sensor_bounds(row);now=row["now_us"];truth=row["ego_speed_mps"]*1000
   state=oracle(state,now-last,previous_command,obs,mode)
   value=None
   if state and row["risk"]!="lockdown":
    center=round((state[0][0]+state[-1][1])/2)
    if all(abs(center-x)<=500 for pair in state for x in pair):value=center
   expected_row=dict(now_us=now,intervals=[[str(a),str(b)] for a,b in state],selected_mmps=value,empty_state=not state,
    reason="empty_latched" if not state else "lockdown" if row["risk"]=="lockdown" else "budget_exhausted" if value is None else "whole_set_budget")
   assert out[mode]==expected_row,(tid,row["step"],mode,"decision mismatch")
   premise=premise and any(x is None or float(x[0])-.002<=truth<=float(x[1])+.002 for x in obs) and not (violations and violations[0]<=now)
   counts["premise_valid_queries"]+=bool(premise)
   inside=any(float(a)-.002<=truth<=float(b)+.002 for a,b in state)
   counts["set_excludes_reference"]+=not inside;counts["conditional_set_excludes_reference"]+=bool(premise and not inside)
   counts["empty_state_queries"]+=not state
   if value is not None:
    error=abs(value-truth)
    counts["selected"]+=1;counts["selected_error_gt05"]+=error>500;counts["selected_error_gt2"]+=error>2000
    counts["conditional_selected_error_gt05"]+=bool(premise and error>500)
    if row["active"]:
     counts["active_selected"]+=1;counts["active_error_gt05"]+=error>500;counts["active_error_gt2"]+=error>2000
   last=now;previous_command=row["requested_mmps2"];count+=1
  summary=dict(trial_id=tid,case=first["case"],age_us=first["age_us"],seed=first["seed"],trajectory_policy=first["policy"],
   estimator=mode,queries=200,**counts,availability=counts["selected"]/200,active_queries=sum(x["active"] for x in rows))
  expected.append(summary)
assert expected==result["summaries"],"summary mismatch"
aggregate=[]
for key,group in itertools.groupby(sorted(expected,key=lambda x:(x["trajectory_policy"],x["case"],x["age_us"],x["estimator"])),key=lambda x:(x["trajectory_policy"],x["case"],x["age_us"],x["estimator"])):
 group=list(group);a=dict(zip(("trajectory_policy","case","age_us","estimator"),key));a["seeds"]=len(group)
 for name in ("queries","selected","selected_error_gt05","selected_error_gt2","active_queries","active_selected","active_error_gt05","active_error_gt2"):
  a[name]=sum(x[name] for x in group)
 a["availability"]=a["selected"]/a["queries"]
 for name in ("empty_state_queries","set_excludes_reference","conditional_set_excludes_reference","conditional_selected_error_gt05","premise_valid_queries"):
  a[name]=None if name not in group[0] else sum(x[name] for x in group)
 aggregate.append(a)
assert aggregate==json.loads((R/"aggregate_results.json").read_text())
assert count==57600 and truth_queries==28800
assert all(x["conditional_set_excludes_reference"]==x["conditional_selected_error_gt05"]==0 for x in expected if x["estimator"].startswith("s3e_"))
assert all(sha(P/n)==h for n,h in lock.items())
dump(R/"verification.json",dict(passed=True,estimator_decisions_reconstructed=count,truth_queries_matched=truth_queries,
 transition_appraisals=transitions,summaries_reconstructed=len(expected),aggregate_cells_reconstructed=len(aggregate),
 independent_algorithm="endpoint-cell decomposition; no estimator imports",physical_vehicle_validation=False,new_algorithm_principle_established=False))
print("INDEPENDENT_SECURE_SET_REPLAY_PASS",count,len(expected),len(aggregate),flush=True)
