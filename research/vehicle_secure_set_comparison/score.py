"""Reference and premise appraisal; never imported by estimator module."""
import gzip,bisect,xml.etree.ElementTree as ET
from fractions import Fraction as F
from ssc_replay import observations

def contains(intervals,x):
 return any(F(a)<=x<=F(b) for a,b in intervals)

def evolution_prefix(trial,rows):
 with gzip.open(trial/"fcd.xml.gz","rt",encoding="utf-8") as f:root=ET.parse(f).getroot()
 states=[(1000000+round(float(t.attrib["time"])*1e6)+10000,float(v.attrib["speed"])*1000) for t in root for v in t if v.attrib.get("id")=="ego"]
 assert states and states[0][0]==1010000
 initial=9750<=states[0][1]<=10250
 starts=[x["now_us"] for x in rows];commands=[x["requested_mmps2"] for x in rows]
 bad={"s3e_global":[],"s3e_command":[]}
 for (s,v),(e,w) in zip(states,states[1:]):
  j=bisect.bisect_right(starts,s)-1;u=0 if j<0 else commands[j];dt=(e-s)/1e6
  for mode in bad:
   low,high=(-4000,2000) if mode=="s3e_global" else (max(-4000,u-1000),min(2000,u+1000))
   if not max(0,v+low*dt)-.002<=w<=max(0,v+high*dt)+.002:bad[mode].append(e)
 return initial,bad

def summarize(rows,items,decisions,mode,initial,bad):
 selected=errors05=errors2=conditional_errors=empty=set_misses=conditional_misses=0
 valid_prefix=initial
 active_selected=active_errors05=active_errors2=0;valid_queries=0
 for row,item,out in zip(rows,items,decisions):
  truth=row["ego_speed_mps"]*1000
  obs=observations(item)
  sensor_ok=any(x is None or x[0]-.002<=truth<=x[1]+.002 for x in obs)
  valid_prefix=valid_prefix and sensor_ok and not (bad[mode] and bad[mode][0]<=item["now"])
  valid_queries+=bool(valid_prefix)
  inside=any(float(F(a))-.002<=truth<=float(F(b))+.002 for a,b in out["intervals"])
  set_misses+=not inside;conditional_misses+=bool(valid_prefix and not inside)
  empty+=out["empty_state"]
  value=out["selected_mmps"]
  if value is not None:
   error=abs(value-truth)/1000;selected+=1;errors05+=error>.5;errors2+=error>2
   conditional_errors+=bool(valid_prefix and error>.5)
   if row["active"]:
    active_selected+=1;active_errors05+=error>.5;active_errors2+=error>2
 first=rows[0]
 return dict(trial_id=first["trial_id"],case=first["case"],age_us=first["age_us"],seed=first["seed"],
 trajectory_policy=first["policy"],estimator=mode,queries=len(rows),selected=selected,availability=selected/len(rows),
 selected_error_gt05=errors05,selected_error_gt2=errors2,empty_state_queries=empty,set_excludes_reference=set_misses,
 conditional_set_excludes_reference=conditional_misses,conditional_selected_error_gt05=conditional_errors,
 premise_valid_queries=valid_queries,active_queries=sum(x["active"] for x in rows),active_selected=active_selected,
 active_error_gt05=active_errors05,active_error_gt2=active_errors2)

def recorded_summary(rows):
 first=rows[0];values=[r for r in rows if r["decision"]["research_speed_mmps"] is not None]
 return dict(trial_id=first["trial_id"],case=first["case"],age_us=first["age_us"],seed=first["seed"],
 trajectory_policy=first["policy"],estimator="recorded_"+first["policy"],queries=len(rows),selected=len(values),
 availability=len(values)/len(rows),selected_error_gt05=sum(x["selected_disagreement_mps"]>.5 for x in values),
 selected_error_gt2=sum(x["selected_disagreement_mps"]>2 for x in values),active_queries=sum(x["active"] for x in rows),
 active_selected=sum(x["active"] for x in values),active_error_gt05=sum(x["active"] and x["selected_disagreement_mps"]>.5 for x in values),
 active_error_gt2=sum(x["active"] and x["selected_disagreement_mps"]>2 for x in values))
