"""Estimators consume only whitelisted inputs; truth is used by score.py separately."""
from ssc_core import predict,measurement,agree,output

def inputs(row):
 d=row["decision"];packets=row["packets"]
 def report(name):
  if d[name+"_status"]!="authenticated":return None
  item=packets[name]
  if item is None:raise ValueError("authenticated missing packet")
  return __import__("json").loads(item["wire"])["body"]
 c=report("candidate");a=report("anchor")
 return dict(now=row["now_us"],risk=row["risk"],
 candidate=None if c is None or row["risk"]!="normal" else c["speed_mmps"],
 anchor=None if a is None else (a["speed_mmps"],a["measured_us"]),
 next_requested=row["requested_mmps2"])

def observations(item):
 c=None if item["candidate"] is None else measurement(item["candidate"],0,0)
 a=None if item["anchor"] is None else measurement(item["anchor"][0],item["now"]-item["anchor"][1],5000)
 return [c,a]

def estimate_trial(items,mode):
 if mode not in ("s3e_global","s3e_command"):raise ValueError("mode")
 state=[(9750,10250)];previous=1010000;command=0
 for item in items:
  state=predict(state,item["now"]-previous,None if mode=="s3e_global" else command)
  state=agree(state,observations(item))
  value=output(state,lockdown=item["risk"]=="lockdown")
  yield dict(now_us=item["now"],intervals=[[str(a),str(b)] for a,b in state],selected_mmps=value,
   empty_state=not state,reason="empty_latched" if not state else "lockdown" if item["risk"]=="lockdown" else "budget_exhausted" if value is None else "whole_set_budget")
  previous=item["now"];command=item["next_requested"]
