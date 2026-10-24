"""Learned residual point selection inside a conditional authenticated scene interval.
No state truth, attack labels, reference state or physical-control interface.
"""
from fractions import Fraction as F
import json,math
FEATURE_NAMES=("scene_midpoint_mps","candidate_minus_midpoint_mps","anchor_minus_midpoint_mps",
 "anchor_age_100ms","last_requested_fraction","lead_mps","gap_m","scene_radius_mps","candidate_present","anchor_present")
def project_point(value,lo,hi):
 if type(value) not in (int,float) or not math.isfinite(value):return None
 lo,hi=F(lo),F(hi)
 if lo<0 or lo>hi:return None
 lower=max(0,math.ceil(hi-500));upper=math.floor(lo+500)
 if lower>upper:return None
 return max(lower,min(upper,round(value)))
def features(payload,lo,hi):
 base=payload["base"];packets=payload["packets"];now=payload["now_us"]
 center=(F(lo)+F(hi))/2;radius=(F(hi)-F(lo))/2
 scene=json.loads(payload["scene_packet"]["wire"])["body"]
 has_c=base.get("candidate_status")=="authenticated" and packets.get("candidate") is not None
 has_a=base.get("anchor_status")=="authenticated" and packets.get("anchor") is not None
 candidate=json.loads(packets["candidate"]["wire"])["body"] if has_c else None
 anchor=json.loads(packets["anchor"]["wire"])["body"] if has_a else None
 command=None
 if base.get("command_status")=="authenticated" and packets.get("command") is not None:
  commands=json.loads(packets["command"]["wire"])["body"]["commands"]
  if commands:command=F(commands[-1][2],commands[-1][3])
 values=[center/1000,(F(candidate["speed_mmps"])-center)/1000 if has_c else F(0),
  (F(anchor["speed_mmps"])-center)/1000 if has_a else F(0),
  F(now-anchor["measured_us"],100000) if has_a else F(0),
  command/4000 if command is not None else F(0),F(scene["lead_mmps"],1000),
  F(scene["gap_um"],1000000),radius/1000,int(has_c),int(has_a)]
 result=[float(x) for x in values]
 if not all(math.isfinite(x) for x in result):raise ValueError("nonfinite features")
 return result
def apply_learned(scene_decision,packets,scene_packet,now,model,shielded=True):
 out={**scene_decision,"scene_decision":dict(scene_decision),"learned_status":"not_evaluated",
 "learned_prediction_mmps":None,"learned_feature_vector":None,"physical_speed_authorized":False,"promotion_allowed":False}
 if scene_decision["research_speed_mmps"] is None:return out
 try:
  if scene_decision.get("scene_status")!="authenticated" or not scene_decision.get("scene_budget_satisfied"):return out
  lo=F(scene_decision["witness_low_n"],scene_decision["witness_low_d"])
  hi=F(scene_decision["witness_high_n"],scene_decision["witness_high_d"])
  x=features(dict(base=scene_decision["base_decision"],packets=packets,scene_packet=scene_packet,now_us=now),lo,hi)
  residual=float(model.predict([x])[0]);prediction=float((lo+hi)/2)+residual*1000
  if not math.isfinite(prediction):raise ValueError("nonfinite prediction")
  point=project_point(prediction,lo,hi) if shielded else max(0,round(prediction))
  if point is None:raise ValueError("no feasible point")
  out.update(selection="learned_shielded" if shielded else "learned_unshielded",reason="learned_scene_projection" if shielded else "unshielded_learning_ablation",
   research_speed_mmps=point,research_speed_mps=point/1000,conditional_budget_satisfied=False,scene_budget_satisfied=bool(shielded),
   learned_status="projected" if shielded else "unshielded",learned_prediction_mmps=prediction,learned_feature_vector=x)
 except Exception as exc:
  out["learned_status"]="invalid_prediction_fallback";out["learned_failure_type"]=type(exc).__name__
 return out
