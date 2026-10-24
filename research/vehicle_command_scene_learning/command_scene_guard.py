from cs_common import *
from fractions import Fraction as F
from scene_core import SceneGuard
from learned_core import project_point
from command_scene_core import refine,relative_features
import math
class CommandSceneGuard:
 def __init__(self,key,context,mode="refined_midpoint",model=None):
  if mode not in ("refined_midpoint","refined_learned"):raise ValueError("unsupported mode")
  if mode=="refined_learned" and model is None:raise ValueError("model required")
  self.scene=SceneGuard(key,context);self.mode=mode;self.model=model
 def evaluate(self,packet,now,base,packets):
  previous=self.scene.previous;scene=self.scene.evaluate(packet,now,base)
  out={**scene,"scene_decision":scene,"refinement_status":"not_evaluated","refinement_reason":None,
   "refined_low_n":None,"refined_low_d":None,"refined_high_n":None,"refined_high_d":None,
   "relative_features":None,"residual_prediction_mps":None,"refined_budget_satisfied":False,"refined_assumptions":"scene_truth_and_command_execution","refined_model":"right_endpoint_10ms"}
  if scene["research_speed_mmps"] is None or not scene.get("scene_budget_satisfied") or previous is None:return out
  if base.get("command_status")!="authenticated":return out
  body=json.loads(packet["wire"])["body"];history=json.loads(packets["command"]["wire"])["body"]["commands"]
  broad=(F(scene["witness_low_n"],scene["witness_low_d"]),F(scene["witness_high_n"],scene["witness_high_d"]))
  refined=refine(previous["gap_um"],body["gap_um"],previous["lead_mmps"],body["lead_mmps"],previous["measured_us"],now,history,broad)
  out.update(refinement_status=refined["status"],refinement_reason=refined["reason"])
  if refined["status"]=="conflict":
   out.update(selection="abstain",reason=refined["reason"],research_speed_mmps=None,research_speed_mps=None,scene_budget_satisfied=False,conditional_budget_satisfied=False);return out
  if refined["status"]!="available":return out
  lo,hi=refined["low"],refined["high"]
  out.update(refined_low_n=lo.numerator,refined_low_d=lo.denominator,refined_high_n=hi.numerator,refined_high_d=hi.denominator)
  prediction=float(refined["center"]);why="refined_scene_midpoint"
  if self.mode=="refined_learned":
   x=relative_features(base,packets,body,now,refined);out["relative_features"]=x
   try:
    value=float(self.model.predict([x])[0])
    if not math.isfinite(value):raise ValueError("nonfinite model output")
    prediction+=value*1000;out["residual_prediction_mps"]=value;why="refined_scene_learning"
   except Exception:why="refined_scene_midpoint_model_fallback"
  point=project_point(prediction,lo,hi)
  if point is None:
   out.update(selection="abstain",reason="refined_budget_exhausted",research_speed_mmps=None,research_speed_mps=None,scene_budget_satisfied=False,conditional_budget_satisfied=False);return out
  out.update(selection=self.mode,reason=why,research_speed_mmps=point,research_speed_mps=point/1000,
   scene_budget_satisfied=False,refined_budget_satisfied=True,conditional_budget_satisfied=False)
  return out
