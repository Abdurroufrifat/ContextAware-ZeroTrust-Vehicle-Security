"""Strict signed-input application for the command-refined research estimator."""
from cs_common import R,P,checked,sha
import sys,json,joblib
sys.path.insert(0,str(R.parent/"project_release"))
from research_session import ResearchSession
from command_scene_guard import CommandSceneGuard
class CommandSceneSession:
 def __init__(self,state_directory,context,keys,*,mode="refined_learned"):
  if mode not in ("refined_midpoint","refined_learned"):raise ValueError("Unknown research mode")
  lock=json.loads(checked(R/"model_lock.json").read_text())
  for name in ("model_selection.json","command_scene_core.py","command_scene_guard.py"):
   path=checked(R/name)
   if lock.get(str(path.relative_to(P)))!=sha(path):raise ValueError("Frozen source/selection integrity check failed")
  selection=json.loads(checked(R/"model_selection.json").read_text());name=selection["selected"]
  if name not in ("ridge","forest","mlp_7","mlp_19","mlp_31"):raise ValueError("Unknown frozen model")
  model_path=checked(R/"models"/(name+".joblib"))
  if lock.get(str(model_path.relative_to(P)))!=sha(model_path):raise ValueError("Frozen model integrity check failed")
  model=joblib.load(model_path) if mode=="refined_learned" else None
  self.base=ResearchSession(state_directory,context,keys,mode="kinematic")
  self.refined=CommandSceneGuard(keys["scene"],context,mode,model)
 def process(self,record):
  scene=self.base.process(record)
  output=self.refined.evaluate(record["scene_packet"],record["now_us"],scene["base_decision"],record["packets"])
  if output["physical_speed_authorized"] is not False or output["promotion_allowed"] is not False:
   raise RuntimeError("Research authorization invariant violated")
  return output
