"""Signed-input application for the frozen, conditionally shielded model."""
from learned_common import R,checked,sha
import sys,json
sys.path.insert(0,str(R.parent/"project_release"))
from research_session import ResearchSession
from learned_core import apply_learned
import joblib
class LearnedResearchSession:
 def __init__(self,state_directory,context,keys):
  lock=json.loads(checked(R/"model_lock.json").read_text())
  for name in ("model_selection.json","learned_core.py"):
   asset=checked(R/name);relative=str(asset.relative_to(R.parent.parent))
   if lock.get(relative)!=sha(asset):raise ValueError("Frozen selection/source integrity check failed")
  selection=json.loads(checked(R/"model_selection.json").read_text())
  self.model_name=selection["selected"]
  if self.model_name not in ("ridge","random_forest","mlp_7","mlp_19","mlp_31"):raise ValueError("Unknown frozen model")
  path=checked(R/"models"/(self.model_name+".joblib"))
  relative=str(path.relative_to(R.parent.parent));self.model_sha256=sha(path)
  if lock.get(relative)!=self.model_sha256:raise ValueError("Frozen model integrity check failed")
  self.model=joblib.load(path)
  self.session=ResearchSession(state_directory,context,keys,mode="kinematic")
 def process(self,record):
  scene=self.session.process(record)
  out=apply_learned(scene,record["packets"],record["scene_packet"],record["now_us"],self.model,True)
  if out["physical_speed_authorized"] is not False or out["promotion_allowed"] is not False:
   raise RuntimeError("Research authorization invariant violated")
  return out
