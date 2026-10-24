"""Unified offline research API. Fixed study assumptions; no physical operation interface."""
from release_common import R,checked
from pathlib import Path
import hashlib
import sumo_bridge as bridge
from temporal_guard import TemporalGuard
from scene_core import SceneGuard
ROLES=("candidate","anchor","command_logger","scene")
def study_keys():
 """Public deterministic test keys only; never describes protected physical keys."""
 return {n:hashlib.sha256(("COACT-SUMO-PUBLIC-TEST-"+n).encode()).digest() for n in ROLES}
def strip_record(row):
 def copy_packet(value):return None if value is None else dict(value)
 return dict(now_us=row["now_us"],risk=row["risk"],
  packets={role:copy_packet(row["packets"][role]) for role in ("candidate","anchor","command")},
  scene_packet=copy_packet(row.get("scene_packet")))
def validate_packet(value):
 if value is None:return None
 if type(value) is not dict or set(value)!={"wire","received_us"} or type(value["wire"]) is not str or len(value["wire"])>16384:
  raise ValueError("Expected a bounded signed wire packet")
 if type(value["received_us"]) is not int or not 0<=value["received_us"]<2**63:raise ValueError("Invalid receiver timestamp")
 return value
class ResearchSession:
 def __init__(self,state_directory,context,keys,*,mode="kinematic"):
  if mode not in ("kinematic","temporal"):raise ValueError("Unsupported research mode")
  if type(context) is not dict or set(context)!={"vehicle_id","session_id","capture_id"}:raise ValueError("Invalid context")
  if any(type(x) is not str or not 0<len(x)<=128 for x in context.values()):raise ValueError("Invalid context value")
  if type(keys) is not dict or set(keys)!=set(ROLES) or any(type(k) is not bytes or not 16<=len(k)<=64 for k in keys.values()):raise ValueError("Four explicit research signing keys are required")
  if len(set(keys.values()))!=4:raise ValueError("Source and scene key material must be distinct")
  state=checked(Path(state_directory))
  if not state.resolve().is_relative_to((R/"runtime").resolve()):raise PermissionError("Research session state must be inside release runtime")
  state.mkdir(parents=True,exist_ok=False)
  cg=bridge.cg;p=cg.p;ce=cg.ce
  ctx=p.auth.Context(context["vehicle_id"],context["session_id"],context["capture_id"])
  database=state/"replay.sqlite";p.auth.create_state(database,ctx)
  rules=[p.auth.KeyRule("candidate","c-key","wheel_speed",keys["candidate"]),p.auth.KeyRule("anchor","a-key","protected_speed",keys["anchor"])]
  profiles=[p.SourcePolicy("candidate","wheel_speed","candidate","drive",None,0,100000000),
   p.SourcePolicy("anchor","protected_speed","anchor","instrument",250,0,100000000,True)]
  self.gateway=cg.CommandGateway(p.auth.Guard(database,ctx,rules),profiles,ce.CommandRule("command_logger","h-key",keys["command_logger"],"control"),
   clock_uncertainty_us=5000,execution_error_mmps2=1000,research_execution_assumption=True)
  self.scene=SceneGuard(keys["scene"],context);self.temporal=TemporalGuard();self.mode=mode;self.last_time=-1;self.packet_type=p.Packet
 def process(self,record):
  if type(record) is not dict or set(record)!={"now_us","risk","packets","scene_packet"}:raise ValueError("Only signed reports, receiver time and risk are accepted")
  now=record["now_us"]
  if type(now) is not int or not 0<=now<2**63 or now<=self.last_time:raise ValueError("Query times must be valid and strictly increasing")
  if type(record["risk"]) is not str:raise ValueError("Invalid risk context")
  reports=record["packets"]
  if type(reports) is not dict or set(reports)!={"candidate","anchor","command"}:raise ValueError("Invalid signed-report roles")
  packets={}
  for role,value in reports.items():
   value=validate_packet(value)
   packets[role]=None if value is None else self.packet_type(value["wire"],value["received_us"])
  scene=validate_packet(record["scene_packet"]);self.last_time=now
  base=self.gateway.evaluate(packets["candidate"],packets["anchor"],packets["command"],now,risk=record["risk"])
  result=self.scene.evaluate(scene,now,base) if self.mode=="kinematic" else self.temporal.evaluate(base,now)
  if result.get("physical_speed_authorized") is not False or result.get("promotion_allowed") is not False:raise RuntimeError("Research authorization invariant violated")
  return result
