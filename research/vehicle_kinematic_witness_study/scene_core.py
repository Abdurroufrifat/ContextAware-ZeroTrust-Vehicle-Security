"""Authenticated scene-pair witness. Sensor truth and motion bounds are declared premises."""
from fractions import Fraction as F
import json,hmac,hashlib
FIELDS=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","gap_um","lead_mmps")
DOMAIN=b"COACTCPS-SCENE-HMAC-SHA256-v1\x00"
def canonical(body):
 return DOMAIN+json.dumps([body[k] for k in FIELDS],ensure_ascii=True,separators=(",",":")).encode("ascii")
def sign_scene(body,key):
 return dict(body=dict(body),tag=hmac.new(key,canonical(body),hashlib.sha256).hexdigest())
def witness_bounds(previous_gap_um,current_gap_um,current_lead_mmps,dt_us):
 if type(dt_us) is not int or not 0<dt_us<=100000:raise ValueError("invalid witness window")
 previous,current,lead=map(F,(previous_gap_um,current_gap_um,current_lead_mmps))
 if min(previous,current,lead)<0:raise ValueError("invalid scene")
 center=lead-(current-previous)*F(1000,dt_us)
 radius=F(3000000,dt_us)+75+F(6000*dt_us,2000000)
 return max(F(0),center-radius),center+radius
def unique(pairs):
 out={}
 for k,v in pairs:
  if k in out:raise ValueError("duplicate key")
  out[k]=v
 return out
class SceneGuard:
 def __init__(self,key,context):
  if type(key) is not bytes or len(key)<16:raise ValueError("invalid key")
  if set(context)!={"vehicle_id","session_id","capture_id"} or any(type(x) is not str or not 0<len(x)<=128 for x in context.values()):raise ValueError("invalid context")
  self.key=key;self.context=dict(context);self.seq=-1;self.previous=None
 def verify(self,packet,now):
  if packet is None:return None,"missing"
  try:
   if type(packet) is not dict or set(packet)!={"wire","received_us"} or type(packet["wire"]) is not str or len(packet["wire"])>8192:raise ValueError()
   obj=json.loads(packet["wire"],object_pairs_hook=unique)
   if type(obj) is not dict or set(obj)!={"body","tag"} or type(obj["body"]) is not dict or set(obj["body"])!=set(FIELDS):raise ValueError()
   b=obj["body"]
   if any(type(b[k]) is not int or not 0<=b[k]<2**63 for k in ("schema","seq","measured_us","gap_um","lead_mmps")) or b["schema"]!=1:raise ValueError()
   if (b["source_id"],b["key_id"],b["channel"])!=("scene","s-key","lead_range"):raise ValueError()
   if any(b[k]!=v for k,v in self.context.items()):return None,"context_mismatch"
   if type(obj["tag"]) is not str or not hmac.compare_digest(obj["tag"],hmac.new(self.key,canonical(b),hashlib.sha256).hexdigest()):return None,"bad_tag"
   if type(now) is not int or type(packet["received_us"]) is not int or not b["measured_us"]==packet["received_us"]==now:return None,"time_mismatch"
   if b["seq"]<=self.seq:return None,"replayed"
   self.seq=b["seq"];return b,"authenticated"
  except (ValueError,TypeError,KeyError,OverflowError):return None,"malformed"
 def evaluate(self,packet,now,base_decision):
  base=dict(base_decision);body,status=self.verify(packet,now)
  previous=self.previous;bounds=None;why="scene_"+status
  if body is not None:
   self.previous=body
   if previous is None:why="scene_history_unavailable"
   elif not 0<body["measured_us"]-previous["measured_us"]<=100000:why="scene_history_outside_window"
   else:
    bounds=witness_bounds(previous["gap_um"],body["gap_um"],body["lead_mmps"],body["measured_us"]-previous["measured_us"])
    if bounds[0]>bounds[1]:bounds=None;why="scene_inconsistent_interval"
    else:why="scene_interval_available"
  out={**base,"base_decision":base,"scene_status":status,"scene_budget_satisfied":False,
   "scene_contract_assumed":True,"scene_previous_seq":previous["seq"] if previous is not None else None,
   "scene_previous_measured_us":previous["measured_us"] if previous is not None else None,
   "witness_low_n":bounds[0].numerator if bounds is not None else None,
   "witness_low_d":bounds[0].denominator if bounds is not None else None,
   "witness_high_n":bounds[1].numerator if bounds is not None else None,
   "witness_high_d":bounds[1].denominator if bounds is not None else None,
   "physical_validation_established":False,"promotion_allowed":False}
  if base["research_speed_mmps"] is None:return out
  if bounds is not None:
   lo,hi=bounds;original=base["research_speed_mmps"]
   if max(abs(original-lo),abs(original-hi))<=500:
    out.update(reason="scene_certified_original",scene_budget_satisfied=True);return out
   point=round((lo+hi)/2)
   if max(abs(point-lo),abs(point-hi))<=500:
    out.update(selection="scene_witness",reason="scene_witness_replacement",research_speed_mmps=point,
     research_speed_mps=point/1000,conditional_budget_satisfied=False,scene_budget_satisfied=True);return out
   why="scene_budget_exhausted"
  out.update(selection="abstain",reason=why,research_speed_mmps=None,research_speed_mps=None,conditional_budget_satisfied=False)
  return out
