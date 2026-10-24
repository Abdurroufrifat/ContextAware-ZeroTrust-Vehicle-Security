"""Domain-separated offline command-log authentication; no physical command attestation."""
from dataclasses import dataclass,field
from fractions import Fraction as F
from pathlib import Path
import hashlib,hmac,json,sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"vehicle_policy_gateway"))
import policy_gateway as p
a=p.auth
FIELDS=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","issued_us","commands")
DOMAIN=b"COACTCPS-COMMAND-HMAC-SHA256-v1\x00"
@dataclass(frozen=True)
class CommandRule:
 source_id:str
 key_id:str
 key:bytes=field(repr=False)
 domain_id:str
 enabled:bool=True

def canonical(body):
 if type(body) is not dict or set(body)!=set(FIELDS):raise ValueError("Unexpected command body")
 if type(body["schema"]) is not int or body["schema"]!=1:raise ValueError("Invalid schema")
 if not all(a.identifier(body[k]) for k in ("source_id","key_id","vehicle_id","session_id","capture_id","channel")):raise ValueError("Invalid command identifier")
 if not all(a.unsigned(body[k]) for k in ("seq","issued_us")):raise ValueError("Invalid sequence/time")
 records=body["commands"]
 if type(records) is not list or len(records)>16:raise ValueError("Invalid history")
 previous=None
 for row in records:
  if type(row) is not list or len(row)!=4 or not all(type(v) is int for v in row):raise ValueError("Invalid history record")
  start,end,num,den=row
  if not a.unsigned(start) or not a.unsigned(end) or not start<end<=body["issued_us"] or not 1<=den<=1000000000 or not -4000*den<=num<=2000*den:raise ValueError("Invalid history interval/input")
  if previous is not None and start!=previous:raise ValueError("History gap/overlap")
  previous=end
 return DOMAIN+json.dumps([body[k] for k in FIELDS],ensure_ascii=True,separators=(",",":")).encode("ascii")
def sign(body,key):
 a.key_check(key)
 return {"body":json.loads(json.dumps(body)),"tag":hmac.new(key,canonical(body),hashlib.sha256).hexdigest()}
def parse(raw):
 if type(raw) not in (str,bytes) or len(raw.encode("utf-8") if type(raw) is str else raw)>8192:raise ValueError("Invalid command wire")
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError("Duplicate JSON key")
   out[k]=v
  return out
 try:
  obj=json.loads(raw,object_pairs_hook=unique)
  if type(obj) is not dict or set(obj)!={"body","tag"} or type(obj["tag"]) is not str or a.TAG.fullmatch(obj["tag"]) is None:raise ValueError("Invalid command envelope")
  canonical(obj["body"]);return obj
 except (ValueError,TypeError,UnicodeError,RecursionError) as e:raise ValueError("Malformed command envelope") from e

class CommandGuard:
 def __init__(self,report_guard,rule):
  if not isinstance(report_guard,a.Guard) or not isinstance(rule,CommandRule):raise ValueError("Provision existing report guard and command rule")
  if not all(a.identifier(v) for v in (rule.source_id,rule.key_id,rule.domain_id)) or type(rule.enabled) is not bool:raise ValueError("Invalid command rule")
  a.key_check(rule.key)
  if any(r.source_id==rule.source_id or r.key==rule.key for r in report_guard.rules.values()):raise ValueError("Command source and key must differ from report sources")
  self.report_guard=report_guard;self.rule=rule
 def verify(self,packet,now):
  if packet is None:return None,"missing"
  if not isinstance(packet,p.Packet):return None,"malformed"
  try:message=parse(packet.wire)
  except (ValueError,TypeError):return None,"malformed"
  body=message["body"];rule=self.rule
  if (body["source_id"],body["key_id"])!=(rule.source_id,rule.key_id):return None,"unknown_key"
  if not rule.enabled:return None,"revoked_key"
  expected=hmac.new(rule.key,canonical(body),hashlib.sha256).hexdigest()
  if not hmac.compare_digest(expected,message["tag"]):return None,"bad_tag"
  ctx=self.report_guard.context
  if (body["vehicle_id"],body["session_id"],body["capture_id"])!=(ctx.vehicle_id,ctx.session_id,ctx.capture_id):return None,"context_mismatch"
  if body["channel"]!="command_history":return None,"channel_denied"
  issued=body["issued_us"];received=packet.received_us
  if not a.unsigned(now) or not a.unsigned(received) or not issued<=received<=now:return None,"invalid_clock_order"
  if now-received>100000:return None,"receipt_stale"
  if now-issued>100000:return None,"measurement_stale"
  con=a.connect(self.report_guard.path)
  try:
   con.execute("PRAGMA synchronous=FULL");con.execute("BEGIN IMMEDIATE")
   last=a.read_context(con,ctx)
   if now<last:con.rollback();return None,"clock_rollback"
   saved=con.execute("SELECT seq FROM counters WHERE source_id=? AND channel=?",(rule.source_id,"command_history")).fetchone()
   if saved is not None and body["seq"]<=saved[0]:con.rollback();return None,"replay_or_reorder"
   con.execute("INSERT INTO counters VALUES (?,?,?) ON CONFLICT(source_id,channel) DO UPDATE SET seq=excluded.seq",(rule.source_id,"command_history",body["seq"]))
   con.execute("UPDATE metadata SET value=? WHERE name='last_now_us'",(str(now),));con.commit()
   return body,"authenticated"
  except (sqlite3.Error,ValueError):
   con.rollback();raise ValueError("Command replay state unavailable")
  finally:con.close()
