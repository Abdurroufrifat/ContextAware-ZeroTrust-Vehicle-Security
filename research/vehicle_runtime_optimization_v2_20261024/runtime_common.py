from pathlib import Path
import sys,json,hashlib,gzip,uuid,time,math
S=Path(__file__).resolve().parent;P=S.parent.parent;R=P/"research/vehicle_evidence_lifecycle";B=R/"confirmation_v2"
sys.path[:0]=[str(P/"src"),str(P/"scripts"),str(R)]
import lc_common
from lc_common import KEYS
from packet_policy_v2 import LifecyclePolicy as Original
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
def fresh_state(label):
 return P/"research/project_release/runtime"/("runtime-study-"+label+"-"+uuid.uuid4().hex)
def trace(t):
 with gzip.open(B/t["trace"],"rt") as f:return [json.loads(s) for s in f]
def same(a,b,path=""):
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+"/"+str(i))
 elif isinstance(a,dict) and isinstance(b,dict):
  assert set(a)==set(b),(path,set(a)^set(b))
  for k in a:same(a[k],b[k],path+"/"+k)
 elif isinstance(a,float) and isinstance(b,(float,int)):
  assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(path,a,b)
 else:assert a==b,(path,a,b)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def frame_evidence(out,packet,accepted,status,card):
 return canonical({"policy":{k:v for k,v in out.items() if k!="source_output"},"packet":packet,"accepted":accepted,"status":status,"card":card})
def run_rows(policy,rows,check=True,evidence=None):
 times=[];values=[]
 for x in rows:
  now=x["input"]["now_us"];start=time.perf_counter()
  out=policy.process(x["input"]);packet=policy.command_packet(out["requested_mmps2"],now,x["step"]+1)
  accepted,status=policy.actuate(x["delivered_command"],now)
  elapsed=time.perf_counter()-start;times.append(elapsed)
  if check:
   same(out,x["output"]);assert packet==x["issued_command"]
   assert (accepted,status)==(x["accepted_mmps2"],x["actuator_status"])
   same(policy.card(out,accepted,status),x["operator_card"])
  card=policy.card(out,accepted,status)
  actual=frame_evidence(out,packet,accepted,status,card)
  if check:assert actual==frame_evidence(x["output"],x["issued_command"],x["accepted_mmps2"],x["actuator_status"],x["operator_card"])
  if evidence is not None:evidence.append(hashlib.sha256(actual).hexdigest())
  values.append((out["estimate_mmps"],out["requested_mmps2"],accepted,status))
 return times,values
