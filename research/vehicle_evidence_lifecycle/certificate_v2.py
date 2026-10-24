"""Signed sidecar plus independent full microstep safety check."""
import json,hmac,hashlib,math
DOMAIN=b"COACT-CPS-LIFECYCLE-CERTIFICATE-v1\0"
def canonical(b):return DOMAIN+json.dumps(b,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def wirehash(w):return hashlib.sha256(w.encode()).hexdigest()
def seal_certificate(wire,key,context,seq,now,command,bounds,error,steps,observation):
    b={"schema":1,"context":dict(context),"seq":seq,"issued_us":now,"expires_us":now+steps*10000,
       "command_mmps2":command,"bounds":dict(bounds),"execution_error":error,"hold_steps":steps,
       "observation_sha256":observation,"command_wire_sha256":wirehash(wire)}
    return {"body":b,"tag":hmac.new(key,canonical(b),hashlib.sha256).hexdigest()}
def independent_margin(command,ego,lead,gap,error,steps):
    if not 0<=error<4:return -math.inf
    v,w,g=ego,lead,gap;least=g-2;elapsed=0
    while elapsed<steps or v>0 or w>0:
        a=max(-4,min(2,command/1000+error)) if elapsed<steps else -4+error
        v=max(0,v+.01*a);w=max(0,w-.04);g+=.01*(w-v);least=min(least,g-2);elapsed+=1
        if elapsed>20000:raise ValueError("unbounded stopping horizon")
    return least
def verify_certificate(cert,wire,key,context,now,bounds,error,observation,seen):
    try:
        if type(cert) is not dict or set(cert)!={"body","tag"}:return False,"malformed"
        b=cert["body"]
        if set(b)!={"schema","context","seq","issued_us","expires_us","command_mmps2","bounds","execution_error","hold_steps","observation_sha256","command_wire_sha256"}:return False,"schema"
        if not hmac.compare_digest(cert["tag"],hmac.new(key,canonical(b),hashlib.sha256).hexdigest()):return False,"bad_tag"
        if b["context"]!=context:return False,"context_mismatch"
        if any(type(b[k]) is not int for k in ("schema","seq","issued_us","expires_us","command_mmps2","hold_steps")):return False,"types"
        if b["schema"]!=1 or b["seq"]<=seen:return False,"replayed"
        if not b["issued_us"]<=now<=b["expires_us"]:return False,"expired"
        if not 1<=b["hold_steps"]<=100 or b["expires_us"]!=b["issued_us"]+b["hold_steps"]*10000:return False,"lifetime"
        if b["bounds"]!=bounds or b["execution_error"]!=error or b["observation_sha256"]!=observation:return False,"evidence_binding"
        if b["command_wire_sha256"]!=wirehash(wire) or not -4000<=b["command_mmps2"]<=2000:return False,"command_binding"
        actual=json.loads(wire)["body"]["accel_mmps2"]
        if type(actual) is not int or actual!=b["command_mmps2"]:return False,"command_value_mismatch"
        actor=json.loads(wire)["body"]
        if now!=b["issued_us"]:return False,"delayed_delivery"
        if actor.get("issued_us")!=b["issued_us"] or actor.get("seq")!=b["seq"]:return False,"wire_time_sequence"
        if type(actor.get("duration_us")) is not int or actor["duration_us"]!=b["hold_steps"]*10000:return False,"duration_binding"

        if set(bounds)!={"ego_high","lead_low","gap_low"} or not all(type(v) in (float,int) and math.isfinite(v) for v in bounds.values()) or min(bounds["ego_high"],bounds["lead_low"])<0:return False,"bounds"
        m=independent_margin(b["command_mmps2"],bounds["ego_high"],bounds["lead_low"],bounds["gap_low"],error,b["hold_steps"])
        if m<0:return False,"unsafe"
        return True,"certificate_verified"
    except (KeyError,ValueError,TypeError,OverflowError):return False,"malformed"
