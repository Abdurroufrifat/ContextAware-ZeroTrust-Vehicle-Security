"""Packet trust, observed context and a discrete numerical control filter."""
import hashlib,hmac,json,math
CONTEXT=("vehicle_id","session_id","capture_id")
RADAR={"schema",*CONTEXT,"seq","measured_us","gap_um","relative_mmps"}
ACTUATOR={"schema",*CONTEXT,"seq","issued_us","accel_mmps2","duration_us"}
def canonical(role,body):
    return ("COACTCPS-CLOSEDLOOP-"+role+"-v1\0").encode()+json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def seal(role,body,key,now):
    return {"wire":json.dumps({"body":body,"tag":hmac.new(key,canonical(role,body),hashlib.sha256).hexdigest()},separators=(",",":")),"received_us":now}
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError("duplicate JSON key")
        out[k]=v
    return out
class PacketGate:
    def __init__(self,role,key,context):
        if role not in ("radar","actuator") or type(key) is not bytes or len(key)<16:raise ValueError("key/domain")
        if set(context)!=set(CONTEXT) or any(type(v) is not str or not 0<len(v)<=128 for v in context.values()):raise ValueError("context")
        self.role=role;self.key=key;self.context=dict(context);self.seq=-1
    def verify(self,packet,now):
        if packet is None:return None,"missing"
        try:
            if type(now) is not int or now<0 or type(packet) is not dict or set(packet)!={"wire","received_us"}:raise ValueError()
            if type(packet["wire"]) is not str or len(packet["wire"])>8192:raise ValueError()
            obj=json.loads(packet["wire"],object_pairs_hook=unique)
            if type(obj) is not dict or set(obj)!={"body","tag"} or type(obj["body"]) is not dict:raise ValueError()
            b=obj["body"];fields=RADAR if self.role=="radar" else ACTUATOR
            if set(b)!=fields or any(type(b[k]) is not int for k in fields-set(CONTEXT)):raise ValueError()
            if b["schema"]!=1 or not 0<=b["seq"]<2**63:raise ValueError()
            if any(b[k]!=v for k,v in self.context.items()):return None,"context_mismatch"
            if type(obj["tag"]) is not str or not hmac.compare_digest(obj["tag"],hmac.new(self.key,canonical(self.role,b),hashlib.sha256).hexdigest()):return None,"bad_tag"
            stamp=b["measured_us" if self.role=="radar" else "issued_us"]
            if type(packet["received_us"]) is not int or stamp!=now or packet["received_us"]!=now:return None,"time_mismatch"
            if self.role=="radar":
                if not 0<=b["gap_um"]<10**12 or not -100000<=b["relative_mmps"]<=100000:raise ValueError()
            elif not -4000<=b["accel_mmps2"]<=2000 or b["duration_us"]!=100000:raise ValueError()
            if b["seq"]<=self.seq:return None,"replayed"
            self.seq=b["seq"];return b,"authenticated"
        except (KeyError,ValueError,TypeError,OverflowError):return None,"malformed"
def stop_distance(v,b):
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (v,b)) or v<0 or b<=0:raise ValueError("stopping input")
    n=max(0,math.ceil(v/(b*.01))-1)
    return .01*(n*v-b*.01*n*(n+1)/2)
def projected_margin(command,ego_high,lead_low,gap_low):
    if type(command) is not int or not -4000<=command<=2000:raise ValueError("command")
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (ego_high,lead_low,gap_low)) or min(ego_high,lead_low)<0:raise ValueError("state")
    ve=ego_high;vl=lead_low;g=gap_low;minimum=g-2.
    upper=min(2.,max(-4.,command/1000+1.))
    for _ in range(10):
        ve=max(0.,ve+upper*.01);vl=max(0.,vl-4.*.01)
        g+=(vl-ve)*.01
        minimum=min(minimum,g-2.)
    # With lead braking 4 >= ego braking 3, future gap minimum is at an endpoint.
    return min(minimum,g-2.+stop_distance(vl,4.)-stop_distance(ve,3.))
def safe_command(desired,ego_high,lead_low,gap_low):
    if type(desired) is not int or not -4000<=desired<=2000:raise ValueError("desired command")
    low=-4000;high=desired
    if projected_margin(low,ego_high,lead_low,gap_low)<0:return low,projected_margin(low,ego_high,lead_low,gap_low)
    while low<high:
        mid=(low+high+1)//2
        if projected_margin(mid,ego_high,lead_low,gap_low)>=0:low=mid
        else:high=mid-1
    return low,projected_margin(low,ego_high,lead_low,gap_low)
def idm(ego,lead,gap):
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (ego,lead,gap)) or min(ego,lead)<0:raise ValueError("controller input")
    desired=2+max(0.,ego*1.5+ego*(ego-lead)/(2*math.sqrt(8)))
    return max(-4000,min(2000,round(2000*(1-(ego/18.)**4-(desired/max(gap,.001))**2))))
def fuse_context(ego,scene,radar):
    if ego is None or scene is None or radar is None:return None
    elo=max(0.,ego-.5);ehi=ego+.5
    glo=max(scene["gap_um"]/1e6-.0015,radar["gap_um"]/1e6-.05)
    ghi=min(scene["gap_um"]/1e6+.0015,radar["gap_um"]/1e6+.05)
    llo=max(0.,scene["lead_mmps"]/1000-.075,radar["relative_mmps"]/1000-.1+elo)
    lhi=min(scene["lead_mmps"]/1000+.075,radar["relative_mmps"]/1000+.1+ehi)
    if glo>ghi or llo>lhi:return None
    return {"ego_low":elo,"ego_high":ehi,"gap_low":glo,"gap_high":ghi,
            "lead_low":llo,"lead_high":lhi,"gap_m":(glo+ghi)/2,"lead_mps":(llo+lhi)/2}
def advisory_action(mode,text):
    return 2000 if mode=="raw" and type(text) is str and "accelerate" in text.casefold() else None
