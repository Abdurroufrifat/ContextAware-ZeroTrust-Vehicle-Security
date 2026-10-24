"""Research context model and durable authenticated-envelope integration."""
import hashlib,hmac,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"vehicle_evidence_auth"))
import auth_guard as auth
FIELDS={"peer","yaw","steering","acceleration"}
MARGINS=np.array([.5,.05,.1,1.])
DOMAIN=b"COACT-CONTEXT-BINDING-v1\x00"
def raw_context(features):
    if type(features) is not dict or set(features)!=FIELDS:
        raise ValueError("Exactly the declared online context fields are required")
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in features.values()):
        raise ValueError("Finite numeric context required")
    return np.array([features["peer"],abs(features["yaw"]),abs(features["steering"]),abs(features["acceleration"])])
def feature_vector(features):
    p,y,s,a=raw_context(features)
    values=np.array([1.,p,y,s,a,p*s*s,p*y,p*p])
    if not np.isfinite(values).all(): raise ValueError("Context basis overflow")
    return values
def fit_model(train,calibration):
    if len(train)<10 or len(calibration)<5: raise ValueError("Insufficient training/calibration data")
    x=np.array([feature_vector(r["features"]) for r in train])
    y=np.array([r["candidate"]-r["features"]["peer"] for r in train])
    mean=x.mean(axis=0);mean[0]=0.
    scale=x.std(axis=0);scale[scale<1e-9]=1.;scale[0]=1.
    z=(x-mean)/scale
    penalty=np.eye(z.shape[1]);penalty[0,0]=0.
    coef=np.linalg.solve(z.T@z+penalty,z.T@y)
    raw=np.array([raw_context(r["features"]) for r in train])
    model=dict(coef=coef.tolist(),mean=mean.tolist(),scale=scale.tolist(),
        raw_min=raw.min(axis=0).tolist(),raw_max=raw.max(axis=0).tolist(),
        global_center=float(np.median(y)),train_count=len(train),calibration_count=len(calibration))
    residuals=[abs(r["candidate"]-r["features"]["peer"]-predict(model,r["features"])) for r in calibration]
    global_residuals=[abs(r["candidate"]-r["features"]["peer"]-model["global_center"]) for r in calibration]
    model["threshold"]=max(.05,float(np.quantile(residuals,.995,method="higher")))
    model["global_threshold"]=max(.05,float(np.quantile(global_residuals,.995,method="higher")))
    model["monitor_mean"]=float(y.mean())
    model["monitor_sd"]=max(.02,float(y.std(ddof=1)))
    if not np.isfinite(coef).all(): raise ValueError("Nonfinite fitted coefficients")
    return model
def predict(model,features):
    value=float(((feature_vector(features)-np.array(model["mean"]))/np.array(model["scale"]))@np.array(model["coef"]))
    if not math.isfinite(value): raise ValueError("Nonfinite model prediction")
    return value
def out_of_support(model,features):
    raw=raw_context(features)
    mask=(raw<np.array(model["raw_min"])-MARGINS)|(raw>np.array(model["raw_max"])+MARGINS)
    return [name for name,bad in zip(("peer","abs_yaw","abs_steering","abs_acceleration"),mask) if bad]
def assess(model,candidate,features,use_ood=True):
    result=dict(accept=False,reason="invalid_input",prediction=None,residual=None,threshold=model["threshold"],ood=[])
    if type(candidate) not in (int,float) or not math.isfinite(candidate) or candidate<0: return result
    try:
        expected=predict(model,features)
        ood=out_of_support(model,features)
        residual=candidate-features["peer"]-expected
    except (ValueError,ArithmeticError): return result
    result.update(prediction=expected,residual=residual,ood=ood)
    if use_ood and ood: result["reason"]="context_out_of_support"
    elif abs(residual)>model["threshold"]: result["reason"]="context_deviation"
    else: result.update(accept=True,reason="context_consistent")
    return result
def context_bytes(body,features):
    raw_context(features)
    if abs(features["peer"]-body["speed_mmps"]/1000)>1e-12:
        raise ValueError("Context peer value differs from signed peer report")
    return DOMAIN+auth.canonical(body)+json.dumps(features,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def bind_context(body,features,key):
    report=auth.sign(body,key)
    return dict(report=report,features=dict(features),
        context_tag=hmac.new(key,context_bytes(body,features),hashlib.sha256).hexdigest())
def parse_context(wire):
    if type(wire) not in (str,bytes) or len(wire)>8192: raise ValueError("Invalid context packet")
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError("Duplicate key")
            out[k]=v
        return out
    obj=json.loads(wire,object_pairs_hook=unique)
    if type(obj) is not dict or set(obj)!={"report","features","context_tag"}: raise ValueError("Invalid context envelope")
    auth.validate_envelope(obj["report"])
    if type(obj["context_tag"]) is not str or auth.TAG.fullmatch(obj["context_tag"]) is None:
        raise ValueError("Malformed context tag")
    context_bytes(obj["report"]["body"],obj["features"])
    return obj
def authenticate(guard,candidate_wire,context_wire,now_us):
    result=dict(candidate_ok=False,context_ok=False,candidate=None,features=None,
        candidate_status="missing",context_status="missing")
    if candidate_wire is not None:
        try:
            packet=auth.parse_wire(candidate_wire)
            b=packet["body"]
            if b["source_id"]!="candidate" or b["channel"]!="speed": raise ValueError("Wrong role")
            check=guard.verify(packet,now_us,now_us)
            result["candidate_status"]=check["status"]
            if check["authenticated_report"]:
                result.update(candidate_ok=True,candidate=b["speed_mmps"]/1000)
        except (ValueError,TypeError): result["candidate_status"]="malformed_or_unavailable_state"
    if context_wire is not None:
        try:
            packet=parse_context(context_wire)
            body=packet["report"]["body"]
            if body["source_id"]!="peer" or body["channel"]!="motion_context": raise ValueError("Wrong context role")
            rule=guard.rules.get((body["source_id"],body["key_id"]))
            if rule is None: raise ValueError("Unknown context key")
            expected=hmac.new(rule.key,context_bytes(body,packet["features"]),hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected,packet["context_tag"]):
                result["context_status"]="bad_context_binding"
                return result
            check=guard.verify(packet["report"],now_us,now_us)
            result["context_status"]=check["status"]
            if check["authenticated_report"]:
                result.update(context_ok=True,features=packet["features"])
        except (ValueError,TypeError,UnicodeError): result["context_status"]="malformed_or_unavailable_state"
    return result
