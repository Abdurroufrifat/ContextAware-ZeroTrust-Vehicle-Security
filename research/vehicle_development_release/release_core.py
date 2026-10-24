"""Exact numeric sidecar and contained artifact verification; no physical appraisal."""
from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,json,re
import numpy as np
_REFERENCE=None
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def artifact_path(root,relative):
    if type(relative) is not str or not relative or "\x00" in relative:raise ValueError("invalid artifact path")
    raw=relative.replace("\\","/");parts=raw.split("/");low=raw.lower()
    if raw.startswith("/") or re.match(r"^[a-zA-Z]:",raw) or ".." in parts:raise ValueError("absolute or traversing artifact")
    if "library03" in low or "rav4" in low or "vehicle_final_evaluation" in low or ("urban02" in low and "night" in low):raise ValueError("reserved artifact")
    base=Path(root).resolve();target=(base/relative).resolve()
    if not target.is_relative_to(base):raise ValueError("artifact escapes root")
    resolved_low=str(target).replace(chr(92),"/").lower()
    if "library03" in resolved_low or "rav4" in resolved_low or "vehicle_final_evaluation" in resolved_low or ("urban02" in resolved_low and "night" in resolved_low):raise ValueError("resolved reserved artifact")
    return target
def verify_manifest(root,manifest_name):
    manifest=json.loads(artifact_path(root,manifest_name).read_text(encoding="utf-8"))
    if not isinstance(manifest,dict):raise ValueError("unsupported artifact manifest")
    result=[]
    for name,digest in manifest.items():
        if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}",digest):raise ValueError("invalid artifact digest")
        path=artifact_path(root,name);actual=file_sha(path)
        if actual!=digest:raise ValueError("artifact digest mismatch: "+name)
        result.append({"path":str(path),"sha256":actual})
    return result
def exact_library():
    global _REFERENCE
    if _REFERENCE is None:
        root=Path(__file__).resolve().parent.parent/"vehicle_reference_uncertainty"
        manifest=json.loads(artifact_path(root,"sha256.json").read_text(encoding="utf-8"));path=artifact_path(root,"reference_bounds.py")
        if manifest.get(path.name)!=file_sha(path):raise ValueError("established exact-reference source changed")
        spec=importlib.util.spec_from_file_location("coact_pinned_exact_reference",path)
        _REFERENCE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_REFERENCE)
    return _REFERENCE
def exact_sensitivity(prediction,reference,epsilon,admitted=None):
    p=np.asarray(prediction,float);y=np.asarray(reference,float)
    mask=np.ones(p.shape,bool) if admitted is None else np.asarray(admitted,bool)
    if p.ndim!=1 or p.shape!=y.shape or mask.shape!=p.shape:raise ValueError("unmatched numeric rows")
    library=exact_library();u=library.number(epsilon) if epsilon is not None else None
    if u is not None and u<0:raise ValueError("negative uncertainty")
    present=mask&np.isfinite(p);scored=present&np.isfinite(y)
    diffs=[abs(library.number(float(a))-library.number(float(b))) for a,b in zip(p[scored],y[scored])]
    observed=library.percentile95(diffs);unknown=int(sum(present&~np.isfinite(y)))
    intervals=[library.error_interval(float(a),float(b),u) for a,b in zip(p[scored],y[scored])] if u is not None else []
    low=library.percentile95([a for a,b in intervals]);high=library.percentile95([b for a,b in intervals])
    within=bool(high is not None and high<=Fraction(1,2))
    budget=Fraction(1,2)-observed if observed is not None and observed<=Fraction(1,2) else None
    return {"attempts":len(p),"admitted":int(present.sum()),"scored":len(diffs),"unknown_reference":unknown,
      "epsilon_input_exact":str(u) if u is not None else None,
      "observed_proxy_p95_exact":str(observed) if observed is not None else None,
      "true_error_p95_lower_exact":str(low) if low is not None else None,
      "true_error_p95_upper_exact":str(high) if high is not None else None,
      "true_error_p95_lower_display":float(low) if low is not None else None,
      "true_error_p95_upper_display":float(high) if high is not None else None,
      "definitely_above2":sum(a>2 for a,b in intervals) if u is not None else None,
      "possibly_above2":sum(b>2 for a,b in intervals) if u is not None else None,
      "max_uniform_epsilon_for_scored_p95_exact":str(budget) if budget is not None else None,
      "scored_subset_upper_within_0p5":within,
      "conditional_full_admitted_set_upper_within_0p5":bool(within and unknown==0 and len(diffs)>0),
      "physical_reference_appraised":False,"promotion_allowed":False,
      "interpretation":"exact conditional arithmetic on stored numeric inputs; supplied bounds are assumptions"}

def validate_scenario_identity(row,historical,index):
    expected={"row":index,**{key:historical[key] for key in ["sequence","group","scope","epsilon_assumption_mps"]}}
    for key,value in expected.items():
        if key not in row or type(row[key]) is not type(value) or row[key]!=value:
            raise ValueError("historical scenario identity mismatch: "+key)
