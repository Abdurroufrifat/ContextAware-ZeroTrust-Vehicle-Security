"""Post-hoc reference diagnostics and conditional arithmetic, not physical certification."""
import re,numpy as np
def parse_status(token):
    match=re.fullmatch(r"status:\s*(\d+)",str(token).strip())
    if not match:raise ValueError("unrecognized INS status token")
    return int(match.group(1))
def latest_status(queries,times,status,max_age=100_000_000):
    q=np.asarray(queries,np.int64);t=np.asarray(times,np.int64);s=np.asarray(status,np.int64)
    if t.ndim!=1 or q.ndim!=1 or not len(t) or s.shape!=t.shape or np.any(np.diff(t)<=0):raise ValueError("invalid status series")
    index=np.searchsorted(t,q,side="right")-1;safe=np.maximum(index,0);age=q-t[safe]
    valid=(index>=0)&(age>=0)&(age<=max_age)
    return np.where(valid,s[safe],-1),np.where(index>=0,age,-1)
def sensitivity(prediction,reference,epsilon,admitted=None):
    p=np.asarray(prediction,float);y=np.asarray(reference,float);mask=np.ones(p.shape,bool) if admitted is None else np.asarray(admitted,bool)
    if p.ndim!=1 or p.shape!=y.shape or mask.shape!=p.shape:raise ValueError("unmatched scored rows")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon<0):raise ValueError("invalid uniform error bound")
    present=mask&np.isfinite(p);scored=present&np.isfinite(y);d=abs(p[scored]-y[scored])
    observed=float(np.quantile(d,.95)) if len(d) else None
    unknown=int(sum(present&~np.isfinite(y)));provided=epsilon is not None
    lower=np.maximum(0,d-epsilon) if provided else None;upper=d+epsilon if provided else None
    low=float(np.quantile(lower,.95)) if provided and len(d) else None
    high=float(np.quantile(upper,.95)) if provided and len(d) else None
    within=high is not None and high<=.5
    return {"attempts":len(p),"admitted":int(present.sum()),"scored":len(d),"unknown_reference":unknown,
      "observed_proxy_p95_mps":observed,"reference_bound_known":provided,"epsilon_assumption_mps":epsilon,
      "interpretation":"counterfactual sensitivity only; no reference bound verified by this function",
      "true_error_p95_lower":low,"true_error_p95_upper":high,
      "definitely_above2":int(sum(lower>2)) if provided else None,"possibly_above2":int(sum(upper>2)) if provided else None,
      "max_uniform_epsilon_for_scored_p95":.5-observed if observed is not None and observed<=.5 else None,
      "scored_subset_upper_within_0p5":within,
      "conditional_full_admitted_set_upper_within_0p5":bool(within and unknown==0 and len(d)>0)}
