"""Frozen v2 ablation: equal sensor mass; no CAN/INS input or timing changes."""
import math
import numpy as np
def fit_velocity(a,b,sensors):
    n=len(b)
    result=dict(accepted=False,reason="insufficient_detections",detections=n,state=None,inlier_indices=[])
    if a.shape!=(n,3) or len(sensors)!=n or not np.isfinite(a).all() or not np.isfinite(b).all():
        result["reason"]="invalid_numeric_data";return result
    unique,counts=np.unique(sensors,return_counts=True)
    if n<30 or len(unique)<3: return result
    weights=np.array([1.0/counts[np.searchsorted(unique,s)] for s in sensors])
    total=float(weights.sum());weighted=a*np.sqrt(weights)[:,None]
    if np.linalg.matrix_rank(weighted)<3 or np.linalg.cond(weighted)>1000:
        result["reason"]="unobservable_geometry";return result
    rng=np.random.default_rng(20261021);best=None;best_score=(-1,float("-inf"))
    indexes_by_sensor={s:np.flatnonzero(sensors==s) for s in unique}
    for _ in range(256):
        picked=rng.choice(unique,3,replace=False)
        indexes=np.array([rng.choice(indexes_by_sensor[s]) for s in picked])
        sample=a[indexes]
        if np.linalg.cond(sample)>1000: continue
        try: state=np.linalg.solve(sample,b[indexes])
        except np.linalg.LinAlgError: continue
        if math.hypot(*state[:2])>60 or abs(state[2])>2: continue
        residual=np.abs(a@state-b);mask=residual<=.25
        mass=float(weights[mask].sum())
        score=(mass/total,-float(np.average(residual[mask],weights=weights[mask])) if mass else float("-inf"))
        if score>best_score: best_score=score;best=mask
    if best is None:
        result["reason"]="no_plausible_consensus";return result
    mask=best
    for _ in range(3):
        if mask.sum()<3 or np.linalg.matrix_rank(weighted[mask])<3: break
        state=np.linalg.lstsq(weighted[mask],b[mask]*np.sqrt(weights[mask]),rcond=None)[0]
        updated=np.abs(a@state-b)<=.25
        if np.array_equal(updated,mask): break
        mask=updated
    if mask.sum()<3 or np.linalg.matrix_rank(weighted[mask])<3:
        result["reason"]="inlier_geometry";return result
    state=np.linalg.lstsq(weighted[mask],b[mask]*np.sqrt(weights[mask]),rcond=None)[0]
    residual=np.abs(a@state-b);mask=residual<=.25
    count=int(mask.sum());condition=float(np.linalg.cond(weighted[mask])) if count>=3 else float("inf")
    fraction=float(weights[mask].sum()/total)
    result.update(inliers=count,inlier_fraction=count/n,balanced_inlier_fraction=fraction,
        inlier_sensors=len(set(sensors[mask].tolist())),matrix_condition=condition if math.isfinite(condition) else None)
    if fraction<.5 or result["inlier_sensors"]<3:
        result["reason"]="insufficient_balanced_scene_support";return result
    if condition>1000 or math.hypot(*state[:2])>60 or abs(state[2])>2:
        result["reason"]="implausible_or_unobservable_motion";return result
    result.update(accepted=True,reason="radar_balanced_scene_consensus",state=state.tolist(),
        speed_mps=float(math.hypot(*state[:2])),inlier_indices=np.flatnonzero(mask).tolist(),
        inlier_residual_p95_mps=float(np.quantile(residual[mask],.95)))
    return result
