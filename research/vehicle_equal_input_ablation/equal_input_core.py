"""Equal-input estimator controls; labels never enter fitting."""
import numpy as np

def fit_motion(level_rays, doppler, dimensions, robust):
    fail={"speed":None,"velocity":None,"reason":"invalid_geometry"}
    if dimensions not in (2,3):
        return {**fail,"reason":"invalid_dimensions"}
    a=np.asarray(level_rays,dtype=float);d=np.asarray(doppler,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or d.shape!=(len(a),) or len(a)<20:
        return fail
    if not np.isfinite(a).all() or not np.isfinite(d).all():
        return {**fail,"reason":"nonfinite"}
    design=a[:,:dimensions]
    if np.linalg.matrix_rank(design)<dimensions:
        return {**fail,"reason":"rank_deficient"}
    try:
        v=np.linalg.lstsq(design,-d,rcond=None)[0]
        for _ in range(20 if robust else 0):
            residual=design@v+d
            weights=np.minimum(1.,.3/np.maximum(abs(residual),1e-12))
            sw=np.sqrt(weights)
            v=np.linalg.lstsq(design*sw[:,None],-d*sw,rcond=None)[0]
        residual=design@v+d
        final_weights=np.minimum(1.,.3/np.maximum(abs(residual),1e-12)) if robust else np.ones(len(d))
        score=design.T@(final_weights*residual)
        objective=np.sum(np.where(abs(residual)<=.3,.5*residual**2,.3*(abs(residual)-.15))) if robust else .5*np.sum(residual**2)
        if not np.isfinite(v).all():return {**fail,"reason":"nonfinite_solution"}
        return {"speed":float(np.linalg.norm(v)),"velocity":v.tolist(),"reason":"numeric",
                "condition":float(np.linalg.cond(design.T@design)),
                "weighted_condition":float(np.linalg.cond(design.T@(final_weights[:,None]*design))),
                "stationarity_inf":float(np.max(abs(score))),"objective":float(objective),
                "iterations":20 if robust else 0,"points":len(d)}
    except np.linalg.LinAlgError:
        return {**fail,"reason":"linear_algebra"}
