"""CAN-independent planar radar ego-motion; empirical quality gates are not error certificates."""
import math
import numpy as np
def quaternion_matrix(q):
    x,y,z,w=(float(q[k]) for k in ("x","y","z","w"))
    norm=math.sqrt(x*x+y*y+z*z+w*w)
    if not math.isfinite(norm) or abs(norm-1)>1e-3: raise ValueError("Invalid calibration quaternion")
    x,y,z,w=(v/norm for v in (x,y,z,w))
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def mounting(transforms):
    graph={}
    for t in transforms:
        mat=np.eye(4);mat[:3,:3]=quaternion_matrix(t["transform"]["rotation"])
        mat[:3,3]=[t["transform"]["translation"][k] for k in ("x","y","z")]
        child=t["child_frame_id"].lstrip("/");parent=t["header"]["frame_id"].lstrip("/")
        if child in graph:
            assert graph[child][0]==parent and np.allclose(graph[child][1],mat,atol=1e-10)
        graph[child]=(parent,mat)
    def resolve(frame,seen=()):
        if frame=="base_link": return np.eye(4)
        if frame in seen or frame not in graph: raise ValueError("Missing/cyclic calibration frame")
        parent,local=graph[frame]
        return resolve(parent,seen+(frame,))@local
    base_from_rear=resolve("vehicle/rear_axis")
    return {int(frame.rsplit("/",1)[1]):np.linalg.inv(base_from_rear)@resolve(frame)
        for frame in graph if frame.startswith("sensor/radar/umrr/") and frame.rsplit("/",1)[1].isdigit()}
def design(frames,calibration):
    rows=[];values=[];sensors=[]
    for frame in frames:
        sensor=frame["sensor"];transform=calibration[sensor]
        rotation=transform[:3,:3];position=transform[:3,3]
        for r,radial,azimuth,elevation,rcs,power,noise in frame["targets"]:
            if r<=0: continue
            direction=rotation@np.array([math.cos(elevation)*math.cos(azimuth),
                math.cos(elevation)*math.sin(azimuth),math.sin(elevation)])
            dx,dy=direction[:2]
            rows.append([dx,dy,-dx*position[1]+dy*position[0]])
            values.append(-radial);sensors.append(sensor)
    return np.asarray(rows,dtype=float).reshape((-1,3)),np.asarray(values,dtype=float),np.asarray(sensors,dtype=int)
def fit_velocity(a,b,sensors):
    n=len(b)
    result=dict(accepted=False,reason="insufficient_detections",detections=n,state=None,inlier_indices=[])
    if a.shape!=(n,3) or len(sensors)!=n or not np.isfinite(a).all() or not np.isfinite(b).all():
        result["reason"]="invalid_numeric_data";return result
    if n<30 or len(set(sensors.tolist()))<3: return result
    if np.linalg.matrix_rank(a)<3 or np.linalg.cond(a)>1000:
        result["reason"]="unobservable_geometry";return result
    rng=np.random.default_rng(20261021);best=None;best_score=(-1,float("-inf"))
    for _ in range(256):
        indexes=rng.choice(n,3,replace=False)
        if len(set(sensors[indexes].tolist()))<3: continue
        sample=a[indexes]
        if np.linalg.cond(sample)>1000: continue
        try: state=np.linalg.solve(sample,b[indexes])
        except np.linalg.LinAlgError: continue
        if math.hypot(*state[:2])>60 or abs(state[2])>2: continue
        residual=np.abs(a@state-b);mask=residual<=.25
        count=int(mask.sum())
        score=(count,-float(np.median(residual[mask])) if count else float("-inf"))
        if score>best_score: best_score=score;best=mask
    if best is None:
        result["reason"]="no_plausible_consensus";return result
    mask=best
    for _ in range(3):
        if mask.sum()<3 or np.linalg.matrix_rank(a[mask])<3: break
        state=np.linalg.lstsq(a[mask],b[mask],rcond=None)[0]
        next_mask=np.abs(a@state-b)<=.25
        if np.array_equal(next_mask,mask): break
        mask=next_mask
    if mask.sum()<3 or np.linalg.matrix_rank(a[mask])<3:
        result["reason"]="inlier_geometry";return result
    state=np.linalg.lstsq(a[mask],b[mask],rcond=None)[0]
    residual=np.abs(a@state-b)
    mask=residual<=.25
    count=int(mask.sum());condition=float(np.linalg.cond(a[mask])) if count>=3 else float("inf")
    result.update(inliers=count,inlier_fraction=count/n,inlier_sensors=len(set(sensors[mask].tolist())),
        matrix_condition=condition if math.isfinite(condition) else None)
    if count<math.ceil(.5*n) or result["inlier_sensors"]<3:
        result["reason"]="insufficient_static_scene_support";return result
    if condition>1000 or math.hypot(*state[:2])>60 or abs(state[2])>2:
        result["reason"]="implausible_or_unobservable_motion";return result
    result.update(accepted=True,reason="radar_static_scene_consensus",state=state.tolist(),
        speed_mps=float(math.hypot(*state[:2])),inlier_indices=np.flatnonzero(mask).tolist(),
        inlier_residual_p95_mps=float(np.quantile(residual[mask],.95)))
    return result

def select_frames(frames,now_ns,track):
    field={"raw_header_association":"raw_header_ns","processed_corrected_time":"corrected_header_ns"}[track]
    selected={}
    for frame in frames:
        stamp=frame.get(field)
        if frame["available_ns"]>now_ns or stamp is None or not 0<=now_ns-stamp<=100_000_000: continue
        sensor=frame["sensor"]
        if sensor not in selected or frame["raw_header_ns"]>selected[sensor]["raw_header_ns"]:
            selected[sensor]=frame
    return [selected[k] for k in sorted(selected)]
