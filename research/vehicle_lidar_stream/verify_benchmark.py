"""Independent input-window, pose, timing and offline reference audit."""
import bisect,hashlib,json,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent;CAN=ROOT.parent/"vehicle_context_policy"
def reference(values,ns):
    us=ns/1000
    times=[v[0] for v in values];i=bisect.bisect_left(times,us)
    if i<len(values) and times[i]==us: return values[i][1]
    if i==0 or i==len(values) or times[i]-times[i-1]>50000: return None
    return values[i-1][1]+(us-times[i-1])/(times[i]-times[i-1])*(values[i][1]-values[i-1][1])
def normals(points):
    _,near=cKDTree(points).query(points,k=min(16,len(points)),workers=1)
    patches=points[near];patches=patches-patches.mean(axis=1,keepdims=True)
    cov=np.matmul(patches.transpose(0,2,1),patches)/patches.shape[1]
    eig,vec=np.linalg.eigh(cov)
    normal=vec[:,:,0]
    return normal,(np.abs(normal[:,2])<=.95)&((eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-12)>=.3)
def main():
    report=json.loads((ROOT/"stream_benchmark_summary.json").read_text());runtime=ROOT/report["runtime_relative_path"]
    assert hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()==report["protocol_sha256"]
    assert json.loads((ROOT/"protocol_freeze.json").read_text())["sha256"]==report["protocol_sha256"]
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    values=json.loads((cr/"offline_references.json").read_text())
    with (cr/"packet_events.jsonl").open() as f:
        nominal={(e["recording"],e["index"]):e for e in map(json.loads,f) if e["case"]=="nominal"}
    with (runtime/"estimates.jsonl").open() as f: records=[json.loads(l) for l in f]
    checked=pose_checks=quality_checks=0;seen=set();interval_errors=[];query_errors=[];bad=[];latencies=[]
    for r in records:
        key=r["recording"],r["index"];assert key not in seen;seen.add(key)
        assert r["now_ns"]==nominal[key]["now_us"]*1000
        assert not r["physical_speed_authorized"] and not r["instantaneous_speed_certified"]
        p=runtime/r["input_file"];assert p.parent==runtime
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r["input_sha256"]
        with np.load(p) as data:
            current,previous=data["current"],data["previous"];ct,pt=data["current_times"],data["previous_times"]
            mount=data["rear_from_lidar"]
            assert current.shape==(len(ct),3) and previous.shape==(len(pt),3)
            assert len(ct)==r["current_points"] and len(pt)==r["previous_points"]
            assert len(ct)<=2000 and len(pt)<=2000 and np.isfinite(current).all() and np.isfinite(previous).all()
            now=r["now_ns"]
            assert np.all((ct<=now)&(ct>=now-20000000)) and np.all((pt<=now-100000000)&(pt>=now-120000000))
            assert r["latest_input_available_ns"]<=now
            if len(ct) and len(pt):
                center_c=now+round(float(np.median(ct-now)));center_p=now+round(float(np.median(pt-now)))
                assert r["velocity_midpoint_ns"]==(center_c+center_p)//2
                assert r["lower_support_midpoint_ns"]==(int(ct.min())+int(pt.min()))//2
                assert r["dt_seconds"]==(center_c-center_p)/1e9
            assert r["algorithm_ns"]>=0
            assert r["completed_ns"]==max(now,r["worker_before_ns"])+r["algorithm_ns"]
            assert r["queue_wait_ns"]==max(0,r["worker_before_ns"]-now)
            fitted=r["estimate"]
            eligible=bool(fitted["accepted"] and r["velocity_midpoint_ns"] is not None and
                0<=r["completed_ns"]-r["velocity_midpoint_ns"]<=100000000 and
                0<=r["completed_ns"]-r["lower_support_midpoint_ns"]<=100000000)
            assert r["deadline_eligible"]==eligible
            if fitted["accepted"]:
                a=np.array(fitted["lidar_pose"]);b=np.array(fitted["rear_pose"])
                np.testing.assert_allclose(a[3],[0,0,0,1],atol=1e-12)
                np.testing.assert_allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-10)
                assert abs(np.linalg.det(a[:3,:3])-1)<1e-10
                np.testing.assert_allclose(b,mount@a@np.linalg.inv(mount),atol=1e-10)
                assert abs(np.linalg.norm(b[:2,3])/r["dt_seconds"]-fitted["speed_mps"])<1e-10
                assert abs(math.atan2(b[1,0],b[0,0])/r["dt_seconds"]-fitted["yaw_rate"])<1e-10
                sn,sm=normals(current);tn,tm=normals(previous)
                source=current[sm];target=previous[tm];tn=tn[tm]
                moved=source@a[:3,:3].T+a[:3,3]
                distance,index=cKDTree(target).query(moved,k=1,workers=1);mask=distance<=1.5
                residual=np.abs(np.sum(tn[index[mask]]*(moved[mask]-target[index[mask]]),axis=1))
                chosen=np.sort(residual)[:max(100,math.ceil(.7*len(residual)))]
                p95=float(np.quantile(chosen,.95))
                assert int(mask.sum())==fitted["correspondences"]
                assert abs(p95-fitted["residual_p95_m"])<1e-10
                assert fitted["converged"] and fitted["condition"]<=1000 and mask.mean()>=.5 and p95<=.3
                assert fitted["speed_mps"]<=60 and abs(fitted["yaw_rate"])<=2
                pose_checks+=1;quality_checks+=1
                query_ref=nominal[key]["reference"];mid_ref=reference(values[r["recording"]],r["velocity_midpoint_ns"])
                qe=abs(fitted["speed_mps"]-query_ref) if query_ref is not None else None
                me=abs(fitted["speed_mps"]-mid_ref) if mid_ref is not None else None
                if qe is not None: query_errors.append(qe)
                if me is not None: interval_errors.append(me)
                if qe is not None and qe>2:
                    bad.append(dict(recording=r["recording"],index=r["index"],speed=fitted["speed_mps"],
                        query_reference=query_ref,query_error=qe,midpoint_reference=mid_ref,midpoint_error=me,
                        condition=fitted["condition"],overlap=fitted["overlap"],residual_p95_m=p95))
            latencies.append((r["completed_ns"]-now)/1e6);checked+=1
    assert checked==report["query_count"]==283 and set(seen)==set(nominal)
    for gate in ("geometry_accepted","deadline_eligible"):
        good=[r for r in records if r["estimate"]["accepted"] and (gate=="geometry_accepted" or r["deadline_eligible"])]
        errors=[abs(r["estimate"]["speed_mps"]-nominal[r["recording"],r["index"]]["reference"]) for r in good if nominal[r["recording"],r["index"]]["reference"] is not None]
        m=report["metrics"][gate]
        assert m["accepted"]==len(good) and m["scored"]==len(errors) and m["discrepancy_above_2_mps"]==sum(x>2 for x in errors)
        assert m["p95_query_time_discrepancy_mps"]==(float(np.quantile(errors,.95)) if errors else None)
    result=dict(all_passed=True,input_window_and_timing_checks=checked,accepted_pose_checks=pose_checks,
        accepted_registration_quality_checks=quality_checks,geometry_large_errors=bad,
        midpoint_time_reference_scored=len(interval_errors),
        midpoint_time_p95_discrepancy_mps=float(np.quantile(interval_errors,.95)) if interval_errors else None,
        midpoint_time_discrepancies_above_2_mps=sum(x>2 for x in interval_errors),
        request_completion_p95_ms=float(np.quantile(latencies,.95)),deadline_eligible=report["metrics"]["deadline_eligible"]["accepted"],
        acquisition_age_certified=False,promotion_allowed=False,
        verification_scope="Saved input windows/hashes, pose geometry, quality, timing arithmetic, and offline reference metrics; no physical acquisition attestation")
    (ROOT/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
