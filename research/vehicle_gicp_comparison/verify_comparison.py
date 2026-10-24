"""Audit fixed inputs, native poses/quality checks, offline metrics and pinned binary."""
import hashlib,importlib.metadata,json,math,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
import small_gicp
ROOT=Path(__file__).resolve().parent;STREAM=ROOT.parent/"vehicle_lidar_stream";CAN=ROOT.parent/"vehicle_context_policy"
def main():
    report=json.loads((ROOT/"comparison_summary.json").read_text());protocol=json.loads((ROOT/"protocol_v1.json").read_text())
    ph=hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()
    assert ph==report["protocol_sha256"]==json.loads((ROOT/"protocol_freeze.json").read_text())["sha256"]
    installed=json.loads((ROOT/"backend_install.json").read_text())
    wheel=ROOT/"runtime"/installed["wheel"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==installed["sha256"]
    binary=Path(small_gicp.__file__)
    with zipfile.ZipFile(wheel) as z:
        member=next(x for x in z.namelist() if Path(x).name==binary.name)
        assert hashlib.sha256(z.read(member)).hexdigest()==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert importlib.metadata.version("small-gicp")==installed["version"]=="1.0.1"
    source=STREAM/report["source_runtime_relative_to_stream"]
    assert hashlib.sha256((source/"estimates.jsonl").read_bytes()).hexdigest()==report["source_input_trace_sha256"]
    with (source/"estimates.jsonl").open() as f: original={(x["recording"],x["index"]):x for x in map(json.loads,f)}
    run=ROOT/report["runtime_relative_path"]
    with (run/"estimates.jsonl").open() as f: rows=[json.loads(l) for l in f]
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    with (cr/"packet_events.jsonl").open() as f:
        references={(e["recording"],e["index"]):e["reference"] for e in map(json.loads,f) if e["case"]=="nominal"}
    seen=set();poses=quality=0;nonempty=0;cache={}
    for key,r in original.items():
        p=source/r["input_file"];assert p.parent==source
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r["input_sha256"]
        with np.load(p) as data:
            cp,pp,mount=data["current"],data["previous"],data["rear_from_lidar"]
            cache[key]=(cp.copy(),pp.copy(),mount.copy())
            if min(len(cp),len(pp))>=150: nonempty+=1
    for r in rows:
        key=r["recording"],r["index"];unique=key,r["method"]
        assert unique not in seen;seen.add(unique)
        old=original[key];assert r["input_file"]==old["input_file"] and r["input_sha256"]==old["input_sha256"]
        assert r["now_ns"]==old["now_ns"] and r["dt_seconds"]==old["dt_seconds"]
        assert r["velocity_midpoint_ns"]==old["velocity_midpoint_ns"] and r["lower_support_midpoint_ns"]==old["lower_support_midpoint_ns"]
        assert r["method_ns"]>=0 and r["end_to_end_deadline_eligible"] is None and not r["physical_speed_authorized"]
        e=r["estimate"];cp,pp,mount=cache[key]
        if r["method"]=="baseline_planar":
            assert e["accepted"]==old["estimate"]["accepted"]
            if e["accepted"]: assert abs(e["speed_mps"]-old["estimate"]["speed_mps"])<1e-9
            continue
        if not min(len(cp),len(pp))>=150:
            assert not e["accepted"] and e["reason"]=="invalid_input";continue
        target,_=small_gicp.preprocess_points(pp,downsampling_resolution=.25,num_neighbors=16,num_threads=1)
        source_cloud,_=small_gicp.preprocess_points(cp,downsampling_resolution=.25,num_neighbors=16,num_threads=1)
        tp=np.asarray(target.points())[:,:3];sp=np.asarray(source_cloud.points())[:,:3];norm=np.asarray(target.normals())[:,:3]
        jac=np.column_stack((norm[:,0],norm[:,1],np.cross(tp,norm)[:,2]))
        condition=float(np.linalg.cond(jac));assert abs(condition-e["reference_planar_condition"])<1e-8
        assert len(tp)==e["target_points"] and len(sp)==e["source_points"]
        if "lidar_pose" not in e:
            assert not e["accepted"] and (condition>1000 or np.linalg.matrix_rank(jac)<3);continue
        pose=np.array(e["lidar_pose"]);rear=np.array(e["rear_pose"]);h=np.array(e["hessian"])
        np.testing.assert_allclose(pose[3],[0,0,0,1],atol=1e-12)
        np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8)
        np.testing.assert_allclose(rear,mount@pose@np.linalg.inv(mount),atol=1e-10)
        assert h.shape==(6,6) and np.isfinite(h).all()
        moved=sp@pose[:3,:3].T+pose[:3,3];distance,_=cKDTree(tp).query(moved,k=1,workers=1)
        mask=distance<=1.5;p95=float(np.quantile(distance[mask],.95)) if mask.any() else None
        assert e["inliers"]==int(mask.sum()) and e["overlap"]==float(mask.mean())
        assert p95==e["euclidean_inlier_p95_m"]
        speed=float(np.hypot(rear[0,3],rear[1,3])/r["dt_seconds"])
        yaw=math.atan2(rear[1,0],rear[0,0])/r["dt_seconds"]
        accepted=bool(e["converged"] and e["native_inliers"]>=100 and mask.sum()>=100 and mask.mean()>=.5 and p95 is not None and p95<=.3 and speed<=60 and abs(yaw)<=2)
        assert e["accepted"]==accepted
        if accepted:
            assert abs(e["speed_mps"]-speed)<1e-10 and abs(e["yaw_rate"]-yaw)<1e-10
        poses+=1;quality+=1
    assert len(seen)==report["variant_outputs"]==4*283
    for method in ("baseline_planar","ICP","PLANE_ICP","GICP"):
        rr=[r for r in rows if r["method"]==method];accepted=[r for r in rr if r["estimate"]["accepted"]]
        errors=[abs(r["estimate"]["speed_mps"]-references[r["recording"],r["index"]]) for r in accepted if references[r["recording"],r["index"]] is not None]
        m=report["metrics"][method]
        assert m["accepted"]==len(accepted) and m["scored"]==len(errors)
        assert m["large_discrepancies_above_2_mps"]==sum(x>2 for x in errors)
        assert m["p95_query_discrepancy_mps"]==(float(np.quantile(errors,.95)) if errors else None)
        assert m["method_p95_ms"]==float(np.quantile([r["method_ns"] for r in rr],.95)/1e6)
        assert m["reasons"]==dict(Counter(r["estimate"]["reason"] for r in rr))
    g=report["metrics"]["GICP"]
    trigger=bool(g["unscorable_accepted"]==0 and g["p95_query_discrepancy_mps"] is not None and g["p95_query_discrepancy_mps"]<=.5 and g["large_discrepancies_above_2_mps"]==0 and g["acceptance_pct_nonempty"]>=95)
    assert trigger==report["full_raw_replay_trigger_passed"]
    assert not report["promotion_allowed"] and not report["production_policy_changed"]
    result=dict(all_passed=True,query_inputs_checked=len(cache),variant_outputs_checked=len(seen),
        native_pose_checks=poses,native_quality_checks=quality,installed_binary_matches_pinned_wheel=True,
        baseline_results_unchanged=True,full_raw_replay_trigger_passed=trigger,
        end_to_end_native_timing_claim_allowed=False,physical_promotion_allowed=False)
    (ROOT/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
