"""One frozen comparison of three native registration variants on unchanged saved inputs."""
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from backend_motion import estimate
ROOT=Path(__file__).resolve().parent;STREAM=ROOT.parent/"vehicle_lidar_stream";CAN=ROOT.parent/"vehicle_context_policy"
spec=importlib.util.spec_from_file_location("previous_motion",STREAM/"motion.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def main():
    protocol=json.loads((ROOT/"protocol_v1.json").read_text())
    sha=hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()
    assert sha==json.loads((ROOT/"protocol_freeze.json").read_text())["sha256"]
    previous=json.loads((STREAM/"stream_benchmark_summary.json").read_text());source=STREAM/previous["runtime_relative_path"]
    with (source/"estimates.jsonl").open() as f: inputs=[json.loads(l) for l in f]
    run=ROOT/"runtime"/("run_"+os.urandom(8).hex());run.mkdir()
    rows=[];nonempty=0
    with (run/"estimates.jsonl").open("w") as out:
        for original in inputs:
            p=source/original["input_file"]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==original["input_sha256"]
            with np.load(p) as d:
                current,previous_points,mount=d["current"],d["previous"],d["rear_from_lidar"]
                if min(len(current),len(previous_points))>=150: nonempty+=1
                for method in ("baseline_planar","ICP","PLANE_ICP","GICP"):
                    start=time.perf_counter_ns()
                    if method=="baseline_planar":
                        if original["dt_seconds"] is None:
                            fitted=dict(accepted=False,reason="no_causally_available_lidar_windows",speed_mps=None,yaw_rate=None)
                        else: fitted=old.estimate(current,previous_points,original["dt_seconds"],mount)
                        assert fitted["accepted"]==original["estimate"]["accepted"]
                        if fitted["accepted"]: assert abs(fitted["speed_mps"]-original["estimate"]["speed_mps"])<1e-9
                    else: fitted=estimate(current,previous_points,original["dt_seconds"],mount,method)
                    duration=time.perf_counter_ns()-start
                    row=dict(recording=original["recording"],index=original["index"],now_ns=original["now_ns"],
                        method=method,estimate=fitted,method_ns=duration,input_file=original["input_file"],
                        input_sha256=original["input_sha256"],dt_seconds=original["dt_seconds"],
                        velocity_midpoint_ns=original["velocity_midpoint_ns"],lower_support_midpoint_ns=original["lower_support_midpoint_ns"],
                        end_to_end_deadline_eligible=None,physical_speed_authorized=False)
                    rows.append(row);out.write(json.dumps(row,allow_nan=False)+"\n")
            if original["index"]%100==0: print(original["recording"],original["index"],"all variants complete",flush=True)
    # Scoring starts only after every variant's output has been recorded.
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    with (cr/"packet_events.jsonl").open() as f: reference={(e["recording"],e["index"]):e["reference"] for e in map(json.loads,f) if e["case"]=="nominal"}
    metrics={};per_record={}
    for method in ("baseline_planar","ICP","PLANE_ICP","GICP"):
        rr=[r for r in rows if r["method"]==method];accepted=[r for r in rr if r["estimate"]["accepted"]]
        errors=[abs(r["estimate"]["speed_mps"]-reference[r["recording"],r["index"]]) for r in accepted if reference[r["recording"],r["index"]] is not None]
        metrics[method]=dict(queries=len(rr),available_nonempty_inputs=nonempty,accepted=len(accepted),
            acceptance_pct_all_queries=100*len(accepted)/len(rr),acceptance_pct_nonempty=100*len(accepted)/nonempty,
            scored=len(errors),unscorable_accepted=len(accepted)-len(errors),
            p95_query_discrepancy_mps=float(np.quantile(errors,.95)) if errors else None,
            maximum_query_discrepancy_mps=max(errors) if errors else None,large_discrepancies_above_2_mps=sum(e>2 for e in errors),
            method_p50_ms=float(np.median([r["method_ns"] for r in rr])/1e6),
            method_p95_ms=float(np.quantile([r["method_ns"] for r in rr],.95)/1e6),
            reasons=dict(Counter(r["estimate"]["reason"] for r in rr)))
        per_record[method]={name:dict(queries=sum(r["recording"]==name for r in rr),
            accepted=sum(r["recording"]==name for r in accepted),
            large_errors=sum(abs(r["estimate"]["speed_mps"]-reference[r["recording"],r["index"]])>2 for r in accepted if r["recording"]==name and reference[r["recording"],r["index"]] is not None))
            for name in previous["records"]}
    candidate=metrics["GICP"]
    trigger=bool(candidate["unscorable_accepted"]==0 and candidate["p95_query_discrepancy_mps"] is not None and
        candidate["p95_query_discrepancy_mps"]<=.5 and candidate["large_discrepancies_above_2_mps"]==0 and
        candidate["acceptance_pct_nonempty"]>=95)
    result=dict(protocol_sha256=sha,runtime_relative_path=str(run.relative_to(ROOT)),
        source_runtime_relative_to_stream=previous["runtime_relative_path"],source_input_trace_sha256=hashlib.sha256((source/"estimates.jsonl").read_bytes()).hexdigest(),
        query_count=len(inputs),variant_outputs=len(rows),nonempty_inputs=nonempty,metrics=metrics,per_record=per_record,
        original_baseline_results_reproduced=True,full_raw_replay_trigger_passed=trigger,
        native_end_to_end_timing_evaluated=False,native_policy_coverage_claim_allowed=False,
        acquisition_age_certified=False,production_policy_changed=False,promotion_allowed=False,
        comparison_scope="general-purpose compiled registration on fixed LiDAR interval pairs; not a full published AV defense/odometry reproduction")
    (ROOT/"comparison_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(metrics=metrics,full_raw_replay_trigger_passed=trigger),indent=2),flush=True)
if __name__=="__main__": main()
