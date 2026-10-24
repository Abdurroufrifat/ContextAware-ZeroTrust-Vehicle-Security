"""Independent three-estimator raw-message check; no shared analysis helpers."""
import json,math,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def run():
    protocol=json.loads((ROOT/"broader_development_protocol_v1.json").read_text())
    summary=json.loads((ROOT/"broader_development_summary.json").read_text())
    reports=[]
    folders={"07_turning":ROOT,**{r["id"]:ROOT/"raw_data"/r["id"] for r in protocol["recordings"]}}
    for record in summary["recordings"]:
        folder=folders[record["recording"]]
        with (folder/"raw_data"/"extracted"/"ego_streams.jsonl").open() as inp:
            raw=[json.loads(line) for line in inp]
        def time(row):
            s=row["message"]["header"]["stamp"]
            return s["sec"]*10**9+s["nanosec"]
        feedback=[r for r in raw if r["topic"]=="actuator/vehicle/feedback"]
        ego=[r for r in raw if r["topic"]=="localization/egomotion/odom"]
        refs=[r for r in raw if r["topic"]=="sensor/ins/oxts_rt3000/gps/odom"]
        errors={k:[] for k in record["overall"]}
        paired=0
        offset=json.loads((folder/"pilot_summary.json").read_text())["reference_offset_ins_to_rear_axis_m"]
        rx,ry,rz=offset
        for ref in refs:
            t,receipt=time(ref),ref["receipt_ns"]
            eligible=[r for r in feedback if r["receipt_ns"]<=receipt and 0<=t-time(r)<=100_000_000]
            valid_four=[r for r in eligible if {w["id"] for w in r["message"]["wheel_speed"] if w["id"] in (1,2,3,4) and w["speed"]["availability"]["data"] in (1,2)}=={1,2,3,4}]
            valid_ego=[r for r in ego if r["receipt_ns"]<=receipt and 0<=t-time(r)<=100_000_000]
            predictions={}
            if valid_four:
                predictions["four_wheel_median"]=statistics.median(w["speed"]["mean"] for w in valid_four[-1]["message"]["wheel_speed"] if w["id"] in (1,2,3,4))
            if eligible:
                ws=[w for w in eligible[-1]["message"]["wheel_speed"] if w["id"] in (3,4)]
                if {w["id"] for w in ws}=={3,4} and all(w["speed"]["availability"]["data"] in (1,2) for w in ws):
                    predictions["rear_wheel_median"]=statistics.median(w["speed"]["mean"] for w in ws)
            if valid_ego:
                v=valid_ego[-1]["message"]["twist"]["twist"]["linear"]
                predictions["wheel_based_egomotion"]=math.hypot(v["x"],v["y"])
            twist=ref["message"]["twist"]["twist"]
            v,w=twist["linear"],twist["angular"]
            reference=math.hypot(v["x"]+w["y"]*rz-w["z"]*ry,v["y"]+w["z"]*rx-w["x"]*rz)
            for method,value in predictions.items(): errors[method].append(abs(value-reference))
            paired+=1
        for method,values in errors.items():
            result=record["overall"][method]
            assert result["queries"]==len(refs) and result["accepted"]==len(values)
            assert result["discrepancy_above_2_mps"]==sum(e>2 for e in values)
            expected=float(np.quantile(values,.95)) if values else None
            assert (expected is None and result["p95_discrepancy_mps"] is None) or abs(expected-result["p95_discrepancy_mps"])<1e-12
            assert abs(result["coverage_pct"]-100*len(values)/len(refs))<1e-12
        reports.append(dict(recording=record["recording"],queries=paired,estimator_checks=paired*3))
    report=dict(all_passed=True,records=reports,total_estimator_checks=sum(r["estimator_checks"] for r in reports))
    (ROOT/"broader_independent_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": run()
