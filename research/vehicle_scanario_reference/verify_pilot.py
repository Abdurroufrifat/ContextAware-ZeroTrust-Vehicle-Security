"""Independent raw-extract check; does not import pilot/alignment/geometry helpers."""
import csv, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def run():
    src=ROOT/"raw_data"/"extracted"/"ego_streams.jsonl"
    with src.open() as inp: source=[json.loads(line) for line in inp]
    with (src.parent/"development_pairs.csv").open() as inp: pairs=list(csv.DictReader(inp))
    summary=json.loads((ROOT/"pilot_summary.json").read_text())
    extraction=json.loads((ROOT/"extraction_summary.json").read_text())
    with src.open("rb") as inp:
        assert hashlib.file_digest(inp,"sha256").hexdigest()==extraction["extracted_sha256"]
    def ns(msg):
        t=msg["header"]["stamp"]
        return t["sec"]*10**9+t["nanosec"]
    wheel_rows=[r for r in source if r["topic"]=="actuator/vehicle/feedback"]
    refs=[r for r in source if r["topic"]=="sensor/ins/oxts_rt3000/gps/odom"]
    calibration={}
    for row in source:
        if row["topic"]=="tf_static":
            for t in row["message"]["transforms"]:
                if t["child_frame_id"] in ("sensor/ins/oxts","vehicle/rear_axis"):
                    q=t["transform"]["rotation"]
                    assert t["header"]["frame_id"]=="base_link"
                    assert (q["x"],q["y"],q["z"],q["w"])==(0.,0.,0.,1.)
                    calibration[t["child_frame_id"]]=t["transform"]["translation"]
    ins_position=calibration["sensor/ins/oxts"]
    rear_position=calibration["vehicle/rear_axis"]
    published_offset=tuple(rear_position[k]-ins_position[k] for k in ("x","y","z"))
    errors=[]
    assert len(refs)==len(pairs)==summary["queries"]
    for ref,pair in zip(refs,pairs):
        receipt,sensor=ref["receipt_ns"],ns(ref["message"])
        assert int(pair["receipt_ns"])==receipt and int(pair["sensor_ns"])==sensor
        eligible=[]
        for w in wheel_rows:
            if w["receipt_ns"]<=receipt and 0<=sensor-ns(w["message"])<=100_000_000:
                values={x["id"]:x["speed"] for x in w["message"]["wheel_speed"] if x["id"] in (1,2,3,4)}
                if len(values)==4 and all(x["availability"]["data"] in (1,2) for x in values.values()):
                    eligible.append(w)
        expected=None
        if eligible:
            chosen=eligible[-1]
            values=[x["speed"]["mean"] for x in chosen["message"]["wheel_speed"] if x["id"] in (1,2,3,4)]
            expected=statistics.median(values)
            assert float(pair["wheel_median_mps"])==expected
            assert int(pair["wheel_age_ns"])==sensor-ns(chosen["message"])
        else:
            assert pair["wheel_median_mps"]=="" and pair["wheel_age_ns"]==""
        twist=ref["message"]["twist"]["twist"]
        v,w=twist["linear"],twist["angular"]
        rx,ry,rz=published_offset
        vx=v["x"]+w["y"]*rz-w["z"]*ry
        vy=v["y"]+w["z"]*rx-w["x"]*rz
        expected_ref=math.sqrt(vx*vx+vy*vy)
        assert abs(float(pair["ins_rear_axis_speed_mps"])-expected_ref)<1e-12
        if expected is not None: errors.append(abs(expected-expected_ref))
    p95=float(np.quantile(errors,.95,method="linear")) if errors else None
    recorded=summary["comparisons"]["wheel_median_mps"]["absolute_discrepancy_mps"]
    recorded_p95=recorded["p95"] if recorded else None
    assert (p95 is None and recorded_p95 is None) or (p95 is not None and recorded_p95 is not None and abs(p95-recorded_p95)<1e-12)
    assert summary["physical_gate_pass"] is None and not summary["policy_promotion_allowed"]
    report=dict(independent_pair_checks=len(pairs),accepted_pair_checks=len(errors),
        abstention_checks=len(pairs)-len(errors),all_passed=True,
        wheel_p95_discrepancy_mps=p95,final_evaluation=False)
    (ROOT/"verification_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": run()
