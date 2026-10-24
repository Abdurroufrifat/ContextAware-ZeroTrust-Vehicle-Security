"""Run frozen descriptive comparisons and independent pair verification on three development clips."""
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import verify_pilot
from alignment import causal_sample
from development_metrics import bin_name, rear_wheel_speed, summarize
ROOT=Path(__file__).resolve().parent
def analyze(name,folder,protocol):
    verify_pilot.ROOT=folder
    verify_pilot.run()
    source_path=folder/"raw_data"/"extracted"/"ego_streams.jsonl"
    pairs_path=source_path.parent/"development_pairs.csv"
    rows=[json.loads(line) for line in source_path.open()]
    pairs=list(csv.DictReader(pairs_path.open()))
    samples=[]
    latencies=defaultdict(list)
    for r in rows:
        if r["topic"]=="tf_static": continue
        m=r["message"]
        s=m["header"]["stamp"]
        sensor=s["sec"]*10**9+s["nanosec"]
        latencies[r["topic"]].append((r["receipt_ns"]-sensor)/1e6)
        if r["topic"]=="localization/egomotion/odom" and m["child_frame_id"]!="vehicle/rear_axis":
            raise RuntimeError("Unexpected egomotion velocity frame")
        if r["topic"]=="actuator/vehicle/feedback":
            samples.append((r["receipt_ns"],sensor,m))
    refs,outputs,groups=[],defaultdict(list),defaultdict(list)
    context_counts=Counter()
    for i,pair in enumerate(pairs):
        receipt,sensor=int(pair["receipt_ns"]),int(pair["sensor_ns"])
        matched=causal_sample(samples,receipt,sensor,100_000_000)
        rear=rear_wheel_speed(matched[2]) if matched else None
        yaw=None
        if matched:
            field=matched[2]["onboard_yaw_rate_esp"]
            if field["availability"]["data"] in (1,2): yaw=abs(field["mean"])
        reference=float(pair["ins_rear_axis_speed_mps"])
        refs.append(reference)
        outputs["four_wheel_median"].append(float(pair["wheel_median_mps"]) if pair["wheel_median_mps"] else None)
        outputs["wheel_based_egomotion"].append(float(pair["egomotion_speed_mps"]) if pair["egomotion_speed_mps"] else None)
        outputs["rear_wheel_median"].append(rear)
        speed_bin=bin_name(reference,protocol["analysis"]["strata"]["speed_mps"])
        yaw_bin=bin_name(yaw,protocol["analysis"]["strata"]["abs_vehicle_esp_yaw_radps"]) if yaw is not None else "unavailable"
        for key in ("speed_mps:"+speed_bin,"abs_esp_yaw_radps:"+yaw_bin):
            groups[key].append(i)
            context_counts[key]+=1
    overall={key:summarize(values,refs) for key,values in outputs.items()}
    strata={key:{method:summarize([values[i] for i in indexes],[refs[i] for i in indexes])
        for method,values in outputs.items()} for key,indexes in groups.items()}
    latency_summary={key:dict(count=len(v),min_ms=min(v),max_ms=max(v),negative_receipts=sum(x<0 for x in v))
        for key,v in latencies.items()}
    pilot=json.loads((folder/"pilot_summary.json").read_text())
    result=dict(recording=name,development_only=True,queries=len(pairs),overall=overall,
        span_seconds=pilot["span_seconds"],ins_speed_mps=pilot["ins_speed_mps"],
        source_extracted_sha256=pilot["source_extracted_sha256"],
        fix_status_counts=pilot["fix_status_counts"],
        ins_velocity_variance_diagonal_range=pilot["ins_velocity_variance_diagonal_range"],
        reference_offset_ins_to_rear_axis_m=pilot["reference_offset_ins_to_rear_axis_m"],
        wheel_age_ms=pilot["wheel_age_ms"],
        fixed_strata=strata,context_counts=context_counts,receipt_minus_sensor_time=latency_summary,
        numerical_gate_only=True,physical_gate_pass=None,independent_confirmation=False)
    (folder/"broader_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
def main():
    protocol_path=ROOT/"broader_development_protocol_v1.json"
    protocol=json.loads(protocol_path.read_text())
    freeze=json.loads((ROOT/"broader_protocol_freeze.json").read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==freeze["protocol_sha256"]
    records=[("07_turning",ROOT)]+[(item["id"],ROOT/"raw_data"/item["id"]) for item in protocol["recordings"]]
    results=[analyze(name,folder,protocol) for name,folder in records]
    summary=dict(protocol_sha256=freeze["protocol_sha256"],recordings=results,
        total_queries=sum(r["queries"] for r in results),recording_count=len(results),
        source_day_count=1,independent_confirmation=False,policy_promotion_allowed=False,
        reference_used_in_policy=False,physical_gate_pass=None,
        limitations=["Three development clips from one acquisition day; not independent sessions",
            "Comparisons are wheel-derived estimators, not published zero-trust methods",
            "No attacks, authorization outcomes, or learned AI contribution are evaluated here",
            "INS estimates and mounting calibration do not certify a hard physical-error bound"])
    (ROOT/"broader_development_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    flat=[]
    for record in results:
        for method,metrics in record["overall"].items():
            flat.append(dict(recording=record["recording"],method=method,**metrics))
    with (ROOT/"broader_development_comparison.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(dict(total_queries=summary["total_queries"],comparisons=flat),indent=2),flush=True)
if __name__=="__main__": main()
