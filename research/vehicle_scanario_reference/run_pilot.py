"""Development-only engineering comparison; INS never enters an authorization policy."""
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from alignment import causal_sample, planar_speed, wheel_median
from geometry import velocity_at_offset
ROOT = Path(__file__).resolve().parent
def stamp(message):
    s = message["header"]["stamp"]
    return s["sec"] * 1_000_000_000 + s["nanosec"]
def vector(obj):
    return tuple(obj[k] for k in ("x", "y", "z"))
def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95, method="linear")), max=float(a.max()))
def run():
    src = ROOT / "raw_data" / "extracted" / "ego_streams.jsonl"
    manifest = json.loads((ROOT / "extraction_summary.json").read_text())
    if hashlib.file_digest(src.open("rb"), "sha256").hexdigest() != manifest["extracted_sha256"]:
        raise RuntimeError("Extracted streams changed")
    streams = defaultdict(list)
    transforms = {}
    invalid_wheels = 0
    for line in src.open():
        row = json.loads(line)
        topic, msg = row["topic"], row["message"]
        if topic == "tf_static":
            for t in msg["transforms"]:
                if t["child_frame_id"] in ("sensor/ins/oxts", "vehicle/rear_axis"):
                    if t["child_frame_id"] in transforms and transforms[t["child_frame_id"]] != t:
                        raise RuntimeError("Conflicting static calibration")
                    transforms[t["child_frame_id"]] = t
            continue
        streams[topic].append((row["receipt_ns"], stamp(msg), msg))
    ins_tf, rear_tf = (transforms[x] for x in ("sensor/ins/oxts", "vehicle/rear_axis"))
    for t in (ins_tf, rear_tf):
        q = t["transform"]["rotation"]
        if t["header"]["frame_id"] != "base_link" or vector(q) != (0.,0.,0.) or q["w"] != 1.:
            raise RuntimeError("Pilot requires documented parallel INS/rear-axis frames")
    offset = tuple(b-a for a,b in zip(vector(ins_tf["transform"]["translation"]), vector(rear_tf["transform"]["translation"])))
    wheels = []
    for receipt, sensor, msg in streams["actuator/vehicle/feedback"]:
        parts = {w["id"]:w["speed"] for w in msg["wheel_speed"] if w["id"] in (1,2,3,4)}
        if len(parts)!=4 or any(w["availability"]["data"] not in (1,2) for w in parts.values()):
            invalid_wheels += 1
            continue
        wheels.append((receipt, sensor, wheel_median([parts[i]["mean"] for i in (1,2,3,4)])))
    ego = [(a,b,planar_speed(*vector(m["twist"]["twist"]["linear"])[:2]))
           for a,b,m in streams["localization/egomotion/odom"]]
    results, fixes, covariance = [], Counter(), []
    for receipt, sensor, msg in streams["sensor/ins/oxts_rt3000/gps/odom"]:
        if msg["child_frame_id"] != "sensor/ins/oxts":
            raise RuntimeError("Unexpected INS twist frame")
        twist = msg["twist"]["twist"]
        v = velocity_at_offset(vector(twist["linear"]), vector(twist["angular"]), offset)
        ref = planar_speed(*v[:2])
        wheel = causal_sample(wheels, receipt, sensor, 100_000_000)
        eg = causal_sample(ego, receipt, sensor, 100_000_000)
        fix = causal_sample(streams["sensor/ins/oxts_rt3000/gps/fix"], receipt, sensor, 100_000_000)
        fixes[str(fix[2]["status"]["status"]) if fix else "missing"] += 1
        cov = msg["twist"]["covariance"]
        covariance.append([cov[0],cov[7],cov[14]])
        results.append(dict(receipt_ns=receipt, sensor_ns=sensor, ins_rear_axis_speed_mps=ref,
            wheel_median_mps=wheel[2] if wheel else None,
            wheel_age_ns=sensor-wheel[1] if wheel else None,
            egomotion_speed_mps=eg[2] if eg else None,
            egomotion_age_ns=sensor-eg[1] if eg else None))
    outdir = ROOT / "raw_data" / "extracted"
    with (outdir / "development_pairs.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    comparisons = {}
    for field in ("wheel_median_mps", "egomotion_speed_mps"):
        errors = [abs(r[field]-r["ins_rear_axis_speed_mps"]) for r in results if r[field] is not None]
        comparisons[field] = dict(coverage_pct=100*len(errors)/len(results),
            absolute_discrepancy_mps=stats(errors), discrepancy_above_2_mps=sum(x>2 for x in errors))
    summary = dict(development_only=True, final_evaluation=False, queries=len(results),
        span_seconds=(results[-1]["sensor_ns"]-results[0]["sensor_ns"])/1e9,
        ins_speed_mps=stats([r["ins_rear_axis_speed_mps"] for r in results]),
        wheel_age_ms=stats([r["wheel_age_ns"]/1e6 for r in results if r["wheel_age_ns"] is not None]),
        comparisons=comparisons, fix_status_counts=fixes, invalid_wheel_messages=invalid_wheels,
        ins_velocity_variance_diagonal_range=[stats([c[i] for c in covariance]) for i in range(3)],
        reference_offset_ins_to_rear_axis_m=offset, fitted_lag_seconds=None,
        fitted_scale=None, physical_error_bound_established=False, physical_gate_pass=None,
        policy_promotion_allowed=False, reference_used_in_policy=False,
        source_extracted_sha256=manifest["extracted_sha256"],
        limitations=["Single short low-speed development recording", "INS accuracy estimates are not hard bounds",
            "Wheel median mixes turning-wheel speeds; no slip or steering correction",
            "Static transform used as published; calibration accuracy not independently certified",
            "ROS clocks assumed common; no fitted alignment", "Not an independent confirmation experiment"])
    (ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
if __name__ == "__main__":
    run()
