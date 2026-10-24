from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development"
rows=json.loads((R/"source_stress_rows.json").read_text());summary=json.loads((R/"source_stress_summary.json").read_text())
assert len(rows)==564*11 and len({(r["sequence"],r["index"],r["case"]) for r in rows})==len(rows)
for name,h in summary["unchanged_input_hashes"].items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==h
for case,reported in summary["cases"].items():
 group=[r for r in rows if r["case"]==case];good=[r for r in group if r["quality_passed"] and r.get("speed_mps") is not None]
 assert len(group)==reported["attempts"]==564 and len(good)==reported["quality_passed"]
 shifts=[abs(r["speed_mps"]-r["nominal_speed_mps"]) for r in good]
 assert sum(v>2 for v in shifts)==reported["quality_passed_nominal_shift_above2"]
 if shifts:assert abs(np.percentile(shifts,95)-reported["nominal_shift_p95_mps"])<1e-12
 for key,label in [("reference_primary_mps","primary"),("reference_INS_mps","INS")]:
  err=[abs(r["speed_mps"]-r[key]) for r in good if r[key] is not None];rep=reported[label]
  assert len(err)==rep["scored"] and sum(e>2 for e in err)==rep["above2"]
  if err:assert abs(np.percentile(err,95)-rep["p95_mps"])<1e-12
 if case=="nominal":assert all(abs(r["speed_mps"]-r["nominal_speed_mps"])<1e-10 for r in good)
 if case=="coherent_speed_plus3":assert len(good)==564 and all(abs(r["speed_mps"]-r["nominal_speed_mps"]-3)<1e-10 for r in good)
 if case in ["imu_missing","quaternion_zero","imu_stale100ms_plus1ns","replay1s_original_timestamp"]:assert len(good)==0
 if case=="replay1s_forged_timestamp":
  assert all(r["request_stamp_ns"]-r["true_source_stamp_ns"]>=1_000_000_000 and r["request_stamp_ns"]==r["claimed_stamp_ns"] for r in good)
diag=json.loads((R/"reference_diagnosis.json").read_text())
for seq in ["Library01","Library02"]:
 a=np.load(R/(seq+"_reference_diagnostics.npz"),allow_pickle=False);saved=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
 np.testing.assert_array_equal(a["stamps"],saved["stamps"]);np.testing.assert_array_equal(a["candidate"],saved["speed"])
 np.testing.assert_allclose(a["pose_speed"],saved["reference_primary"],equal_nan=True)
 e=abs(a["pose_speed"]-a["raw_position_speed"]);e=e[np.isfinite(e)]
 assert abs(np.percentile(e,95)-diag[seq]["bands"]["all"]["pose_derivative_vs_raw_position_derivative"]["p95"])<1e-12
 for name,h in diag[seq]["source_hashes"].items():assert hashlib.sha256((H/"data"/seq/"sensor_data"/name).read_bytes()).hexdigest()==h
for e in json.loads((H/"download_lock.json").read_text())["files"]:
 assert e["sequence"] in ["Library01","Library02"]
 assert hashlib.sha256((H/e["relative_path"]).read_bytes()).hexdigest()==e["sha256"]
tests=json.loads((R/"test_validation.json").read_text());assert tests["core"]["exit_code"]==tests["pilot"]["exit_code"]==0 and tests["denials"]==[]
assert not (H/"data"/"Library03").exists()
out={"passed":True,"attack_rows_recomputed":len(rows),"nominal_frame_reproductions":564,"coherent_forgery_quality_passes":564,"coherent_forgery_relative_speed_shift_mps":3,"original_source_and_results_unchanged":True,"reference_pair_replays":2,"core_tests":tests["core"]["counts"],"focused_tests":tests["pilot"]["counts"],"Library03_payloads_accessed":False,"original_segment_failure_retained":True,"scope":"artifact/arithmetic checks; source-integrity failure is an observed research result, not a passing defense"}
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
