import hashlib,importlib.metadata,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;project=R.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((R/"benchmark_summary.json").read_text());v=json.loads((R/"independent_verification.json").read_text());t=json.loads((R/"test_validation.json").read_text());f=json.loads((R/"timing_feasibility.json").read_text())
assert v["all_passed"] and all(x["exit_code"]==0 for x in t.values())
assert sum(x["count"] for x in t.values())==119
prior={}
for folder,manifest in [(R.parent/"vehicle_gicp_comparison","file_manifest.json"),(R.parent/"vehicle_lidar_stream","file_manifest.json"),(R.parent/"vehicle_radar_context","source_timing_file_manifest.json"),(R.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
 expected=json.loads((folder/manifest).read_text())
 for name,h in expected.items():assert (folder/name).parent==folder and sha(folder/name)==h,(folder,name)
 prior[folder.name+"/"+manifest]=len(expected)
from kiss_icp.pybind import kiss_icp_pybind
native=Path(kiss_icp_pybind.__file__);pin=json.loads((R/"backend_pin.json").read_text());wheel=Path(pin["wheel"])
assert sha(wheel)==pin["sha256"]
with zipfile.ZipFile(wheel) as z:
 matches=[n for n in z.namelist() if n.endswith("/"+native.name)]
 assert len(matches)==1 and hashlib.sha256(z.read(matches[0])).hexdigest()==sha(native)
env={n:importlib.metadata.version(n) for n in ["numpy","scipy","kiss-icp","small-gicp","pydantic","pydantic-settings"]}
(R/"environment.json").write_text(json.dumps(dict(versions=env,api_only_minimal_install=True,native_binary_sha256=sha(native)),indent=2)+"\n")
run=R/s["runtime_relative_path"]
frames=[json.loads(x) for x in (run/"frames.jsonl").read_text().splitlines()]
queries=json.loads((run/"queries.json").read_text());ceiling=0
for q in queries:
 candidates=[x for x in frames if x["recording"]==q["recording"] and x["receipt_ns"]<=q["now_ns"] and x["motion"] is not None]
 last=max(candidates,key=lambda x:x["receipt_ns"]) if candidates else None
 ceiling+=bool(last and 0<=q["now_ns"]-last["motion"]["midpoint_ns"]<=100000000)
assert ceiling==f["zero_compute_timing_upper_bound"]==177
a=s["metrics"]["available_numeric"];e=a["midpoint"]["p95_error_mps"]
findings=f"""Completed one frozen KISS-ICP 1.3.0 full-scene scan-to-map development benchmark.
Three already-seen development recordings; 1,041 clouds received before the final fixed query; 283 fixed requests. This is not independent final confirmation.

Accuracy and freshness:
- Numeric speed output available for 283/283 requests, before any physical certification.
- At inferred interval midpoint: P95 discrepancy {e:.9f} m/s, maximum {a["midpoint"]["max_error_mps"]:.9f} m/s, zero discrepancies above 2 m/s among 281 scorable outputs; 2 unscorable.
- At actual request time: P95 discrepancy {a["query"]["p95_error_mps"]:.9f} m/s, maximum {a["query"]["max_error_mps"]:.9f} m/s; 282 scorable and 1 unscorable.
- Zero of 283 outputs meet the original 100 ms inferred-midpoint age limit after processing and queue availability. Zero timing-eligible outputs provides no positive attack-defense result.
- Even hypothetical zero compute and no queue yield only 177/283 (62.544%) timing opportunities for this fixed scan schedule and interval definition. Faster hardware alone cannot reach 95% here.
- Processing P95: turning 203.088 ms, stationary cross traffic 74.325 ms, country road 163.424 ms, on this CPU replay. These are measured processing costs, not universal runtime guarantees.

Method:
Official wheel SHA256 {pin["sha256"]}; upstream defaults with a declared four-thread CPU cap, full scene, causal prior map, per-point deskew to latest retained firing time. No INS/CAN values initialize, select, tune or enter odometry. Map resets at each recording. The shipped native deskew end convention was tested with a known translation. Original headers, firing support, receipts, poses, queue times and exclusions are retained.
Only after online estimates and query selections were frozen was INS loaded for offline interpolation. Reference brackets may not exceed 50 ms; extrapolation is forbidden.
A tiny offline epoch-to-float rounding error was corrected by integer timestamp subtraction. Original scores/source are preserved; maximum reference correction is {s["offline_precision_correction"]["maximum_reference_correction_mps"]:.12f} m/s. No online pose, query selection, measured runtime or eligibility changed.

Verification:
119 tests passed. Independent arithmetic checks cover all 1,041 frame poses/queues and all 283 query selections/scores; 9 source clouds independently re-extracted and hashed. Installed native binary matches the pinned wheel. Prior four result manifests remain unchanged.

Decision:
Keep this as a published odometry comparison and a negative timing result; do not promote it into the policy. Official numeric outputs alone provide no geometric error certificate. Bridge clock/acquisition authenticity and a physical freshness bound remain unproven. The existing research policy remains 224/283 nominal coverage (79.152%), below 95%. Reserved Urban02 NIGHT, RAV4 and vehicle_final_evaluation payloads were not opened or hashed. No manuscript or production changes.

Next engineering batch:
Test a causal higher-rate motion observer using packet-level rolling map updates, with the same full-scene prior geometry and retained firing times, a bounded/latest-available worker schedule, and explicit uncertainty/abstention. Freeze that protocol before outcomes and evaluate existing fixed requests. Preserve original thresholds; do not relabel scan starts, relax age limits, reuse CAN as an independent source or claim a hard physical guarantee from development scores. If acquisition/update-rate evidence cannot meet the targets, record the unmet requirement and narrow the project claims. Published AV-defense comparisons, novelty assessment and untouched confirmation remain required before journal-readiness can be claimed.
"""
(R/"FINDINGS.md").write_text(findings)
(R/"README.md").write_text("""Research-only full-scene scan-to-map comparison. Read FINDINGS.md and protocol_v1.json before use.
Run with D:/CO-ACT-CPS/.venv/Scripts/python.exe:
test_map_core.py or run_tests.py for tests; verify_results.py for existing-output verification.
run_benchmark.py performs a new measured replay with a fresh runtime directory and updates the summary; preserve prior outputs first. Do not rerun freeze_protocol.py or correct_offline_precision.py after completion: these are one-time provenance scripts with assertions.
The backend is KISS-ICP 1.3.0; the native wheel is pinned by backend_pin.json. Only the odometry API dependencies are installed; no CLI/visualizer claim is made. Keep license files.
runtime contains development-derived numeric artifacts and the downloaded wheel and is excluded from source control.
No physical authorization, final confirmation or production policy modification.
""")
(R/".gitignore").write_text("runtime/\n__pycache__/\n")
status=project/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_MAP_BENCHMARK.md"
assert not backup.exists();shutil.copyfile(status,backup)
old=status.read_text(encoding="utf-8")
addition=f"""
Latest verified research batch: causal full-scene KISS-ICP comparison (2026-10-21).
Completed 1,041 scans / 283 requests; 119 tests passed, independent verification passed.
Interval-midpoint P95 discrepancy {e:.6f} m/s (281 scored; 2 unscorable).
Request-time P95 discrepancy {a["query"]["p95_error_mps"]:.6f} m/s (282 scored; 1 unscorable).
Timing eligibility 0/283 under the unchanged 100 ms limit. Hypothetical zero-compute timing ceiling 177/283 (62.544%); acquisition/update granularity also blocks 95%.
Not promoted. Prior research policy remains 79.151943% coverage. Physical freshness/authenticity unproven, production disabled, final holdouts unopened. The project is not complete or journal-ready.
Next bounded engineering task: causal packet-level rolling map motion updates with bounded scheduling and explicit uncertainty, frozen before testing; do not repeat backend swaps on the same slow full-cloud schedule.
Evidence: research/vehicle_lidar_map/FINDINGS.md, benchmark_summary.json, independent_verification.json, timing_feasibility.json.
Earlier status is preserved below for history.

"""
status.write_text(addition+old,encoding="utf-8")
completion=dict(batch_complete=True,full_project_complete=False,tests_passed=119,independent_verification_passed=True,prior_manifests_unchanged=prior,native_binary_matches_pin=True,timing_ceiling_independently_checked=True,policy_changed=False,promotion_allowed=False,physical_speed_authorized=False,holdouts_accessed=False,
 runtime_files={str(p.relative_to(R)):sha(p) for p in run.iterdir() if p.is_file()},project_status_sha256=sha(status))
(R/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ("file_manifest.json","package_audit.json")}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(batch_complete=True,tests=119,frames=1041,queries=283,manifest_files=len(manifest),previous_manifests_unchanged=prior),indent=2))
