import hashlib,importlib.metadata,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;PROJECT=R.parent.parent;FULL=R.parent/"vehicle_lidar_map"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((R/"benchmark_summary.json").read_text());v=json.loads((R/"independent_verification.json").read_text());t=json.loads((R/"test_validation.json").read_text());freeze=json.loads((R/"protocol_freeze.json").read_text())
assert v["all_passed"] and all(x["exit_code"]==0 for x in t.values())
count=sum(x["count"] for x in t.values());assert count==129
assert all(sha(R/n)==h for n,h in freeze["implementation_hashes"].items())
previous={}
for folder,manifest in [(FULL,"file_manifest.json"),(R.parent/"vehicle_gicp_comparison","file_manifest.json"),(R.parent/"vehicle_lidar_stream","file_manifest.json"),(R.parent/"vehicle_radar_context","source_timing_file_manifest.json"),(R.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
 expected=json.loads((folder/manifest).read_text())
 for name,h in expected.items():assert (folder/name).parent==folder and sha(folder/name)==h,(folder,name)
 previous[folder.name+"/"+manifest]=len(expected)
pin=json.loads((FULL/"backend_pin.json").read_text())
from kiss_icp.pybind import kiss_icp_pybind
native=Path(kiss_icp_pybind.__file__)
with zipfile.ZipFile(pin["wheel"]) as z:
 names=[n for n in z.namelist() if n.endswith("/"+native.name)]
 assert len(names)==1 and hashlib.sha256(z.read(names[0])).hexdigest()==sha(native)
shutil.copyfile(FULL/"BACKEND_LICENSE.txt",R/"BACKEND_LICENSE.txt")
(R/"source_and_environment.json").write_text(json.dumps(dict(kiss_icp=pin,decoder_source=str(R.parent/"vehicle_lidar_stream"/"stream_decoder.py"),decoder_sha256=sha(R.parent/"vehicle_lidar_stream"/"stream_decoder.py"),
 calibration_sha256=sha(R.parent/"vehicle_lidar_stream"/"factory_calibration.json"),installed_native_sha256=sha(native),versions={n:importlib.metadata.version(n) for n in ["numpy","scipy","kiss-icp","small-gicp"]},new_dependencies_installed_this_batch=False),indent=2)+"\n")
m=s["metrics"];num=m["numeric"];tim=m["timing"]
text=f"""One frozen packet-window map candidate was implemented and tested on three already-seen development recordings.
2,457 jobs processed; 283 fixed evaluation requests; no parameter changes after outcomes.

Result:
Numeric motion available at 247/283 requests (87.279%). There were 246 scorable numeric outputs and one missing reference. Unguarded request-time P95 discrepancy was {num["query"]["p95_error_mps"]:.9f} m/s, maximum {num["query"]["max_error_mps"]:.9f} m/s; 92 discrepancies exceeded 2 m/s. Midpoint-time P95 was {num["midpoint"]["p95_error_mps"]:.9f} m/s. These are rejected diagnostic hypotheses, not accepted policy outputs.
Geometry passes at 0/283 request times. Joint geometry-plus-freshness eligibility is 0/283. Timing alone passes 1/283 requests under both midpoint support checks. No error or security success can be inferred from zero joint accepts.
Across all processing jobs, 10 geometry passes occurred in the stationary control, none at fixed request times; no moving-record geometry pass occurred. Frequent rejected matches triggered 466 state resets. Conservative guards prevented map updates from rejected poses. Residual failures alone do not identify a unique cause; 92 large numeric errors demonstrate that acceptance cannot safely be based on numerical availability alone.

Implementation:
Bootstrap uses up to 100 ms of raw firing observations, then a minimum 20 ms job cadence with latest 20 ms firing windows. One serialized worker includes raw deserialization/decoding, selection, motion scaling, deskew, registration, quality checks and map/state update. It processes the newest pending support when free, stores only a bounded 220 ms raw pending buffer, and never decodes expired work retroactively. Bag disk I/O and audit artifact writes are excluded from the replay timing model.
Matching uses KISS-ICP 1.3.0 components with the prior default configuration and four CPU threads, plus custom variable-interval SE3 motion scaling, guarded map updates and the bounded scheduler. This is a research extension using a published registration core, not reproduction of an AV security method or the complete official odometry pipeline.
Original raw bridge headers, firing offsets, packet hashes/receipts, sampled geometry, previous accepted states and job times are retained. Current matching uses the earlier map before any current update. INS/CAN values never enter online estimation. INS was loaded only after online job/query outputs were saved; integer timestamp interpolation forbids extrapolation and brackets over 50 ms.
The fixed local geometry checks use at least 150 points, 100 correspondences within 1.5 m, overlap at least half, untrimmed inlier Euclidean P95 at most 0.3 m and fixed-correspondence Jacobian condition at most 1000. These checks are neither calibrated uncertainty estimates nor hard physical error certificates. Numeric intervals must be 5 to 200 ms with speed at most 60 m/s. Query eligibility requires completed work and both endpoint and first-firing support midpoints no older than 100 ms.
The chosen pending-work bound removes the former unbounded queue, but it does not guarantee timely valid motion. P95 job processing costs were 104.142 ms turning, 90.489 ms stationary control and 81.941 ms country road on this CPU replay. The processing and freshness targets still fail.

Verification:
129 tests passed. An independent checker verified all 2,457 jobs' causal scheduling, accepted-state history, pose/velocity arithmetic, all 283 query selections and offline scores, 29 frozen input snapshots and 24 geometry snapshots. Nine sampled source windows were redecoded from the bags with the previously validated decoder. The installed native binary matches the pinned wheel. All five prior result manifests remain unchanged. This does not verify every raw payload independently.
No reserved Urban02 NIGHT, RAV4 or vehicle_final_evaluation payload was opened, hashed or scored. No manuscript, deployment, commit, push or production policy change occurred.

Decision:
Candidate fails; do not use it as an independent security witness. Existing research policy coverage remains 224/283 (79.152%), below the original 95% target. Physical acquisition authenticity, clock bounds and a physical freshness/error certificate remain unestablished. The full project is not complete or journal-ready.

Next evidence requirement:
Establish a defensible motion-quality and timing model before adding another estimator variant. Use controlled static/moving scene diagnostics and the training/calibration portions to distinguish map sampling, motion compensation, scene motion and correspondence ambiguity. A replacement quality model must be calibrated independently of the fixed evaluation outcomes, with explicit abstention and uncertainty; a fine voxel or relaxed residual threshold alone is not supporting evidence.
Proceed to a frozen policy/attack comparison only when a candidate jointly meets fresh-data coverage and accuracy requirements. Preserve the untouched confirmation data until that decision. Published AV-defense comparisons and novelty remain unfinished. More backend swaps on these same intervals are not the next task.
"""
(R/"FINDINGS.md").write_text(text)
(R/"README.md").write_text("""Research-only latest-window packet map experiment. Start with FINDINGS.md and protocol_v1.json.
All files belong only to D:/CO-ACT-CPS/phase2_joint_history/research/vehicle_packet_map.
Use D:/CO-ACT-CPS/.venv/Scripts/python.exe with run_tests.py for regression checks or verify_packet_results.py for saved-result verification.
run_packet_benchmark.py creates a fresh timed runtime and updates the summary; preserve existing results before a new replay. freeze_protocol.py and finalize_package.py are one-time provenance steps with overwrite-preventing assertions.
The runtime folder contains development-derived numeric evidence. Keep it out of git. Source and environment pins are recorded in source_and_environment.json; keep upstream license.
Zero joint accepts is a failed availability result and provides no successful defense evidence.
""")
(R/".gitignore").write_text("runtime/\n__pycache__/\n")
status=PROJECT/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_PACKET_BENCHMARK.md"
assert not backup.exists();shutil.copyfile(status,backup);old=status.read_text(encoding="utf-8")
status.write_text(f"""Latest verified research batch: bounded packet-window map observer (2026-10-21).
Completed 2,457 causal jobs and 283 fixed requests. All 129 regression tests and independent result verification passed.
Candidate failed: 0/283 geometry accepts and 0/283 joint geometry/freshness eligible requests. Only 1/283 passes the timing screen alone.
Unguarded numeric hypotheses: 247 outputs, 246 scorable, P95 request-time discrepancy {num["query"]["p95_error_mps"]:.6f} m/s and 92 errors over 2 m/s. These never enter the policy.
The guarded state resets frequently after rejected matches. No moving-record geometry accept occurred. Bounded scheduling prevents the prior long work queue but does not meet the validity or freshness target.
No promotion. Existing research policy remains 224/283 nominal coverage (79.151943%), below 95%. Physical freshness/error/authenticity remain uncertified. Full project complete: no.
Next evidence requirement: controlled geometry/motion/timestamp diagnosis and independently calibrated uncertainty with abstention, using training/calibration data. Stop ungrounded backend/window variants. Freeze policy/attack evaluation only after a witness meets joint targets; reserve final confirmation until then.
Evidence: research/vehicle_packet_map/FINDINGS.md, benchmark_summary.json, rejection_diagnosis.json, independent_verification.json.
Previous status is retained below as history.

"""+old,encoding="utf-8")
run=R/s["runtime_relative_path"]
completion=dict(batch_complete=True,full_project_complete=False,tests_passed=count,independent_verification_passed=True,
 jobs_verified=v["jobs_checked"],queries_verified=283,candidate_passed=False,prior_manifests_unchanged=previous,
 installed_native_matches_pin=True,policy_changed=False,promotion_allowed=False,physical_speed_authorized=False,holdouts_accessed=False,
 runtime_files={str(p.relative_to(R)):sha(p) for p in run.iterdir() if p.is_file()},project_status_sha256=sha(status))
(R/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ("file_manifest.json","package_audit.json")}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(batch_complete=True,tests_passed=count,jobs=2457,candidate_passed=False,prior_manifests_unchanged=previous,manifest_files=len(manifest)),indent=2))
