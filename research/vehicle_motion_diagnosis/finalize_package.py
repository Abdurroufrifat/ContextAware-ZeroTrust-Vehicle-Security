import hashlib,importlib.metadata,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;PROJECT=R.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((R/"diagnosis_summary.json").read_text());surface=json.loads((R/"surface_summary.json").read_text());v=json.loads((R/"independent_verification.json").read_text());tests=json.loads((R/"test_validation.json").read_text())
assert v["all_passed"] and all(x["exit_code"]==0 for x in tests.values())
count=sum(x["count"] for x in tests.values());assert count==136
previous={}
for folder,manifest in [(R.parent/"vehicle_packet_map","file_manifest.json"),(R.parent/"vehicle_lidar_map","file_manifest.json"),(R.parent/"vehicle_gicp_comparison","file_manifest.json"),(R.parent/"vehicle_lidar_stream","file_manifest.json"),(R.parent/"vehicle_radar_context","source_timing_file_manifest.json"),(R.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
 expected=json.loads((folder/manifest).read_text())
 for name,h in expected.items():assert (folder/name).parent==folder and sha(folder/name)==h,(folder,name)
 previous[folder.name+"/"+manifest]=len(expected)
decision=dict(batch_complete=True,full_project_complete=False,
 downsampling_alone_causes_true_pose_rejection="not supported: zero of eighteen exact-identity anchors fail the fixed residual screen",
 registration_core_cannot_recover_known_motion="not supported in these same-parent controls; not a generalization result",
 fixed_global_residual_is_a_calibrated_motion_error_bound="not established; all six separate stationary pairs fail residual screening despite speed discrepancy below0.264m/s",
 point_plane_residual_is_an_immediate_safe_replacement="not established; temporal median normal P95 remains0.527m and single-plane synthetic geometry is rank deficient",
 explanation_of_92_large_moving_numeric_errors="unresolved by same-parent and near-stationary controls; do not extrapolate these diagnostic results to moving vehicles",
 uncertainty_calibration_status="not established; correlated controls and six stationary temporal pairs are insufficient",
 next_requirement="train/calibrate quality and abstention using separate temporal moving observations, persistent scene support, ambiguity/observability and motion compensation; freeze before any renewed evaluation; retain original age/error/coverage targets",
 observer_changed=False,production_policy_changed=False,promotion_allowed=False)
(R/"DIAGNOSIS_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n")
text=f"""Completed a controlled training/calibration diagnosis inside CO-ACT-CPS, without changing the observer or policy.

Initial fixed protocol:
Selected the 25%, 50% and 75% metadata indices separately in the training and calibration portions of each of the three allowed development recordings. Eighteen unique anchors were used. Raw payloads were decoded only in each anchor's preceding 202 ms receipt support, stopping before evaluation intervals. Original bridge headers, firing offsets, receipt times and payload hashes were preserved.
Each anchor provides a full 100 ms parent observation and its latest 20 ms exact subset, plus an earlier nonoverlapping 100 ms observation. Same-parent comparisons deliberately share points: they isolate geometry/registration behavior and cannot serve as independent online evidence or performance confirmation.

What the controls show:
1. Downsampling alone did not reproduce the failed residual screen. At exact identity, all eighteen anchors passed the unchanged 0.3 m screen after default voxel-map insertion. Median map residual P95 was {s["identity_floor_by_stage"]["default_map"]["median_p95_m"]:.9f} m; maximum was {s["identity_floor_by_stage"]["default_map"]["maximum_p95_m"]:.9f} m. The raw-parent and source-preserving controls had zero P95 residual.
2. The official registration core recovered prescribed rigid motions on these controls. Across 54 default-map known-pose fits, the translation error divided by a hypothetical 20 ms interval had P95 {s["known_pose_fits"]["default_map"]["translation_equivalent_p95_mps"]:.9f} m/s and maximum {s["known_pose_fits"]["default_map"]["translation_equivalent_maximum_mps"]:.9f} m/s. The 54 source-preserving controls had P95 {s["known_pose_fits"]["source_preserving_map"]["translation_equivalent_p95_mps"]:.9f} m/s. No translation-equivalent error exceeded 2 m/s. These are synthetic transform metrics on eighteen underlying anchors, not independent vehicle-speed observations.
3. Across six separate near-stationary temporal pairs, every fitted speed discrepancy was below 0.5 m/s; maximum {surface["maximum_speed_discrepancy_mps"]:.9f} m/s. Nevertheless, all six failed the Euclidean residual screen. Their median point-distance P95 was {surface["euclidean_p95_median_m"]:.9f} m. This documents false abstention in these stationary controls; it does not justify accepting the previously unsafe moving estimates.

Follow-up diagnostic:
After the initial controls, a separate protocol was frozen before examining temporal surface metrics. The timing of this follow-up decision is explicit in surface_protocol_v1.json; it was not part of the initial prespecified hypothesis.
Using 16-neighbor PCA normals on the six earlier-scan targets, median point-to-plane residual P95 was {surface["plane_p95_median_m"]:.9f} m. It therefore does not establish a simple safe replacement for the 0.3 m Euclidean screen. All six local fitted Jacobians had rank six, while a separate single-plane synthetic test correctly has rank below six. Local rank is not a global correspondence or motion-error certificate.
Possible sampling changes, dynamic objects, correspondence ambiguity, surface-estimation noise and initial motion compensation are not isolated from one another by these results. The 92 large unguarded moving errors in the previous packet experiment remain unexplained by these controls. No unique physical root cause is claimed.

Verification:
All 136 relevant tests passed. The independent checker verified eighteen anchor point sets and their roles/timing, 108 known-pose fits, six temporal fits and six surface comparisons, reference interpolation and aggregate arithmetic. Every anchor's source support was reread from the original bags and redecoded with the previously validated decoder.
Source verification initially exposed an audit assumption: several packets share the same bag receipt timestamp. The audit was corrected to identify each packet by receipt, original header and payload SHA256. Original extraction already retained every packet in order, so no experiment input, pose or score changed.
All six prior result manifests remain unchanged. No evaluation raw payload was decoded by this diagnostic run, and no reserved Urban02 NIGHT, RAV4 or vehicle_final_evaluation payload was opened, hashed or scored.

Decision:
Diagnosis batch complete; project incomplete. Neither a calibrated uncertainty model nor a safe fresh speed witness has been established. Existing research policy nominal coverage remains 224/283 (79.152%), below 95%. Physical acquisition authenticity/clock/error bounds remain unproved. No observer, policy, dependency, manuscript, deployment, commit or push changed in this batch.

Next implementation requirement:
Use separate temporal training/calibration observations across moving conditions to assess persistent scene support, ambiguity and motion compensation. A quality/abstention model must be calibrated against motion discrepancy, with explicit uncertainty limitations. Freeze that model before renewed evaluation and preserve final confirmation data. Do not relax age limits, accept low residual alone, switch to surface distance alone or present same-parent controls as independent vehicle-security evidence. Current controls are insufficient to calibrate generalization uncertainty.
"""
(R/"FINDINGS.md").write_text(text)
(R/"README.md").write_text("""Read FINDINGS.md and DIAGNOSIS_DECISION.json for the result and its limits.
This folder contains diagnostic controls only; it does not modify an observer or authorize a policy.
Use D:/CO-ACT-CPS/.venv/Scripts/python.exe with run_tests.py for regression tests or verify_diagnosis.py for saved-result verification.
run_diagnosis.py, freeze_protocol.py, surface_diagnosis.py and finalize_package.py are one-time provenance runs with overwrite-preventing assertions. Preserve evidence before starting a separate diagnostic version.
runtime contains development-derived numeric arrays and is excluded from git.
Same-parent controls intentionally share source/target observations; they must never be used to claim independent online accuracy, speed freshness or attack resistance.
""")
(R/".gitignore").write_text("runtime/\n__pycache__/\n")
source=R.parent/"vehicle_lidar_stream"/"stream_decoder.py";cal=R.parent/"vehicle_lidar_stream"/"factory_calibration.json"
(R/"source_and_environment.json").write_text(json.dumps(dict(decoder_sha256=sha(source),calibration_sha256=sha(cal),
 versions={n:importlib.metadata.version(n) for n in ["numpy","scipy","kiss-icp","small-gicp"]},dependencies_changed=False),indent=2)+"\n")
status=PROJECT/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_DIAGNOSIS.md"
assert not backup.exists();shutil.copyfile(status,backup);old=status.read_text(encoding="utf-8")
status.write_text(f"""Latest verified research batch: training/calibration motion-quality diagnosis (2026-10-21).
Completed eighteen real-data anchors, 108 synthetic known-pose fits, six separate near-stationary temporal fits and six surface diagnostics. All 136 regression tests and independent verification passed; all eighteen source supports redecoded.
Downsampling alone did not reproduce the failed gate: all eighteen exact-pose default-map controls pass the0.3m residual screen. The core recovers imposed motions on shared-source controls.
Separate stationary temporal estimates have maximum speed discrepancy {surface["maximum_speed_discrepancy_mps"]:.6f}m/s but all fail the distance screen. Median Euclidean P95 is {surface["euclidean_p95_median_m"]:.6f}m; median surface-normal P95 is {surface["plane_p95_median_m"]:.6f}m. A switch to point-to-plane alone is unsupported.
The global residual screen is not a calibrated motion-error bound. The previous92 large moving errors remain unresolved; same-parent/static controls cannot establish uncertainty or moving-vehicle performance.
No observer/policy promotion. Prior research coverage remains79.151943%, below95%; physical freshness/error/authenticity remain uncertified. Full project complete:no.
Next requirement: temporal moving training/calibration study of persistent scene support, correspondence ambiguity and compensation; calibrate uncertainty and abstention before freezing renewed evaluation. Retain original targets and preserve final confirmation payloads.
Evidence: research/vehicle_motion_diagnosis/FINDINGS.md, DIAGNOSIS_DECISION.json, diagnosis_summary.json, surface_summary.json, independent_verification.json.
Earlier status retained below as history.

"""+old,encoding="utf-8")
run=R/s["runtime_relative_path"]
completion=dict(batch_complete=True,full_project_complete=False,tests_passed=count,independent_verification_passed=True,
 training_calibration_anchors_checked=18,known_pose_fits_checked=108,temporal_fits_checked=6,
 source_anchors_redecoded=18,prior_manifests_unchanged=previous,uncertainty_calibrated=False,observer_changed=False,policy_changed=False,
 evaluation_payloads_decoded=False,protected_holdouts_accessed=False,promotion_allowed=False,
 runtime_files={str(p.relative_to(R)):sha(p) for p in run.iterdir() if p.is_file()},project_status_sha256=sha(status))
(R/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ("file_manifest.json","package_audit.json")}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(batch_complete=True,tests_passed=count,independent_verification_passed=True,prior_manifests_unchanged=previous,manifest_files=len(manifest),uncertainty_calibrated=False),indent=2))
