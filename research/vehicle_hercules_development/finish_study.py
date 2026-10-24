from pathlib import Path
import json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parent;P=R.parent.parent
summary=json.loads((R/"development_summary.json").read_text())
validation=json.loads((R/"independent_verification.json").read_text());assert validation["passed"]
pairs=json.loads((R/"paired_comparison.json").read_text())
for repo in ["HeRCULES_file_player","HeRCULES_Pointcloud_Toolbox"]:
 lock=json.loads((R/(repo+"_source_lock.json")).read_text());zpath=R/(repo+"_source.zip")
 assert hashlib.sha256(zpath.read_bytes()).hexdigest()==lock["sha256"]
 with zipfile.ZipFile(zpath) as z:
  for name in z.namelist():
   if name.endswith("/"):continue
   rel=Path(*Path(name).parts[1:])
   if rel.suffix.lower() not in [".py",".cpp",".h",".hpp",".md",".txt",".xml",""]:continue
   assert (R/"official_source"/repo/rel).read_bytes()==z.read(name)
rows=[];frame_rows=[];cohort_rows=[];segments=[]
for name,s in summary["sequences"].items():
 r=s["requests"];a=s["candidate_all_frames_primary"];b=s["candidate_all_frames_INS_secondary"];c=pairs[name]
 rows.append(f'| {name} | {s["frames"]:,} | {r["count"]:,} | {100*r["ready_fraction"]:.3f}% | {r["scored"]:,} | {r["p95_mps"]:.6f} | {r["max_mps"]:.6f} | {r["above_2_mps"]} |')
 frame_rows.append(f'- {name}: same-radar-pose interval reference P95 {a["p95_mps"]:.6f} m/s ({a["scored"]:,} scored); direct INS-speed P95 {b["p95_mps"]:.6f} m/s ({b["scored"]:,} scored).')
 cohort_rows.append(f'| {name} | {c["paired_scored_frames"]} | {c["candidate"]["p95_mps"]:.6f} | {c["author_3d_ransac"]["p95_mps"]:.6f} | {c["ols_3d"]["p95_mps"]:.6f} |')
 segments.extend(r["segments_60s"])
segment_passes=sum(s["ready_fraction"]>=.95 and s["p95_mps"] is not None and s["p95_mps"]<=.5 and s["above_2_mps"]==0 for s in segments)
total_frames=sum(s["frames"] for s in summary["sequences"].values())
total_requests=sum(s["requests"]["count"] for s in summary["sequences"].values())
scored=sum(s["requests"]["scored"] for s in summary["sequences"].values())
points=sum(s["points"] for s in summary["sequences"].values())
downloads=json.loads((R/"download_lock.json").read_text())
readiness={"scope":"HeRCULES two-sequence development only","download_access_resolved":True,"native_radar_INS_and_Xsens_available":True,"candidate_frames":total_frames,"requests":total_requests,"scored_requests":scored,"ready_requests_without_reference":total_requests-scored,"sequences":{n:s["numeric_request_gates"] for n,s in summary["sequences"].items()},"physical_acquisition_age_bound_ns":None,"reference_velocity_error_bound_mps":None,"sensor_authenticity_established":False,"new_candidate_attack_validation_complete":False,"learned_published_method_reproduction_complete":False,"novelty_established":False,"independent_final_confirmation_complete":False,"Library03_payloads_accessed":False,"production_changed":False,"overall_project_complete":False}
(R/"candidate_readiness.json").write_text(json.dumps(readiness,indent=2),encoding="utf-8")
body=f"""# HeRCULES development experiment

Completed2026-10-21. The supplied links resolved the download-access blocker. Acquired14 exact files for Library01/02, totaling{downloads["total_bytes"]:,}bytes. Camera, LiDAR and spinning-radar payloads were omitted. Library03 folder metadata was inventoried, but its numeric measurement/reference payloads remain undownloaded, unopened and unscored.

The fixed radar/Xsens candidate was run on{total_frames:,}frames and{points:,}radar returns. It estimates planar velocity from raw radial Doppler with standard Huber IRLS after leveling sight directions using independent Xsens orientation. No INS/reference labels enter the predictor; no score-dependent fitting, sign change, timestamp offset or threshold adjustment was made. Range3..100m, Huber0.3m/s,20iterations and all quality limits were frozen before outcomes. This is a standard estimator adaptation and development result, not a demonstrated new algorithm.

## Scheduled20Hz development replay

| Sequence | Radar frames | Requests | Timestamp-plus-compute ready | Reference-scored requests | P95 disagreement(m/s) | Maximum(m/s) | Scored>2m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+f"""

Retained60-second segments meeting >=95% replay availability, P95<=0.5m/s and zero scored discrepancies>2m/s: {segment_passes}/{len(segments)}. Every segment remains in development_summary.json. Four endpoint requests have no primary reference score. Their correctness is unknown; they were not counted as known-safe or removed from the availability denominator.

The primary reference is a central secant of published same-radar-point INS-derived pose positions, with offline interpolation to scheduled query times. It measures disagreement with interval motion, not a certified instantaneous velocity error. Secondary direct INS-speed comparisons are also retained:
"""+'\n'.join(frame_rows)+"""

These are two representations of the publisher's INS/navigation reference, not two independent ground-truth systems. The INS reference location and radar location differ; Xsens calibration is not an INS lever-arm certificate.

## Same-cohort classical comparison

Every20th chronological frame was predeclared; the table uses the identical finite reference/method intersection, with all original cohort counts retained in paired_comparison.json and development_summary.json.

| Sequence | Paired frames | Candidate P95(m/s) | Author3D RANSAC P95(m/s) | Ordinary3D LS P95(m/s) |
|---|---:|---:|---:|---:|
"""+'\n'.join(cohort_rows)+"""

The author RANSAC implementation is from paaraujo/deep-ego-velocity commit2285343bb1e433e3b7dfbc8e200eb969805388c0. Adaptation: native HeRCULES Cartesian unit directions supply azimuth/elevation, identity rotation, native300trials/0.1threshold/seed42. This is a classical-method implementation comparison, not reproduction of the paper's deep network or its original test-set protocol. The candidate has an additional IMU orientation input and a planar-motion assumption; this is not a like-for-like sensor-budget or statistically established superiority claim. Both sequences cover the same Library route; generalization and novelty remain unestablished.

## Timing, verification and limits

The first replay was interrupted before completed sequence scores because random access in unsorted compressed archives repeatedly decompressed the source. An instrumented probe measured1.78s for30timestamp-ordered reads versus0.34s for streaming all6,302files. The revised bounded loader reads each archive once. Its first30raw member hashes match the original loader, and the original estimator core identity is preserved. No estimator/reference/guard was changed. Original files, lock and performance evidence remain.

Final candidate timing ran after core verification completed. Replay assumes publisher timestamps are arrival times and simulates serial service using measured decode/IMU/fit time. This is conditional offline timing, not measured hardware acquisition-to-enforcement latency. Unknown source clock/buffering, measurement authenticity and physical reference error bounds stay unknown.

731core tests pass,4skip, one existing legacy residual-calibration warning. The28focused tests include14new HeRCULES tests and14previous decoder/compatibility tests. Six full real frames were cross-decoded against the official Python struct reader; geometry was checked on those frames. Independent arithmetic replay verifies all scheduled ready decisions, metrics and segments;14downloaded inputs were rehashed. Official source archives/extracted code were checked for identity. No reserved old payload was accessed.

The development source now supports further experiments. Physical-witness admission and overall project completion remain false. Next work: freeze and evaluate the new candidate's source-compromise, replay/staleness, corrupted-context and claim-tampering cases on development data; define uncertainty and trust assumptions; complete compatible published-method/novelty work. Only then decide whether the frozen protocol warrants opening Library03 as a separate confirmation. A few good replay metrics do not establish attack resistance or journal readiness.

No manuscript, vehicle deployment, commit or GitHub push. Production policy unchanged.

## Provenance

Dataset: HeRCULES, Kim et al.,ICRA2025, DOI10.1109/ICRA55743.2025.11128678.
Data license: CC BY-NC-SA4.0, official notice https://sites.google.com/view/herculesdataset/contact.
Format and reference: https://sites.google.com/view/herculesdataset/system.
Official parser sources: https://github.com/hanjun815/HeRCULES_Pointcloud_Toolbox and https://github.com/hanjun815/HeRCULES_file_player.
Classical implementation: https://github.com/paaraujo/deep-ego-velocity, MIT source license retained in the previous source-qualification module.
"""
(R/"FINDINGS.md").write_text(body,encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_HERCULES.md";assert not backup.exists();shutil.copyfile(status,backup)
prefix=f"""Latest: HeRCULES radar/Xsens development experiment completed(2026-10-21).

Dataset access resolved: user supplied Library01/02/03 official links. Downloaded only14needed development files for Library01/02. Library03 measurement payloads remain unopened and undownloaded.
Fixed candidate evaluated on{total_frames:,}frames,{total_requests:,}20Hz requests. Ready fraction100%in both sequences under nominal timestamp-plus-measured-compute replay. Request P95 primary reference disagreement: Library01=.295711m/s; Library02=.240059m/s. Zero scored requests>2m/s; four endpoint requests have no reference score. Retained60s segments passing the three reported numeric criteria: {segment_passes}/{len(segments)}. No threshold tuning.
Reference is interval velocity derived from same-radar INS-based poses, with direct INS speeds retained as secondary. Their common navigation ancestry, unknown physical clock/error bounds and unknown source authenticity limit claims. This is not certified physical error or security acceptance.
Same-cohort classical author3D RANSAC and ordinary3D LS comparisons are implemented; candidate gains remain descriptive, with extra IMU input. Deep-model reproduction, novelty and new-candidate attack validation remain unfinished.
731core tests pass,4skip, one existing calibration warning.28focused tests pass. Fourteen input hashes, six official-reader frame checks and independent metric/timing replay verified.
Full project complete:NO. New candidate not promoted; production unchanged. Reserved prior payloads and Library03 remain excluded. No manuscript/deployment/push.
Next: evaluate this frozen candidate against development attack and corruption cases, justify uncertainty/trust assumptions, then complete comparison/novelty work before separate final confirmation.
Evidence: research/vehicle_hercules_development/FINDINGS.md, development_summary.json, paired_comparison.json, independent_verification.json, candidate_readiness.json.
Prior verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"module_files":len(manifest),"frames":total_frames,"points":points,"requests":total_requests,"scored_requests":scored,"source_files":14,"core_passed":731,"core_skipped":4,"focused_passed":28,"physical_witness_admitted":False,"Library03_payloads_accessed":False,"current_status_sha256":hashlib.sha256(status.read_bytes()).hexdigest()}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(completion,indent=2))
