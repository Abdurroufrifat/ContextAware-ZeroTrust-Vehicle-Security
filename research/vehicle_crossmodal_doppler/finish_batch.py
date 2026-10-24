from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent
v=json.loads((R/"independent_verification.json").read_text());assert v["passed"]
s=json.loads((R/"prefix_summary.json").read_text())["sequences"]
b=json.loads((R/"ACQUISITION_BLOCKER.json").read_text());assert b["status"]=="blocked_by_publisher_quota"
table="\n".join(f'| {seq} | {x["optical_frames"]} | {x["requests"]} | {x["accepted"]} | {100*x["availability"]:.2f}% | {x["primary"]["p95_mps"]:.6f} |' for seq,x in s.items())
attacks="\n".join(f'| {case} | {x["supported"]} | {x["accepted"]} | {x["primary_above2"]} |' for case,x in v["cases"].items())
text="""# Conditional radar/optical Doppler cross-check: development evidence

2026-10-21. This bounded experiment is complete; the full-sequence study and overall project are incomplete.

## Implemented behavior

An independent optical Doppler estimate from Aeva FMCW LiDAR now checks the existing Continental radar/IMU speed estimate. Decode the official packed29byte schema, sample512uniform original records, fit3Dstatic-scene Doppler using the frozen Huber estimator, and apply the provided rigid transform and gyro lever-arm correction to the radar origin. Evaluation INS/GPS/pose labels never enter the optical predictor. The output remains the original radar estimate, conditionally admitted only when both source qualities and claimed ages pass and the scalar speeds agree within1m/s.

This is offline development research. No production policy or physical witness was enabled. Separate sensor modalities do not establish source authenticity, physical clock correctness, independent failure probabilities or absolute error bounds. Shared Xsens gyro/context remains a dependency. The check concerns scalar speed; equal-speed heading attacks are outside this result.

Official primary sources: https://sites.google.com/view/herculesdataset/system and https://arxiv.org/html/2502.01946v1 ; official reader retained in the prior HeRCULES module. Raw data remain local under the provided CC BY-NC-SA4.0 terms.

## Source admission and acquisition limit

The predeclared128frame pilot passed128/128fixed optical-quality and1m/s agreement checks. Timestamp-only full-sequence freshness ceilings were99.73%for Library01 and97.09%for Library02 before processing. These are ceilings, not measured continuous availability.

The subsequent two full archive downloads were blocked by the publisher. The expected HTTP206binary response was replaced by HTTP200HTML titled "Google Drive - Quota exceeded". Incorrect responses were rejected before append. Stop quota retries. Retain402,653,184and268,435,456validated compressed bytes. Neither is a complete archive. PREFIX_STUDY_AMENDMENT.md was frozen before estimating the extended prefixes.

Use all385complete frames in those prefixes, including their unordered timestamp coverage. There was no error-based member selection. The original20Hz query grid was restricted only to each prefix's minimum/maximum complete-member timestamps; every internal missing interval stayed in the denominator. Thus sparse archive order cannot be hidden by evaluating only successful frames.

## Limited-prefix results

| Sequence | Optical frames | Fixed interval requests | Admitted requests | Availability | Admitted primary-reference P95(m/s) |
|---|---:|---:|---:|---:|---:|
"""+table+"""

All385optical fits passed the frozen quality screen. Combined availability fails the unchanged95%requirement on both partial intervals. All716admitted requests have primary reference scores and zero recorded discrepancies above2m/s. This does not demonstrate certified physical safety, whole-sequence performance or the95%availability requirement. Secondary INS scores are missing for26Library01 and12Library02admitted requests; these remain unknown.

Measured binary read/decode/fit P95was16.16ms/16.29ms on this PC. That includes reading the decompressed member bytes and fit, but is not a measured full sensor/network publication stack. Replay service is serial on nominal dataset publisher timestamps. Physical arrival/clock bounds remain unverified.

## Fixed attack replay

| Scalar-interface case | Supported query pairs | Admitted outputs | Admitted recorded primary discrepancy>2m/s |
|---|---:|---:|---:|
"""+attacks+"""

The two single-modality+3m/s cases are rejected on all716supported pairs each. Simultaneous+3m/s shifts pass all716and produce716recorded>2m/s discrepancies. Both outcomes must be reported together. They establish the tested conditional benefit and its common-mode limitation; they do not establish resistance to arbitrary attacks.

Honest-timestamp1second replays are expired for both modalities wherever prior pairs exist. Forged-current timestamps pass129/139radar and132/139optical cases. No recorded primary discrepancy exceeds2m/s among those admitted cases, but their acceptance explicitly shows that the cross-check does not authenticate timestamps. Missing/stale optical context causes abstention.

These are scalar-interface propagation tests, not complete raw-stream attack reproductions. Separately, six first/middle/last recorded optical frames were altered in their encoded float32Doppler records. Coherent velocity shifts retained optical quality and changed inferred speed by3m/s within1e-6. This verifies the raw optical estimator's own coherent-forgery limitation. Six official-reader field checks on these prefix frames and six on the initial pilot validated the parser.

## Verification and retained failures

731configured core tests pass,4skip; one existing legacy calibration warning remains.55focused tests pass, including20new optical/replay tests and35previous source/decoder tests. Independent arithmetic replay checked all6490interval requests, all ten attack cases, per-segment metrics, hash locks, pilot-prefix membership and unchanged original inputs/results.

Original four primary endpoint unknowns, the Library01last302request failure(P951.1022975m/s) and older failures are retained. This partial study does not replace or repair them. No reference swap, test-data tuning, Library03 payload access, manuscript, installation, commit, push or production change occurred.

## What remains

Normal access to the complete Library01/02optical archives is the immediate acquisition blocker. Saved resume_full_study.py will resume into separate working files, preserving these immutable prefix artifacts, then execute the full replay and its independent arithmetic audit. That full path is saved and syntax-checked; it has not been run end-to-end because the complete archives are unavailable. It does not automatically certify physical safety or project completion.

Full continuous availability, retained-segment performance, reference/clock/source-trust justification, compatible published-method comparisons and novelty assessment remain necessary. Library03 stays reserved until the model/protocol and development decisions are fixed. Current results cannot support a claim of journal readiness or complete source-compromise resistance.
"""
# Derive secondary unknown counts from source rather than silently guessing.
for seq,x in s.items():
 actual=x["secondary_INS"]["unknown_reference"]
 text=text.replace("12Library02admitted",str(actual)+"Library02admitted") if seq=="Library02" else text
(R/"FINDINGS.md").write_text(text,encoding="utf-8")
decision={"batch_scope":"independent optical-Doppler adapter and bounded partial-download development study","bounded_batch_complete":True,"full_sequence_study_complete":False,"overall_project_complete":False,"pilot_frames":128,"extended_optical_frames":v["optical_frames"],"supported_nominal_pairs":v["nominal_paired_requests"],"single_modality_plus3_rejected_each":716,"both_modalities_plus3_admitted":716,"availability_gates_passed":False,"blocker":"Google Drive publisher quota exceeded","reference_failure_retained":True,"original_unknown_endpoints":4,"source_authentication_established":False,"clock_bound_established":False,"physical_witness_admitted":False,"production_changed":False,"Library03_payloads_accessed":False,"full_resume_end_to_end_executed":False}
(R/"RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_CROSSMODAL.md"
if not backup.exists():shutil.copyfile(status,backup)
prefix="""Latest: conditional radar/optical Doppler cross-check development batch,2026-10-21.

Implemented independent Aeva optical radial-velocity estimator and fixed1m/s cross-sensor agreement guard.128/128pilot frames passed. Extended partial-download study:385optical frames,716nominal admitted query pairs. Single radar-only/optical-only+3m/s scalar alterations rejected716/716each; simultaneous+3shifts admitted716/716, all with recorded>2m/s primary discrepancy. Forged timestamps remain an explicit limitation. No arbitrary source-compromise guarantee.
Publisher Google Drive quota blocked full archives. Partial-interval availability30.51%/4.87%fails95%; admitted primary-reference P95discrepancies0.268180/0.245343m/s. Preserve missing intervals, unknown secondary scores and all old failed segments. Full continuous study has not run; saved full resume/acquisition/replay/audit tools are syntax-checked but not end-to-end validated.
731core tests pass,4skip, one existing legacy warning;55focused tests pass. Independent6490request arithmetic/provenance audit passes.
Overall project complete:NO. Production witness disabled. Reference/physical clock/authenticity evidence, full development coverage, comparisons/novelty and later final confirmation remain incomplete. Library03 and older reserved payloads untouched.
Evidence: research/vehicle_crossmodal_doppler/FINDINGS.md, prefix_summary.json, RESEARCH_DECISION.json, ACQUISITION_BLOCKER.json and independent_verification.json.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
# Immutable explicit local artifact manifest; raw payloads are not redistributed.
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
files={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"full_sequence_study_complete":False,"module_files":len(files),"optical_frames":385,"paired_requests":716,"core_passed":731,"core_skipped":4,"focused_passed":55,"acquisition_blocked_by_publisher_quota":True,"current_status_sha256":sha(status)}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(completion,indent=2))
