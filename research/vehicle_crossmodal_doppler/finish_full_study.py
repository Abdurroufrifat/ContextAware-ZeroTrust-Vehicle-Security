"""Record the complete development study without replacing earlier partial/failed evidence."""
from pathlib import Path
import hashlib,json,shutil,numpy as np
R=Path(__file__).resolve().parent;P=R.parent.parent
proof=json.loads((R/"full_independent_verification.json").read_text());assert proof["passed"] and proof["full_sequence_study_complete"]
summary=json.loads((R/"full_summary.json").read_text());s=summary["sequences"]
manual=json.loads((R/"manual_import_audit.json").read_text());assert set(manual["sequences"])=={"Library01","Library02"}
table=[];segments=[];availability={}
for seq,x in s.items():
 a=np.load(R/(seq+"_crossmodal_replay.npz"),allow_pickle=False);i=a["optical_index"];safe=np.maximum(i,0)
 reason=np.full(len(i),"admitted",dtype="<U32")
 reason[~a["accepted"]]="agreement_or_radar_context"
 reason[(i>=0)&~a["quality"][safe]]="invalid_optical_quality"
 reason[(i>=0)&(a["optical_age_ns"]>100_000_000)]="stale_optical"
 reason[i<0]="no_completed_optical"
 availability[seq]={str(k):int(v) for k,v in zip(*np.unique(reason,return_counts=True))}
 table.append(f'| {seq} | {x["optical_frames"]} | {x["requests"]} | {x["accepted"]} | {100*x["availability"]:.2f}% | {x["primary"]["p95_mps"]:.6f} | {x["primary"]["above2"]} | {x["primary"]["unknown_reference"]} |')
 for seg in x["segments"]:
  if not seg["numeric_pass"] or not seg["complete_reference_support"]:
   segments.append({"sequence":seq,**seg})
failures=R/"full_failed_or_incomplete_segments.json";failures.write_text(json.dumps(segments,indent=2),encoding="utf-8")
attack_table="\n".join(f'| {name} | {x["supported"]} | {x["accepted"]} | {x["primary_above2"]} |' for name,x in proof["cases"].items())
findings="""# Full two-sequence radar/optical Doppler development study

2026-10-21. Full Library01/02 development replay and independent artifact/arithmetic verification completed. Overall project completion remains false.

## Manual acquisition resolved

The user downloaded the two development archives through normal browser access. Their exact compressed sizes and complete128MiB prefixes matched the previously locked official sources. Both gzip streams passed end-of-stream/CRC validation. Copies were imported into this project with matching SHA256hashes; original Downloads files and earlier partial-study bytes remain intact. The previous Google Drive quota report is retained as historical evidence. No additional dataset payload was opened, hashed or scored, including Library03. Archives were streamed without extracting a large raw dataset tree.

## Frozen full-sequence results

No predictor, threshold, reference or acceptance rule changed after the pilot/prefix outcomes. The optical predictor reads point coordinates/radial velocities and latest Xsens gyro only. Original radar predictions and scoring labels remain separate. Availability is evaluated on the entire original20Hz radar query grid, including missing/stale/invalid context. A newer invalid optical result supersedes older valid results.

| Sequence | Optical frames | Requests | Admitted | Availability | Admitted primary P95(m/s) | Recorded>2m/s | Admitted unknown primary |
|---|---:|---:|---:|---:|---:|---:|---:|
"""+"\n".join(table)+"""

The fixed requirements remain availability>=95%, recorded P95discrepancy<=0.5m/s and zero recorded discrepancies>2m/s, with source/output age<=100ms and cross-modal scalar agreement<=1m/s. Artifact verification can pass even when these research requirements fail. Unknown reference scores do not count as successes.

Per-sequence60second segments, every failed/incomplete segment and the exact availability-reason counts are retained. The original four endpoint reference unknowns and the earlier Library01final302request segment failure remain unchanged in the original module. This new study does not erase those results or resolve the upstream reference disagreement by replacing labels.

## Fixed attack-interface replay

| Case | Supported query pairs | Admitted | Admitted recorded primary>2m/s |
|---|---:|---:|---:|
"""+attack_table+"""

These are fixed scalar-interface propagation tests, not full raw-stream attack reproductions. They separately test radar-only, optical-only and common shifts; missing/stale context; and1second replay with honest/forged timestamps. Unsupported pairs stay unavailable. Common-mode shifts and forged timestamps must be reported alongside the single-modality results.

Six recorded optical frames also passed independent official-reader field checks and encoded raw-float32coherent-forgery checks. The latter demonstrate that the optical estimator alone also accepts coherent Doppler translation. They do not establish a successful multisensor raw-attack defense.

## Timing, trust and evaluation limits

The saved service durations include reading/decompressing each binary member and decoding/fitting it, replayed serially at nominal publisher timestamps. They are specific to this offline archive path and PC; they do not measure a certified vehicle sensor/network stack. No timestamps were shifted or stale measurements restamped as fresh. Physical clocks and source authenticity remain unproven.

Separate radio/optical channels provide conditional redundancy. Both share gyro/context dependencies; simultaneous sensor compromise, common-clock compromise and equal-speed heading attacks remain limitations. No physical witness or production policy was enabled.

The full arithmetic audit independently recomputes causal query selection, readiness, all ten attack cases, per-segment metrics, source hashes and unchanged original results. The configured731core and55focused tests passed previously with4core skips and one existing legacy calibration warning; the full-study adapter/estimator code is hash-identical to those tested versions. Newly added manual-import/reporting helpers are syntax-checked and their acquisition/report outputs verified. No model retuning, manuscript, deployment, installation, commit or push occurred.

## Remaining project work

Read FULL_RESEARCH_DECISION.json and full_failed_or_incomplete_segments.json for measured gates. Preserve failures rather than tune them away. Whole-sequence results alone do not establish novelty or journal readiness. Physical source/clock/reference justification, compatible published-method comparisons and a justified later final-confirmation protocol remain open. Library03 stays reserved.
"""
(R/"FULL_FINDINGS.md").write_text(findings,encoding="utf-8")
tests=proof["tests"]
decision={"full_development_study_complete":True,"manual_acquisition_verified":True,"overall_project_complete":False,"availability_pass":{seq:x["availability"]>=.95 for seq,x in s.items()},"whole_sequence_numeric_pass":{seq:x["numeric_pass"] for seq,x in s.items()},"all_segments_numerically_pass":all(seg["numeric_pass"] for x in s.values() for seg in x["segments"]),"all_admitted_primary_scores_known":all(x["primary"]["unknown_reference"]==0 for x in s.values()),"availability_reason_counts":availability,"original_reference_failure_retained":True,"original_unknown_primary_endpoints":4,"source_authentication_established":False,"physical_clock_bound_established":False,"physical_witness_admitted":False,"Library03_payloads_accessed":False,"production_changed":False,"next_requirement":"Investigate measured development gate failures without changing frozen outcomes; complete compatible comparisons and physical reference/clock/source justification before reserved confirmation"}
(R/"FULL_RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
resolution={"previous_blocker":"Google Drive programmatic download quota exceeded","resolved_for_development_archives":True,"method":"user manual browser download, exact size/prefix/CRC/SHA256verified local import","manual_import_audit":"manual_import_audit.json","old_blocker_record_retained":True,"Library03_payloads_accessed":False}
(R/"ACQUISITION_RESOLUTION.json").write_text(json.dumps(resolution,indent=2),encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_FULL_OPTICAL.md"
if not backup.exists():shutil.copyfile(status,backup)
prefix="Latest: complete Library01/02optical/radar development replay,2026-10-21.\n\nManual downloads verified by exact size, locked-prefix match, gzip CRC and SHA256; publisher acquisition blocker resolved for these two development archives.\n"
for seq,x in s.items():
 prefix+=f'{seq}: {x["optical_frames"]}optical frames; {x["accepted"]}/{x["requests"]}requests admitted ({100*x["availability"]:.2f}%); recorded primary P95={x["primary"]["p95_mps"]:.6f}m/s; {x["primary"]["above2"]}recorded>2m/s; {x["primary"]["unknown_reference"]}admitted unknown primary scores; whole-sequence numeric pass={x["numeric_pass"]}.\n'
prefix+="Independent full arithmetic/provenance verification passed. Earlier partial outcomes, original reference failures and unknown scores retained. Full development study complete:YES. Overall project complete:NO. Physical source/clock/reference evidence, published-method comparisons/novelty and later final confirmation remain open. Production witness disabled; Library03/older holdouts untouched.\nEvidence: research/vehicle_crossmodal_doppler/FULL_FINDINGS.md, full_summary.json, FULL_RESEARCH_DECISION.json, full_independent_verification.json.\n\nPrevious verified status retained below.\n\n"
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
excluded={"full_file_manifest.json","full_completion.json","full_package_audit.json"}
files={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in excluded and p.stat().st_size<32*1024**2}
(R/"full_file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"full_development_study_complete":True,"overall_project_complete":False,"module_small_artifacts":len(files),"optical_frames":proof["optical_frames"],"queries":proof["fixed_interval_requests"],"core_passed":tests["core"]["counts"]["passed"],"focused_passed":tests["focused"]["counts"]["passed"],"manual_import_verified":True,"current_status_sha256":sha(status),"raw_archive_identities":"full_data_lock.json","Library03_payloads_accessed":False}
(R/"full_completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(decision,indent=2));print(json.dumps(completion,indent=2))
