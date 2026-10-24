from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent;C=R.parent/"vehicle_crossmodal_doppler"
v=json.loads((R/"independent_verification.json").read_text());assert v["passed"]
s=json.loads((R/"summary.json").read_text())["sequences"];d=json.loads((R/"diagnosis.json").read_text())["sequences"]
failed=[{"sequence":seq,**x} for seq,y in s.items() for x in y["segments"] if not x["numeric_pass"] or not x["complete_reference_support"]]
(R/"failed_or_incomplete_segments.json").write_text(json.dumps(failed,indent=2),encoding="utf-8")
table="\n".join(f'| {seq} | {100*x["old_gzip_path_availability"]:.2f}% | {100*x["availability"]:.2f}% | {x["combined_service_p95_ms"]:.3f} | {x["primary"]["p95_mps"]:.6f} | {x["primary"]["above2"]} | {x["primary"]["unknown_reference"]} |' for seq,x in s.items())
attacks="\n".join(f'| {case} | {x["supported"]} | {x["accepted"]} | {x["primary_above2"]} |' for case,x in v["cases"].items())
text="""# Optical input timing diagnosis and compact-representation replay

2026-10-21. This development batch is complete. Overall project/physical certification remain incomplete.

## Root cause and one representation change

The128frame component profile reproduced every saved original prediction. Compressed member reading/decompression costs P9515.76ms/16.27ms while the complete estimator costs1.37ms/1.77ms. Library02's fixed source timestamps leave little deadline margin: a constant-service counterfactual supports95%freshness only up to3.25ms on the predeclared0.25ms grid. Its zero-service timestamp ceiling is97.09%. These are timing-feasibility diagnostics, not observed deployment results.

The original radar benchmark preloads archive bytes before timing the estimator, while the optical archive benchmark times extraction/decompression too. Both original results remain intact. Their timing boundaries differ. Nothing here redefines old recorded outcomes.

Implemented a compact raw packet containing exactly the at-most512original records already chosen by the frozen estimator, before finite/range filtering. The original source stamp/count/full-member SHA256lineage remain in the packet. All metadata and bytes are covered by an unkeyed integrity digest. That digest is not source authentication; a compromised producer can forge it. No predictor, calibration, threshold, reference, gyro association or acceptance rule changed. The new packet caches total84,112,200bytes, compared with15,639,983,844original point-record bytes. They preserve estimator inputs, not the entire point cloud for arbitrary future tasks.

## Full development results

| Sequence | Previous gzip-path availability | Compact replay availability | Construction+consumer P95(ms) | Admitted primary P95(m/s) | Recorded>2m/s | Admitted unknown primary |
|---|---:|---:|---:|---:|---:|---:|
"""+table+"""

All5630speeds match the previous full-data predictions within1e-10m/s; measured maximum differences are zero and all quality decisions match. Every original20Hz query is retained:11258requests. Conversion CPU, packet read, metadata/integrity validation, gyro lookup and original fitting are included in each measured service. Source stamps are unchanged; completion times are serial and newer invalid results supersede old valid results. All60second segments and all failed/incomplete segments remain visible.

These are two different input representations, with explicit timing boundaries. Compact replay assumes the complete uncompressed source-byte buffer is already available for packet construction, then measures construction and cache consumer work. Gzip preparation, full-member provenance hashing, cache output preparation, vehicle network delivery, cold-cache behavior and physical source-clock/producer implementation are outside that consumer experiment. Their costs were not converted into artificial fresh source timestamps. Thus96%compact-replay availability does not prove that the original compressed delivery path or a vehicle deployment meets95%. The previous80.57%Library02gzip failure remains a valid observed result.

The original Library01last302request reference disagreement persists. The original four unknown reference endpoints stay unknown; this run also reports its admitted unknown primary/secondary scores separately. No failing window was dropped, label replaced, test-data gate retuned or witness admitted.

## Fixed interface stress cases

| Case | Supported query pairs | Admitted | Admitted recorded primary>2m/s |
|---|---:|---:|---:|
"""+attacks+"""

Cases are fixed scalar-interface propagation tests, not end-to-end raw multisensor attack demonstrations. Report all of them, including simultaneous-modality and forged-timestamp limitations. Equal-speed heading changes, shared gyro/clock compromise and source authenticity remain unresolved. Compact encoding does not fix those trust problems.

## Verification

16new tests first failed because the adapter was missing, then passed.71focused tests and731configured core tests pass;4core tests skip and one existing legacy calibration warning remains. Independent verification checks all5630packet schemas/metadata/digests,128raw-byte sampling oracles,5630exact prediction equivalences and11258causal query/metric replays across ten interface cases. Old full-study hashes/core/failed decisions remain unchanged. No Library03/other protected payload, network install, manuscript, production policy, commit or push.

Next research work: compatible published-method comparisons under the same explicit input/timing scope, reference/clock/source-trust justification, and a finite final-evaluation protocol. Do not reopen residual-only tuning or claim physical readiness from this conditional replay.
"""
# Derive point bytes from locked source metadata, not a hand-entered total.
actual_source=sum(x["source_bytes"] for x in s.values())
text=text.replace("15,639,983,844",f"{actual_source:,}")
(R/"FINDINGS.md").write_text(text,encoding="utf-8")
decision={"timing_batch_complete":True,"root_cause":"compressed archive extraction dominates optical service and original radar/optical timing boundaries differ","representation":"same fixed512records, original source stamp, source-lineage metadata, unkeyed integrity digest","compact_replay_numeric_pass":{seq:x["whole_numeric_pass"] for seq,x in s.items()},"compact_availability":{seq:x["availability"] for seq,x in s.items()},"all_segments_numerically_pass":all(x["numeric_pass"] for y in s.values() for x in y["segments"]),"all_admitted_primary_reference_known":all(x["primary"]["unknown_reference"]==0 for x in s.values()),"exact_prediction_equivalences":5630,"old_gzip_failure_retained":True,"old_reference_failure_retained":True,"physical_end_to_end_availability_established":False,"source_authentication_established":False,"physical_witness_admitted":False,"overall_project_complete":False,"Library03_payloads_accessed":False,"production_changed":False}
(R/"RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_TIMING.md"
if not backup.exists():shutil.copyfile(status,backup)
prefix="Latest: optical timing diagnosis and fixed-record compact-replay study completed,2026-10-21.\n\nArchive read/decompression dominates original optical service(~16msP95); estimator alone<2ms. Compact representation preserves original selected512records/stamps and includes conversion+consumer costs.\n"
for seq,x in s.items():prefix+=f'{seq}: compact-replay availability={100*x["availability"]:.2f}%, construction+consumer P95={x["combined_service_p95_ms"]:.3f}ms; admitted primary P95={x["primary"]["p95_mps"]:.6f}m/s; recorded>2={x["primary"]["above2"]}; unknown admitted primary={x["primary"]["unknown_reference"]}.\n'
prefix+="All5630predictions/qualities unchanged.731core/71focused tests pass,4core skip, one existing calibration warning. Independent packet/arithmetic/state audit passes. Original Library02gzip80.57%failure and Library01last-segment reference failure remain intact. Compact replay excludes upstream archive preparation/vehicle transport and does not prove physical end-to-end availability or source authenticity. Overall project complete:NO. Production disabled; Library03/protected payloads untouched.\nEvidence: research/vehicle_optical_timing/FINDINGS.md, summary.json, RESEARCH_DECISION.json, independent_verification.json.\n\nPrevious verified status retained below.\n\n"
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
files={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"files":len(files),"optical_frames":5630,"queries":11258,"focused_passed":71,"core_passed":731,"core_skipped":4,"current_status_sha256":sha(status)}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(decision,indent=2));print(json.dumps(completion,indent=2))
