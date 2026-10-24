from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent;H=R.parent/"vehicle_hercules_development"
proof=json.loads((R/"independent_verification.json").read_text());assert proof["passed"]
attack=json.loads((R/"source_stress_summary.json").read_text());diag=json.loads((R/"reference_diagnosis.json").read_text())
table=[]
for case,x in attack["cases"].items():
 table.append(f'| {case} | {x["quality_passed"]}/{x["attempts"]} | {x["quality_passed_nominal_shift_above2"]} | {x["primary"]["above2"]}/{x["primary"]["scored"]} |')
findings="""# Reference diagnosis and source-stress experiment

Completed2026-10-21. This batch resolves the question of whether a scoring-code change is justified and completes a fixed development source-stress suite. Overall project completion remains false.

## Reference diagnosis

The existing failed Library01 segment remains unchanged:302requests over15.05s,301reference-scored, P95disagreement1.1022975m/s under the original0.5m/s limit. No reference was swapped, offset tuned, model changed or segment deleted.

Recomputed raw ECEF position derivatives directly from the navigation latitude/longitude/height. In Library01's last15s, the published radar-pose derivative agrees fairly closely with raw navigation-position derivatives: median scalar discrepancy0.007416m/s and P950.158420m/s. Raw navigation-position-derived speed itself differs from interpolated direct INS speed by P951.394351m/s and maximum1.499909m/s. The discrepancy is therefore also present upstream of our pose differentiation. It is not explained solely by a bug in that differentiation code.

Candidate-versus-direct-INS P95 in that diagnostic window is0.456436m/s. This does not establish that the candidate or the INS velocity is correct. Reference representations have shared navigation ancestry and different sensor locations/processing; missing physical uncertainty/clock/lever-arm evidence remains unknown. A third trustworthy measurement or justified reference model is needed to settle the true error. Library02 provides a comparison sequence, with much smaller last-window reference disagreement. All1second/whole-sequence summaries and diagnostic arrays are retained.

## Fixed development source-stress results

564predeclared frames,11cases per frame,6,204trials. Nominal recomputation reproduces every saved candidate speed/quality. Cases and parameters were frozen before the stress outcomes. No reference is consulted to construct the coherent forgery. Library03 and earlier reserved payloads remain excluded.

| Case | Quality passes | Quality passes with>2m/s shift from nominal | Primary-reference recorded>2m/s / scored |
|---|---:|---:|---:|
"""+'\n'.join(table)+"""

These are estimator quality outcomes; production authorization remains disabled. Missing primary reference scores remain unknown. The reference-disagreement counts are not certified physical false-acceptance rates.

10%and20%uniformly indexed point corruptions of+10m/s retain all outputs, with nominal-relative P95speed shifts0.047622m/s and0.102150m/s. At40%corruption,498/564outputs fail the quality guard; the remaining66have no recorded>2m/s discrepancy in this cohort. This test family does not establish robustness to arbitrary outlier placement.

Missing IMU, a zero quaternion and IMU age100ms+1ns are rejected in every trial. Correctly timestamped1second replays are rejected whenever replay history exists. A30degree orientation spoof remains a valid quaternion and passes556/564quality screens, with P95nominal-relative speed shift0.944569m/s. Checking quaternion syntax alone cannot authenticate orientation.

A coherent+3m/s radar forgery passes564/564quality screens and shifts every estimate by3m/s. Of562primary-reference-scored cases,560exceed2m/s disagreement; all543secondary-reference-scored cases exceed2m/s. Replay with forged-current timestamps passes562/562supported cases and produces one recorded>2m/s discrepancy under each reference form. The two unsupported initial replay cases remain unavailable.

## Why residual tuning cannot close coherent-source forgery

For the fixed geometric design A, the candidate minimizes a robust loss of residual A*v+d. Under forged Doppler d'=d-A*Delta, the estimate v'=v+Delta has the same residual. Inlier fraction and residual RMSE therefore cannot distinguish the coherent forgery. The synthetic invariance test and all564real-input trials verify that property. This is a compromised-radar-source threat; the result does not imply that every independent-sensor or authenticated-source threat model is impossible.

The observed failure requires additional information: an independently trusted absolute motion check and justified timing/uncertainty, or an explicitly narrower threat model that assumes the radar producer is uncompromised. Signing transport alone does not make a compromised producer's measurement true. Reusing the evaluation INS reference as a predictor would invalidate the claimed independent evaluation. More thresholds on the same residual cannot supply the missing evidence.

## Verification and preserved state

731configured core tests passed,4skipped, one existing legacy calibration warning.35focused tests passed:7new source-stress/invariance checks plus28previous decoder/estimator checks. Independent replay recomputed6,204row outcomes, metrics and nominal-relative shifts, verified source/reference identities and confirmed the original candidate/code/predictions were unchanged. No installation, production-policy change, manuscript, deployment, commit or push.

Next substantive requirement: independent trustworthy motion evidence/reference justification and an explicit supported source-compromise scope. Complete compatible published-method reproduction and novelty assessment before using reserved data for final confirmation. Do not continue residual-only tuning or erase the retained final-segment failure.
"""
(R/"FINDINGS.md").write_text(findings,encoding="utf-8")
decision={"reference_code_fix_justified":False,"reason":"reference disagreement also exists between raw navigation position derivatives and direct INS velocity; true error unresolved","original_failed_segment_retained":True,"candidate_source_stress_completed":True,"trials":6204,"cohort_frames":564,"coherent_forgery_quality_passes":564,"coherent_forgery_speed_shift_mps":3,"independent_trusted_motion_evidence_required":True,"source_compromise_resistance_established":False,"physical_witness_admitted":False,"overall_project_complete":False,"Library03_payloads_accessed":False,"production_changed":False}
(R/"RESEARCH_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_REFERENCE_STRESS.md";assert not backup.exists();shutil.copyfile(status,backup)
prefix="""Latest: reference consistency diagnosis and6,204source-stress trials completed(2026-10-21).

No scoring correction was justified: Library01's last-window pose/raw-position derivatives agree, while raw navigation-position-derived speed and direct INS velocity disagree by P951.394351m/s in that window. Actual error remains unresolved. The original failed302-request segment and all original reference scores remain intact.
Completed564fixed development frames x11cases. Nominal predictions reproduce. The candidate tolerates the tested10%/20%isolated Doppler corruptions, rejects missing/invalid/stale context and correctly timestamped replays, but coherent+3m/s radar forgery passes564/564quality checks. Forged-current timestamp replay passes562/562supported cases, including one recorded>2m/s discrepancy. These are quality-check outcomes, not production authorization.
Residual invariance explains the coherent-forgery failure. Additional thresholds on the same residual cannot close it. Independent trustworthy absolute motion information, justified timing/error evidence, or an explicitly narrower source-compromise scope is required. Do not reuse scoring labels as predictor inputs.
731core tests pass,4skip, one existing calibration warning.35focused tests pass. Independent6,204-row metric/state replay verified unchanged candidate/source results.
Full project complete:NO. Source-compromise resistance/novelty/full published-method reproduction/final confirmation remain unresolved. Production witness disabled; Library03 and older reserved payloads remain unopened.
Evidence: research/vehicle_hercules_reference_diagnosis/FINDINGS.md, reference_diagnosis.json, source_stress_summary.json, RESEARCH_DECISION.json, independent_verification.json.
Prior verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
(R/"file_manifest.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"module_files":len(files),"attack_trials":6204,"core_passed":731,"core_skipped":4,"focused_passed":35,"original_candidate_unchanged":True,"reference_failure_retained":True,"Library03_payloads_accessed":False,"current_status_sha256":hashlib.sha256(status.read_bytes()).hexdigest()}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(completion,indent=2))
