from shared import *
from release_core import file_sha
verification=read("independent_verification.json");tests=read("test_validation.json");review=read("code_review.json")
assert verification["passed"] and review["completed"] and review["unresolved_important_findings"]==0
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0 and tests["denials"]==[]
release=read("release_summary.json");correction=read("NUMERICAL_CORRECTION.json")
assert release["source_modules_verified"]==8 and release["exact_sensitivity_rows"]==182
scope_note="""# CO-ACT-CPS — verified development evidence release

The development evidence package is complete for the checked scope. Physical validation and journal readiness remain unestablished.

The project studies multi-source, context-aware zero-trust decisions for self-driving-vehicle security. It combines authenticated reports, freshness/replay checks, contextual policy rules and conditional motion-consistency evidence. Its sensor-trust and physical-error assumptions remain explicit.

## Start here

Current release: research/vehicle_development_release/README.md.
Results/index: research/vehicle_development_release/evidence_index.json.
Exact numerical correction: research/vehicle_development_release/NUMERICAL_CORRECTION.json.
Final-validation requirements: research/vehicle_evidence_freeze/FINAL_VALIDATION_REQUIREMENTS.json.

No dataset experiment, download or upload is required to inspect this completed release.

## Completed in this release

- Checked eight existing experiment manifests:218entries,259tracked input identities, including three selected native model checkpoints and their source snapshots.
- Recalculated182HeRCULES hypothetical reference-error scenarios with the existing exact-arithmetic implementation. A floating-point boundary defect is corrected in the authoritative exact sidecar; original frozen reports/code remain unchanged. None of the182recorded flags/counts changed.
- Confirmed731configured core tests and184focused research tests pass, with4core skips and one existing legacy calibration warning.
- Added one repeatable verifier. From D:/CO-ACT-CPS/phase2_joint_history, optionally run:
  D:/CO-ACT-CPS/.venv/Scripts/python.exe research/vehicle_development_release/verify_release.py
  It checks source identities and exact quantities; it does not refit models or repeat raw-data experiments.

These checks establish artifact and calculation consistency. They do not certify sensor calibration, clock accuracy, source authenticity or physical safety.

## Evidence remains separated by cohort

| Evidence | What it supports | Limits retained |
|---|---|---|
| Historical979-scene development | Prior frozen report metadata and conditional comparisons | Raw full event ledgers not rerun here; source-compromise failures remain |
| External Civic reference audit | Existing18-recording proxy-based evaluation artifact | Original64large GNSS disagreements and unknown physical reference bound remain |
| HeRCULES Library01/02 full query replay | Fixed development query-grid/representation results | Original Library01 final302queries/301scores have P951.1022975m/s disagreement |
| Adapted neural comparator | Three fixed seeds compared on matched available-input cohorts | This is an adaptation;61-frame temporal candidate P950.560221m/s exceeds0.5 |
| IMU differential diagnostic | Causal acceleration-change/interface tests | Constant and slow coordinated offsets pass; frozen-schedule timing is conditional |

The earlier authenticated gateway and synthetic gateway benchmark remain separate prior component milestones under research/vehicle_policy_gateway and research/vehicle_gateway_benchmark. Their synthetic results are not pooled with vehicle measurement results or newly rerun in this release.

Current physical references remain unqualified. All302queries of the failed HeRCULES interval carry recorded navigation status3; status filtering does not explain the failure. A solution-status label is not a numerical instantaneous-speed accuracy bound.

## What remains before stronger validation claims

The existing S1–S6 requirements are unchanged: source/collector protection, measurement-time evidence, calibration/uncertainty, endpoint appraisal, protected receiver/rollback state, and an independent reference with an untouched evaluation protocol.
The exact missing reference evidence is listed in research/vehicle_reference_qualification/REFERENCE_EVIDENCE_REQUIREMENTS.json.

Native reproduction of a recent complete published AV security framework and a specific supported novelty claim remain unresolved; current component adaptations must keep their labels. No human-centered effectiveness claim follows from software tests.

The next substantive physical-validation step needs the documented evidence package and justified new confirmation groups. More changes to the same filters do not supply those missing measurements. Proxy-only development claims may be reported with these limitations; original stop rules remain unresolved.

Production policy unchanged. Physical witness disabled. Promotion forbidden. Reserved final measurement payloads remain closed. No manuscript, commit or push was performed.

The previous project README and current-status history are preserved in research/vehicle_development_release/FINAL_README_BEFORE_RELEASE.md and PROJECT_STATUS_BEFORE_RELEASE.md.
"""
(R/"README.md").write_text(scope_note,encoding="utf-8")
(P/"FINAL_PROJECT_README.md").write_text(scope_note,encoding="utf-8")
decision={"development_release_complete":True,"verification_scope":"8existing manifests,218entries,259tracked inputs;184focused/731configured core tests",
 "exact_numeric_sidecar_authoritative":True,"historical_floating_reports_preserved":True,
 "physical_reference_appraised":False,"physical_validation_complete":False,"native_full_published_framework_reproduction_complete":False,
 "novelty_established":False,"journal_readiness_established":False,"overall_project_complete":False,
 "physical_witness_enabled":False,"production_policy_changed":False,"promotion_allowed":False,
 "required_next_evidence":"Existing S1-S6 package and justified new independent confirmation; no same-data filter retuning",
 "reserved_payloads_opened":False}
write("FINAL_READINESS.json",decision)
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_RELEASE.md"
assert sha(backup)==read("input_lock.json")["CURRENT_PROJECT_STATUS.md"]
prefix="""Latest: consolidated development evidence release verified,2026-10-21.

The checked development release is complete:8source manifests/218entries/259tracked inputs,3native checkpoint/source identities,182exact conditional-reference sensitivity rows,184focused and731configured core test passes,4core skips; legacy calibration warning retained.
A reproduced floating-boundary defect in the newer sensitivity checker is corrected by reusing the established exact Fraction implementation. Historical locked reports/code are preserved. Actual182scenario flags/counts unchanged; exact_sensitivity_rows.json is authoritative for conditional numerical decisions.
Canonical project entrypoint: FINAL_PROJECT_README.md, with evidence_index.json/FINAL_READINESS.json and one repeatable verify_release.py under research/vehicle_development_release. Legacy and new cohorts are not pooled; earlier failures/unknowns remain explicit. Raw datasets/estimators not rerun.
Existing development/gateway/component milestones are finished within their stated scopes. Physical validation, full published-framework reproduction, specific novelty and journal readiness are not established. Original HeRCULES302query failure, temporal neural-comparison failure, external64large GNSS disagreements and coordinated-source/rollback counterexamples remain.
Next substantive validation requires the existing S1–S6source/time/calibration/endpoint/receiver/evaluation evidence package and justified new confirmation groups. No further same-data filter retuning is warranted to imply those facts.
Overall project fully validated:NO. Physical witness disabled; production unchanged; promotion forbidden; reserved payloads closed. No manuscript/commit/push.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
manifest={p.name:file_sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
write("file_manifest.json",manifest)
write("completion.json",{"development_release_complete":True,"overall_project_complete":False,
 "top_level_files":len(manifest),"core_passed":731,"focused_passed":184,"core_skipped":4,
 "current_status_sha256":file_sha(status),"final_readme_sha256":file_sha(P/"FINAL_PROJECT_README.md")})
print(json.dumps(decision,indent=2))
