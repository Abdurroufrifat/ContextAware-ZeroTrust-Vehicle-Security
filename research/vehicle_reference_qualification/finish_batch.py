from shared import *
verification=read("independent_verification.json");tests=read("test_validation.json");review=read("code_review.json")
assert verification["passed"] and review["completed"] and review["unresolved_important_findings"]==0
assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
summary=read("summary.json");sensitivity=read("sensitivity_rows.json")
lines=["# Absolute-reference qualification audit","",
"2026-10-21. This existing-data qualification audit is complete. No independently validated absolute-speed reference or numerical total reference-error bound was found in the inspected frozen inputs. That is a scoped finding, not a claim about all possible external data.",
"",
"## Source provenance and status finding","",
"HeRCULES describes sensor-specific poses generated from navigation information, extrinsics and interpolation: https://sites.google.com/view/herculesdataset/system . Thus direct INS velocity and derivatives of navigation/pose positions share source ancestry; their agreement is not independent validation. Optical Doppler and radar estimates are components of the evaluated system and cannot validate their own absolute accuracy.",
"",
"The inspected sensor_data folders each contain Continental timestamps, inspva.csv and xsens_imu.csv. Every INS row has11columns, including status, with no per-row velocity-uncertainty fields. Native receiver documentation distinguishes PVA/status from separate uncertainty logs: https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVA.htm and https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSSTDEV.htm . Standard deviations, when available, still need an error/probability model and do not automatically supply deterministic error bounds.",
"",
"Raw status tokens are preserved. Codes3/6 are interpreted using the NovAtel INS enum:3 is solution-good;6 is solution-free, where the filter does not accept available updates and suspects GNSS error. Source: https://docs.novatel.com/oem7/Content/SPAN_Logs/INSATT.htm . Applicability is inferred from the documented SPAN-CPT7/INSPVA context; firmware/configuration/authenticity are not independently proved.",
"",
"| Sequence | INS records code3 / code6 | Query tags code3 / code6 | Final-segment code3 / code6 |",
"|---|---:|---:|---:|"]
for seq,result in summary["sequences"].items():
    tail=result["groups"]["segment_6000" if seq=="Library01" else "segment_4800"]
    ins=result["ins_status_counts"];query=result["query_status_counts"];end=tail["status_counts"]
    lines.append(f'| {seq} | {ins.get("3",0)} / {ins.get("6",0)} | {query.get("3",0)} / {query.get("6",0)} | {end.get("3",0)} / {end.get("6",0)} |')
lines+=["",
"All302queries in the original failed Library01 interval map causally to raw status3, with status age<=100ms. Status6 does not overlap this interval. Therefore the observed status6 hypothesis does not explain this failure; excluding status6 would not remove it. Status3 itself supplies no quantitative physical true-error bound. No root-cause claim, data exclusion, alternate-reference selection or quality-based redefinition of the benchmark is made.",
"",
"The original failed result remains302attempts/301scored/1unknown, P951.102297507859495m/s disagreement. All11retained segments and all reference unknowns remain visible. The status strata are explicitly post-hoc diagnostics, not a replacement accuracy table.",
"",
"## Conditional reference-error sensitivity","",
"If a uniform pointwise total error bound epsilon on the reference were independently established for the same speed quantity, point and time, triangle inequality gives per-row true-error bounds max(0,abs(pred-reference)-epsilon) and abs(pred-reference)+epsilon. Empirical P95 bounds and definite/possible counts above2m/s follow on scored rows. This is deterministic conditional arithmetic, not a confidence interval, measured uncertainty, inferred covariance or population proof.",
"",
"The fixed scenarios None/0/0.05/0.1/0.2/0.5/1.0m/s are hypothetical. None remains unknown rather than zero. The result field reference_bound_known means a numeric assumption was supplied to the sensitivity function; it does not mean an actual reference bound was verified. Actual registry qualification is false throughout. Unknown admitted references block the conditional full-admitted-set P95 flag.",
"",
"| Current diagnostic mask | Recorded proxy P95 m/s | Largest hypothetical uniform epsilon compatible with upper P95<=0.5 on scored rows |",
"|---|---:|---:|"]
for seq in ["Library01","Library02"]:
    row=next(r for r in sensitivity if r["sequence"]==seq and r["group"]=="all" and r["scope"]=="current_inertial_mask" and r["epsilon_assumption_mps"] is None)
    lines.append(f'| {seq} | {row["observed_proxy_p95_mps"]:.9f} | {row["max_uniform_epsilon_for_scored_p95"]:.9f}m/s |')
lines+=["",
"These budgets are requirements under the stated hypothetical bound, not estimates of real reference accuracy. On the failed302-query interval, no nonnegative epsilon can make the conservative upper-P95 bound satisfy0.5 because the recorded P95 already exceeds it. Conversely, an unknown reference error also prevents a certified physical-failure conclusion; the recorded benchmark failure remains a failure under its unchanged proxy scoring.",
"",
"## Verified outputs and evidence requirements","",
"Independent monotone scanning rechecks all11,258causal status tags,104metric/stratum groups and182sensitivity rows. The16new tests first failed against the absent implementation and then passed.150focused and731configured core tests pass,4core tests skipped; the existing legacy calibration warning remains. Independent static code/method review found no material issues. Source and prior-result hashes are unchanged.",
"",
"Machine-readable reference_registry.json keeps shared ancestry, missing uncertainty/location/clock/calibration/authenticity evidence explicit. REFERENCE_EVIDENCE_REQUIREMENTS.json identifies what additional navigation uncertainty/configuration and independent absolute-motion validation would support stronger physical claims, with coverage of the original failure and missing endpoints. It does not demand that every journal study prove a deterministic pointwise bound. A probability-based validation would require its own justified error model.",
"",
"This existing-data investigation is closed with no qualifying physical anchor. Additional agreement filters cannot establish the missing reference evidence. Two clearly separated scopes remain possible: use documented additional evidence for physical validation, or retain transparent proxy-only development claims while leaving existing physical stop rules unresolved. No further algorithm retuning is justified merely to remove this failed interval.",
"",
"No production/witness promotion, reserved measurement access, manuscript, installation, commit, push or external message. Overall project scientific completion remains false."
]
(R/"FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
decision={"batch_complete":True,"existing_data_reference_investigation_complete":True,
 "qualified_independent_absolute_reference_found":False,"actual_verified_reference_error_bound_mps":None,
 "status6_explains_original_failed_interval":False,"failed_interval_status3_queries":302,
 "original_failed_window_unchanged":True,"all_reference_unknowns_retained":True,
 "physical_witness_admitted":False,"reserved_payloads_accessed":False,"overall_project_complete":False}
write("RESEARCH_DECISION.json",decision)
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_REFERENCE_AUDIT.md"
assert sha(backup)==read("input_lock.json")[str(status)]
prefix="""Latest: absolute-reference qualification audit completed,2026-10-21.

Existing-data reference investigation closed: no independently validated absolute-speed anchor or numerical total reference-error bound found in inspected frozen inputs. Pose/directINS/raw-position proxies share navigation ancestry; optical remains an evaluated predictor.
Raw INS codes3/6 preserved and tagged causally with100ms age limit. All302queries of failed Library01 interval have code3, not6. Thus status6 does not explain the original P951.1022975/301-score failure; status3 does not establish physical truth.
Original queries, masks, thresholds, reference labels,11segments and unknowns unchanged.182hypothetical error-budget rows are sensitivity scenarios only; no assumed epsilon adopted as measured accuracy. Missing admitted reference rows block full-admitted-set conditional P95 claims.
Independent11258query/104metric/182sensitivity audit passes.150focused/731core tests pass,4core skips; existing legacy calibration warning retained.
Next evidence requirement for stronger physical claims: timestamped uncertainty/configuration/point-frame/clock-calibration evidence and independent absolute-motion validation. Requirements saved. Further filters cannot supply this missing evidence. Proxy-only development claims remain available; existing physical stop rules unresolved.
Overall project complete:NO. Physical witness disabled; reserved payloads closed. No manuscript/production/commit/push changes.
Evidence: research/vehicle_reference_qualification/FINDINGS.md, reference_registry.json, REFERENCE_EVIDENCE_REQUIREMENTS.json, summary.json, sensitivity_rows.json, independent_verification.json and RESEARCH_DECISION.json.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
write("file_manifest.json",manifest)
write("completion.json",{"batch_complete":True,"overall_project_complete":False,"top_level_files":len(manifest),
 "focused_passed":150,"core_passed":731,"core_skipped":4,"current_status_sha256":sha(status)})
print(json.dumps(decision,indent=2))
