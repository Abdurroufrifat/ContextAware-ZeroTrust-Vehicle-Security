from shared import *
import csv
verify_lock()
verification=read("independent_verification.json");assert verification["passed"]
tests=read("test_validation.json");assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
review=read("code_review.json");assert review["completed"] and review["unresolved_important_findings"]==0
summary=read("summary.json")
lines=["# Causal accelerometer-consistency development diagnostic","",
"2026-10-21. The bounded diagnostic is complete. It does not resolve project-level scientific completion.","",
"## Implemented check","",
"The module reconstructs all11,259 original radar velocity vectors using the unchanged radar/IMU planar fit. Every original scalar prediction and quality flag is reproduced. It corrects velocity from radar origin to IMU origin using the measured calibration translation and causal gyro/quaternion, then compares each one-second horizontal velocity change with causally integrated accelerometer measurements. The17-column IMU layout was checked against https://sites.google.com/view/herculesdataset/system . The final three columns are magnetometer, not gravity-free acceleration.",
"",
"One-second history, maximum1.1s actual interval,100ms maximum age throughout every held IMU interval, and1m/s inclusive innovation threshold were fixed before numerical outcomes. No reference speeds, fitted offsets, bias calibration, lag alignment or neural outputs enter this check. The check uses raw accelerometer columns11..13 and causal zero-order hold. Missing initial history remains rejected and in the denominator.",
"",
"This calculation assumes the existing quaternion convention maps to a world frame whose z-axis is vertical. Attitude/gravity projection, accelerometer scale/bias, extrinsic/clock error bounds and source authenticity are not physically certified by this experiment. Integration consistency is therefore a conditional diagnostic, not a trusted physical witness.",
"",
"## Nominal fixed-schedule replay","",
"| Sequence | Guard | Queries | Admitted | Availability | Scored/unknown | Proxy P95 m/s |",
"|---|---|---:|---:|---:|---:|---:|"]
nominal_rows=[];attack_rows=[]
for seq,result in summary["sequences"].items():
    for name in ["original_guard","with_inertial_diagnostic"]:
        row=result[name]
        lines.append(f'| {seq} | {name} | {row["attempts"]} | {row["admitted"]} | {100*row["availability"]:.4f}% | {row["scored"]}/{row["unknown_reference"]} | {row["p95_mps"]:.6f} |')
        nominal_rows.append({"sequence":seq,"guard":name,**row})
lines+=["",
"The added mask keeps recorded frozen-schedule availability above95% in both sequences. This does not prove production readiness: it is applied to the previous causal completion/index schedule; additional check time is measured separately and is not charged back into that clock. No sensor-to-policy latency, deployment or physical timing claim. All prior gzip-path timing failures remain preserved.",
"",
"The original11 retained60s segments are reported individually in summary.json, including the final partial segments. Rejecting rows is not scored as successful protection or improved groundtruth. The original reference uncertainty and four full-radar reference endpoint unknowns remain; current combined-guard admitted unknown counts stay visible.",
"",
"## Inertial-stage attack diagnostics","",
"These cases modify estimated horizontal velocity and/or integrated IMU delta at the diagnostic interface only. They do not reproduce optical-point-cloud attacks, authenticated sensor compromise, or full-policy attack detection. Attack statistics use identical eligible frame pairs and keep the full original frame denominator for proxy-error metrics.",
"",
"| Sequence | Case | Supported | Accepted | Onset accepted/support | Persistent accepted/support | Accepted proxy >2 m/s |",
"|---|---|---:|---:|---:|---:|---:|"]
for seq,result in summary["sequences"].items():
    for name,row in result["attacks"].items():
        onset=row["onset_1p1s"];persistent=row["persistent_after_onset"]
        lines.append(f'| {seq} | {name} | {row["supported"]} | {row["accepted"]} | {onset["accepted"]}/{onset["supported"]} | {persistent["accepted"]}/{persistent["supported"]} | {row["proxy_metrics_on_full_frame_denominator"]["above2"]} |')
        attack_rows.append({"sequence":seq,"case":name,"supported":row["supported"],"accepted":row["accepted"],"rejected":row["rejected"],
          "onset_supported":onset["supported"],"onset_accepted":onset["accepted"],"persistent_supported":persistent["supported"],
          "persistent_accepted":persistent["accepted"],"accepted_proxy_above2":row["proxy_metrics_on_full_frame_denominator"]["above2"]})
lines+=["",
"A constant world-frame velocity offset leaves every differential innovation unchanged within floating-point tolerance: (v(t)+c)-(v(t-h)+c)-integral(a) = v(t)-v(t-h)-integral(a). Thus this check cannot authenticate absolute speed or remove a pre-existing constant common-mode offset. An abrupt step may be visible during onset, but that does not imply persistent rejection after the offset becomes constant. The10s ramp and forged-integrated-IMU cases remain explicit counterexamples/limits; no threshold was retuned after seeing them.",
"",
"## Original failed-window accounting",""]
for seq,result in summary["sequences"].items():
    tail=result["segments"][-1]
    old=tail["original_guard"];new=tail["with_inertial_diagnostic"]
    lines.append(f'{seq} final retained segment: {tail["requests"]}queries. Original admitted={old["admitted"]},scored={old["scored"]},unknown={old["unknown_reference"]},P95={old["p95_mps"]}; added check admitted={new["admitted"]},scored={new["scored"]},unknown={new["unknown_reference"]},P95={new["p95_mps"]}. Numeric-pass original/added={old["numeric_pass"]}/{new["numeric_pass"]}.')
lines+=["",
"The Library01 original302-query reference-disagreement failure is not fixed by this extra channel. No reference replacement, temporal shift, hidden exclusion or condition relaxation. This result also does not alter the previous neural comparison's61-frame temporal-test P950.560221m/s failure.",
"",
"## Verification and remaining work","",
"17initial tests and7supplemental tests each failed against absent implementations, then passed after implementation. The134focused and731configured core tests pass;4core tests remain skipped and the existing legacy calibration warning remains. An independent implementation re-integrates all eligible intervals using a separate sample-by-sample loop, checks every original radar speed/quality identity, mounting-position correction,11retained segments and all nominal/attack metrics. Source and prior-result hashes are preserved.",
"",
"New diagnostic artifacts and a static code review are saved locally. Reserved payloads remain closed; physical witness disabled. No production gate or manuscript changed, no installation, commit, push or external message.",
"",
"This batch adds evidence about a usable third measurement channel and a precise limit of differential checks. The next required evidence is an independently justified absolute-motion reference/anchor with calibration, clock and source-trust bounds; more agreement filters cannot establish that missing fact. The current record supports conditional development replay and proxy-disagreement claims only. Overall project completion remains false."
]
details=read("details_summary.json")
lines+=["","## Review corrections with initial outcomes preserved","",
"Original case step_with_forged_imu_delta replaces the whole integrated IMU delta with attacked radar change; it fully cancels nominal mismatch too. Its initial wording was too broad. Original protocol/code/results stay frozen and unchanged. CLARIFICATIONS.md records this limitation.",
"",
"Supplemental additive_step_imu_forgery adds only bias(current)-bias(previous) to the original IMU delta. It preserves every nominal mismatch and acceptance decision. It is separately reported rather than overwriting or relabelling the first case. Seven additional RED-to-GREEN tests cover this distinction and rejection reasons.",
"",
"Query rejection reasons below follow the original scalar policy first and then the added inertial stage; counts sum to the original complete query grid. Startup/missing history remains a benign availability cost, not successful attack detection.",
"",
"| Sequence | Query reason | Count |","|---|---|---:|"]
for seq,row in details["sequences"].items():
    for reason,count in row["new_query_reason_counts"].items():
        lines.append(f'| {seq} | {reason} | {count} |')
    lines.append(f'{seq} additive-only forgery: supported={row["additive_forgery_supported"]},accepted={row["additive_forgery_accepted"]}; nominal mismatch preserved. Frame reason counts: {row["frame_reason_counts"]}.')
(R/"FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
for filename,rows in [("nominal_comparison.csv",nominal_rows),("attack_diagnostics.csv",attack_rows)]:
    with (R/filename).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
decision={"batch_complete":True,"diagnostic_implemented_and_verified":True,"diagnostic_enabled_in_production":False,
 "full_radar_frames":11259,"full_query_attempts":11258,"availability_condition_on_frozen_schedule_met_both":True,
 "original_failed_window_fixed":False,"reference_gap_resolved":False,"constant_offset_detectable_by_this_statistic":False,
 "full_attack_policy_reproduced":False,"physical_witness_admitted":False,"overall_project_complete":False,
 "reserved_payloads_accessed":False}
decision["review_findings_addressed"]=2
write("RESEARCH_DECISION.json",decision)
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_INERTIAL.md"
assert sha(backup)==read("input_lock.json")[str(status)]
prefix="""Latest: causal accelerometer-consistency diagnostic completed,2026-10-21.

Reconstructed all11,259 original radar vectors with unchanged scalar speeds/quality; added IMU acceleration-change consistency with causal1s history,100ms freshness and fixed1m/s innovation tolerance. Sensor-origin translation/gyro correction included. No reference or learned-target features.
On the FROZEN prior completion schedule, added-mask availability99.4446% Library01 and96.0856% Library02. Additional check time separately measured, not inserted into prior clock; no production/vehicle timing claim.
All11retained segments and unknowns visible. Original Library01 final302-query/301-score P951.1022975 failure remains; previous61-frame neural-comparison temporal failure also remains.
Original whole-delta forgery is explicitly distinguished from supplemental additive-only forgery; rejection reasons recorded on full denominators. Interface tests explicitly retain constant-world-offset invariance, step-onset vs persistent behavior, slow ramp and forged IMU delta. This is an inertial-stage diagnostic only; no optical injection or full attack-policy reproduction. Absolute motion/source trust is not proven by differential consistency.
Independent sample-loop integration/metric/source checks pass.134focused/731core tests pass,4core skips; legacy calibration warning retained.
Next required evidence: independently justified absolute-motion reference/anchor plus calibration, clock and source-authenticity bounds. Overall project complete:NO. Physical witness disabled; all reserved payloads closed.
Evidence: research/vehicle_inertial_consistency/FINDINGS.md, summary.json, nominal_comparison.csv, attack_diagnostics.csv, independent_verification.json and RESEARCH_DECISION.json.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
(R/"PROGRESS.md").write_text("Complete: frozen protocol;17RED-to-GREEN tests;11,259frame vector/source replay;11258query nominal masks;11retained segments;five inertial-interface cases;independent integration/statistics verification;134focused/731core passes,4skips;static review and package audit. Original failures and unknowns retained. Remote test-file write response timed out once; read-back confirmed file complete before RED test execution.\n",encoding="utf-8")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
write("file_manifest.json",manifest)
write("completion.json",{"batch_complete":True,"overall_project_complete":False,"top_level_files":len(manifest),
 "focused_passed":134,"core_passed":731,"core_skipped":4,"current_status_sha256":sha(status)})
print(json.dumps(decision,indent=2))
for seq,result in summary["sequences"].items():
    print(seq,"tail",json.dumps(result["segments"][-1]),flush=True)
    for name,row in result["attacks"].items():
        print(seq,name,json.dumps({k:row[k] for k in ["supported","accepted","onset_1p1s","persistent_after_onset"]}),flush=True)
