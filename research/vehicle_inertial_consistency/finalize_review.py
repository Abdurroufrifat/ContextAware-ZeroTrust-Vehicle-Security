from shared import *
details=read("details_summary.json")
for name,digest in details["initial_outcomes_preserved"].items():assert sha(R/name)==digest
write("code_review.json",{"completed":True,"type":"independent read-only static code/method review plus parent tested corrections",
 "initial_findings":[{"issue":"Original forged-delta case replaces whole delta, stronger than stated","resolution":"Preserved initial outcome; explicit full-replacement clarification and separate additive-only forgery supplement","evidence":["CLARIFICATIONS.md","details_summary.json","test_additive_imu_forgery_preserves_nominal_mismatch"]},
 {"issue":"Rejection reasons absent","resolution":"Added ordered frame/query reason audit on full unchanged denominators","evidence":["diagnostic_details.py","details_summary.json","test_support_reasons_preserve_original_failure_priority"]}],
 "unresolved_important_findings":0,"limitations":["Reviewer did not execute tests or read measurement payloads","Parent runtime verification and RED-to-GREEN tests verify fixes; no claim of physical calibration/authenticity"]})
path=R/"finish_batch.py";source=path.read_text(encoding="utf-8").replace("127","134")
source=source.replace("17new tests were first run against the absent module and failed, then passed after implementation.","17initial tests and7supplemental tests each failed against absent implementations, then passed after implementation.")
needle='(R/"FINDINGS.md").write_text("\\n".join(lines)+"\\n",encoding="utf-8")'
assert source.count(needle)==1
insertion='''details=read("details_summary.json")
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
'''
source=source.replace(needle,insertion+needle)
source=source.replace('write("RESEARCH_DECISION.json",decision)','decision["review_findings_addressed"]=2\nwrite("RESEARCH_DECISION.json",decision)')
source=source.replace("Interface tests explicitly retain constant-world-offset invariance,","Original whole-delta forgery is explicitly distinguished from supplemental additive-only forgery; rejection reasons recorded on full denominators. Interface tests explicitly retain constant-world-offset invariance,")
path.write_text(source,encoding="utf-8")
print("Two static-review findings addressed in separately frozen supplemental evidence; initial outcomes unchanged.")
