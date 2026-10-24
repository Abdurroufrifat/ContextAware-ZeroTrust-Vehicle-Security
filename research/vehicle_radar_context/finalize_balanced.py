"""Persist verified development results and actionable remaining requirements."""
import hashlib,json,subprocess,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
def rows(path):
    with path.open() as f: return [json.loads(l) for l in f]
report=json.loads((ROOT/"balanced_experiment_summary.json").read_text())
verification=json.loads((ROOT/"balanced_independent_verification.json").read_text())
tests=json.loads((ROOT/"test_validation_v2.json").read_text())
assert verification["all_passed"] and all(t["exit_code"]==0 for t in tests.values())
runtime=ROOT/report["runtime_relative_path"]
est=rows(runtime/"estimates.jsonl")
old_bad=[r for r in json.loads((ROOT/"consensus_diagnosis.json").read_text())["queries"] if r["error"] is not None and r["error"]>2]
comparisons=[]
for r in old_bad:
    v=next(e for e in est if e["estimator"]=="sensor_balanced_v2" and e["recording"]==r["recording"] and e["index"]==r["index"] and e["track"]=="raw_header_association")
    f=v["estimate"]
    comparisons.append(dict(recording=r["recording"],index=r["index"],reference=r["reference"],
        old_speed=r["speed"],old_error=r["error"],new_accepted=f["accepted"],new_reason=f["reason"],
        new_speed=f.get("speed_mps"),new_error=abs(f["speed_mps"]-r["reference"]) if f["accepted"] else None))
per_record={}
for name in report["source_frame_hashes"]:
    per_record[name]={}
    for estimator in report["quality"]:
        vv=[e for e in est if e["estimator"]==estimator and e["recording"]==name and e["track"]=="raw_header_association"]
        per_record[name][estimator]=dict(requests=len(vv),supported=sum(e["estimate"]["accepted"] for e in vv),
            reasons=dict(Counter(e["estimate"]["reason"] for e in vv)))
comparison=dict(previous_large_error_cases=comparisons,per_record_motion_support=per_record)
(ROOT/"balanced_scene_comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
lines=["# Sensor-balanced radar development comparison","",
"Status: the bounded estimator improvement and expanded signed-attack evaluation are complete. The full project is not complete and no policy is promoted.","",
"## Method and preserved controls","",
"Point-count RANSAC let dense detections dominate the consensus. The v2 ablation samples three sensor IDs uniformly, gives each sensor equal total evidence mass, scores weighted consensus and fits weighted least squares. The 50% scene-quality criterion now measures sensor-balanced mass rather than point count; it is a different empirical estimator, not a physical certificate.",
"",
"The frozen protocol keeps the residual threshold at 0.25 m/s, 256 hypotheses, three inlier sensors, condition limit 1000, speed limit 60 m/s and yaw limit 2 rad/s. Candidate agreement remains 1 m/s inclusive and evidence age remains 100 ms inclusive. Neither CAN nor INS is an input to radar estimation. INS is used only for offline discrepancy scoring. No thresholds were selected after v2 outcomes.",
"",
"## Nominal results","",
"| Raw-header development metric | Point-count v1 | Sensor-balanced v2 |","|---|---:|---:|"]
for label,field in [("Supported radar estimates","accepted"),("Standalone P95 discrepancy (m/s)","p95_discrepancy_mps"),("Standalone maximum discrepancy (m/s)","maximum_discrepancy_mps"),("Standalone discrepancies >2 m/s","above_2_mps")]:
    q=[report["quality"][e]["raw_header_association"][field] for e in report["quality"]]
    lines.append(f"| {label} | {q[0]:.6g} | {q[1]:.6g} |")
for label,field in [("Combined clean coverage (%)","coverage_pct"),("Combined P95 discrepancy (m/s)","p95_discrepancy_mps")]:
    q=[report["results"][e]["raw_header_association"]["nominal"]["context_no_ood_plus_radar"][field] for e in report["quality"]]
    lines.append(f"| {label} | {q[0]:.6g} | {q[1]:.6g} |")
lines+=["","All 229 supported v2 radar estimates have an offline reference. The combined policy accepts 224/283 nominal requests, with zero unscorable accepted values. Its remaining 59 rejections comprise 54 missing scene-support estimates and five learned-context rejections.","",
"## Coordinated CAN compromise","",
"The attacker controls authorized candidate and peer signers, preserving yaw, steering and acceleration. All packets are signed and freshness/replay checks are exercised. Attack onset is one second into each evaluation partition: 252 active requests per attack. Radar matching assumes the attacker knows the available radar estimate; when radar is unavailable, that attack is not craftable and those requests must not be described as detected forgeries.",
"",
"| Active attack | Old active accepted | Old accepted error >2 m/s | New active accepted | New accepted error >2 m/s | New craftable active requests |",
"|---|---:|---:|---:|---:|---:|"]
for case in json.loads((ROOT/"protocol_v2.json").read_text())["attacks"]:
    if case=="nominal": continue
    a=report["results"]["point_count_v1"]["raw_header_association"][case]["context_no_ood_plus_radar"]
    z=report["results"]["sensor_balanced_v2"]["raw_header_association"][case]["context_no_ood_plus_radar"]
    crafted=report["crafted_active_counts"].get("sensor_balanced_v2|raw_header_association|"+case,0)
    lines.append(f'| {case} | {a["active_accepted"]} | {a["accepted_discrepancy_above_2_mps"]} | {z["active_accepted"]} | {z["accepted_discrepancy_above_2_mps"]} | {crafted} |')
lines+=["",
"Some forged reports are accepted, especially when the vehicle is stationary or an attacker matches an accurate radar estimate. Zero large accepted discrepancies in these development cases does not mean every forged packet is detected, and does not establish security against arbitrary radar compromise.",
"",
"## Previous large-error cases","",
"| Record / index | Old speed (m/s) | New outcome |","|---|---:|---|"]
for c in comparisons:
    outcome=f'{c["new_speed"]:.6f} m/s; discrepancy {c["new_error"]:.6f} m/s' if c["new_accepted"] else "Declined: "+c["new_reason"]
    lines.append(f'| {c["recording"]} / {c["index"]} | {c["old_speed"]:.6f} | {outcome} |')
lines+=["","## Verification and limits","",
f'All 84 relevant tests passed: 24 radar, 20 CAN-context and 40 authentication. Independent verification checked {verification["causal_frame_selection_checks"]} causal frame selections, {verification["accepted_geometry_checks"]} accepted geometries, {verification["signed_packet_checks"]} signed packets and {verification["policy_decisions"]} decisions. The original baseline estimates are unchanged.',
"",
"Known limitations remain:",
"- All three recordings are already inspected development data from the same day; this is not independent confirmation.",
"- Raw packet headers establish transport association, not certified acquisition age or sensor authenticity.",
"- Original processed headers remain 106 ms earlier. Under that track, all radar estimates fail the unchanged 100 ms age limit, so combined coverage is zero.",
"- Raw-header combined nominal coverage is 79.152%, below the 95% requirement.",
"- No hard reference-error certificate, original acquisition attestation or adversarial error bound has been established.",
"- Sensor-balanced RANSAC remains a static-scene research assumption; moving returns can still defeat it under other conditions.",
"- Reserved evaluation payloads remain unopened, unhashed and unscored.",
"",
"## Next priority","",
"Resolve the independent-source freshness/coverage blocker before adding further tuned policy variants: obtain validated acquisition-time evidence or an adequately fresh independent stream with broader static-scene support. Then freeze a new evaluation protocol, run fair published-method comparisons and ablations, and only afterward unlock independent confirmation. Do not fill missing radar estimates with CAN-derived truth, retain old values as fresh, relax the deadline or relabel transport time as acquisition time.","",
"No manuscript, deployment, commit or GitHub push was performed."]
(ROOT/"BALANCED_FINDINGS.md").write_text("\n".join(lines)+"\n")
status="""# Current CO-ACT-CPS status

The sensor-balanced independent radar ablation and broader signed-attack evaluation are implemented and independently verified. The full project still fails the coverage and acquisition-timing requirements.

Latest batch:
- Diagnosed point-count consensus failures concentrated in dense radar returns.
- Preserved v1 and froze a single sensor-balanced v2 candidate before evaluating it.
- Tested nominal operation plus coordinated positive/negative bias, zeroing, scaling and radar-matching forgeries.
- All 84 relevant tests passed (24 radar, 20 CAN context, 40 authentication).
- Independently checked 1,132 causal frame selections, 450 accepted geometries, 15,084 signed packets and 27,168 decisions.
- Under raw-header association, supported radar estimates rise from 221 to 229 out of 283.
- Standalone radar discrepancies above 2 m/s fall from three to zero; maximum discrepancy falls from 8.626 to 0.14446 m/s among supported estimates.
- Combined nominal coverage rises from 75.265% to 79.152%; P95 candidate discrepancy is 0.09900 m/s.
- The new tests expose three large accepted discrepancies in each old scaling and radar-matching attack case; v2 has zero in all five tested active attack cases.
- Several forged reports remain accepted with small physical discrepancy; this is not complete forgery detection.

Main results:
research/vehicle_radar_context/BALANCED_FINDINGS.md
research/vehicle_radar_context/balanced_experiment_summary.json
research/vehicle_radar_context/balanced_independent_verification.json
research/vehicle_radar_context/balanced_scene_comparison.json
research/vehicle_radar_context/balanced_completion.json

Remaining blockers:
1. Combined clean coverage is 79.152%, below 95%: 54 radar-support rejections and five learned-context rejections.
2. Original processed radar headers remain 106 ms earlier than raw packet headers. With those original timestamps, all radar scans are too old for the unchanged 100 ms deadline. Physical acquisition freshness is not certified.
3. All three records are previously inspected development data from one day. Independent confirmation, broader source/day generalization, published-method reproductions and novelty assessment remain unfinished.
4. No hard physical reference-error bound or original acquisition attestation exists.

Next priority is validated fresh independent evidence and sufficient scene coverage, before final comparisons and frozen independent confirmation. Missing support must remain missing; do not retime old values or fill the independent witness with CAN inputs.

Candidate agreement remains 1 m/s inclusive and age remains 100 ms inclusive. V2 changes the empirical scene estimator to equal-sensor mass; it is not a hard physical certificate. Production policy is unchanged, physical witness use and promotion remain disabled. Reserved payloads remain unopened/unhashed/unscored. No manuscript, deployment, commit or GitHub push was performed.
"""
(ROOT.parent.parent/"CURRENT_PROJECT_STATUS.md").write_text(status)
completion=dict(batch_complete=True,full_project_complete=False,all_tests_passed=True,test_count=84,
    independent_verification_passed=True,nominal_coverage_pct=79.15194346289752,
    nominal_coverage_requirement_pct=95,acquisition_age_certified=False,original_corrected_track_coverage_pct=0,
    development_only=True,reserved_payloads_accessed=False,production_policy_changed=False,
    physical_witness_enabled=False,promotion_allowed=False,next_priority="validated fresh independent evidence and sufficient scene coverage")
(ROOT/"balanced_completion.json").write_text(json.dumps(completion,indent=2)+"\n")
files=["protocol_v2.json","protocol_v2_freeze.json","radar_balanced.py","attack_values.py","run_balanced_experiment.py",
    "verify_balanced.py","test_balanced.py","test_attack_values.py","freeze_and_test_v2.py","validate_v2_tests.py",
    "test_validation_v2.json","tests_v2_radar.log","tests_v2_CAN_context.log","tests_v2_authentication.log",
    "diagnose_consensus.py","consensus_diagnosis.json","balanced_experiment_summary.json","balanced_independent_verification.json",
    "balanced_scene_comparison.json","BALANCED_FINDINGS.md","balanced_completion.json","finalize_balanced.py"]
manifest={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
(ROOT/"balanced_file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(completion=completion,old_error_outcomes=comparisons),indent=2))
