"""Read-only evidence replay and explicit project completion decision."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
from completion_core import complete,metric_gate
R=Path(__file__).resolve().parent;P=R.parent.parent;B=P/"research";fingerprints={}
def safe(path):
 p=Path(path).resolve();n=p.as_posix().lower()
 if not p.is_relative_to(P) or "vehicle_final_evaluation" in n or "rav4" in n or ("urban02" in n and "night" in n):raise ValueError("excluded evidence path")
 return p
def touch(path):
 p=safe(path);fingerprints[p.relative_to(P).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(path):return json.loads(touch(path).read_text())
def metrics(y,p):
 e=np.abs(np.asarray(y)-np.asarray(p))
 return dict(count=len(e),mae_mps=float(e.mean()),rmse_mps=float(np.sqrt(np.mean(e*e))),p95_mps=float(np.quantile(e,.95)),maximum_mps=float(e.max()),above_2_mps=int((e>2).sum()))
def match(actual,expected):
 for k,v in actual.items():
  if k in expected:
   assert abs(v-expected[k])<=1e-10,(k,v,expected[k])
def main():
 core=read(R/"core_test_validation.json");focused=read(R/"focused_test_validation.json");pack=read(R/"packaging_validation.json")
 axis=B/"vehicle_signed_axis_correction";result=read(axis/"results.json")
 # Only this module's explicit local manifest is checked; historical manifests that
 # include reserved paths are not followed.
 manifest=read(axis/"sha256.json")
 for name,h in manifest.items():assert hashlib.sha256(touch(axis/name).read_bytes()).hexdigest()==h
 members=read(B/"vehicle_route_validation"/"membership.json")
 routechecks={}
 with np.load(touch(axis/"oof_predictions.npz"),allow_pickle=False) as z:
  y=z["target_mps"];pred=z["signed_axis_corrected__prediction_mps"];assert len(y)==len(members)==27178
  pooled=metrics(y,pred);match(pooled,result["route"]["pooled"]["models"]["signed_axis_corrected"])
  labels=np.array([x["window_id"] for x in members])
  for label in sorted(set(labels)):
   mask=labels==label;score=metrics(y[mask],pred[mask])
   match(score,result["route"]["by_window"][label]["models"]["signed_axis_corrected"])
   routechecks[label]=score
 references=read(B/"vehicle_availability_audit"/"request_reference.json")
 with np.load(touch(axis/"development_replay.npz"),allow_pickle=False) as z:
  pre="signed_axis_corrected__request__";times=z[pre+"request_times_ns"];status=z[pre+"status"];pred=z[pre+"prediction_mps"]
  np.testing.assert_array_equal(times,[x["time_ns"] for x in references])
  y=np.array([np.nan if r["reference_mps"] is None else r["reference_mps"] for r in references])
  selected=status==0;valid=selected&np.isfinite(y);scored=metrics(y[valid],pred[valid])
  expected=result["full_development"]["methods"]["signed_axis_corrected"]["requests"]["overall"]
  assert len(times)==expected["requests"]==37340 and int(selected.sum())==expected["accepted"]["count"]==36046
  match(scored,expected["accepted"])
  segments={}
  for i in sorted(set(((times-times[0])//300000000000).tolist())):
   mask=(times-times[0])//300000000000==i;score=metrics(y[mask&valid],pred[mask&valid])
   old=result["full_development"]["methods"]["signed_axis_corrected"]["requests"]["segments_300s"][str(i)]
   assert int(mask.sum())==old["requests"];match(score,old["accepted"])
   segments[str(i)]=dict(requests=int(mask.sum()),accepted=int((mask&selected).sum()),metrics=score)
 blocks=result["full_development"]["methods"]["signed_axis_corrected"]["previous_supported_blocks"]
 failedblocks={k:v["accepted"]["p95_mps"] for k,v in blocks.items() if v["accepted"]["p95_mps"]>.5}
 failedwindows={k:v["p95_mps"] for k,v in routechecks.items() if v["p95_mps"]>.5}
 axis_audit=dict(route_rows=27178,route_pooled=pooled,route_windows_checked=15,failed_route_windows=failedwindows,
  scheduled_requests=37340,accepted_requests=36046,availability_pct=100*36046/37340,accepted_metrics=scored,segments_recomputed=segments,
  retained_block_failures_recorded=failedblocks,retained_blocks_recomputed=False,
  limitation="saved prediction/reference metric replay and local module hashes; no new model fit, feature/source redecoding, physical reference validation or fresh confirmation")
 (R/"strongest_pipeline_replay.json").write_text(json.dumps(axis_audit,indent=2)+"\n")
 quality=read(B/"vehicle_quality_calibration"/"quality_summary.json")
 source=read(B/"vehicle_threat_model"/"evidence_assessment.json")
 comparison=touch(B/"vehicle_comparison_synthesis"/"COMPARISON_DECISION.md")
 requirements={
  "core_software_tests":"passed" if core["exit_code"]==0 else "failed",
  "focused_research_tests":"passed" if all(x["exit_code"]==0 for x in focused.values()) else "failed",
  "package_build":"passed" if pack["exit_code"]==0 and pack["wheel_verified"] else "failed",
  "motion_accuracy_availability_all_retained_groups":"passed" if not failedwindows and not failedblocks and all(result["full_development"]["gates"]["signed_axis_corrected"].values()) else "failed",
  "justified_measurement_trust_and_reference":"not_established",
  "compatible_published_method_comparison_and_novelty":"not_established",
  "complete_declared_threat_case_success":"not_established",
  "frozen_independent_final_confirmation":"not_performed"}
 assert source["authentication_component"]["origin_protection_not_established"] is True
 assert not result["reserved_final_opened"] and not result["promotion_allowed"]
 ready=complete(requirements)
 report=dict(full_project_complete=ready,journal_readiness_established=False,production_policy_changed=False,promotion_allowed=False,
  requirements=requirements,core_tests=dict(passed=core["counts"].get("passed",0),skipped=core["counts"].get("skipped",0),warnings=core["warnings"]),
  focused_tests=sum(x["count"] for x in focused.values()),packaging=pack,strongest_motion_pipeline=axis_audit,
  seven_vehicle_policy_track=dict(nominal_coverage_pct=quality["policy_nominal_coverage_unchanged_pct"],calibration_blocks=quality["calibration_block_count"],calibration_passed=quality["block_quantile"] is not None,
   scope="different dataset/reference and policy track; not interchangeable with NavINST motion-estimator availability"),
  final_payload_access_in_this_audit=False,new_experiments_run=False,source_inputs_sha256=fingerprints,
  next_required_work=[
   dict(id="motion_evidence",action="Develop a reliable independent motion source on separate driving data, with justified reference/source/timing assumptions and enough calibration; pass all retained accuracy, availability and threat conditions",current_blockers=["NavINST retained window/block failures","7V independent-policy coverage below 95%","unestablished physical reference/source trust"],requires_measurement_evidence=True),
   dict(id="comparison",action="Run appropriate published comparators with compatible native inputs and the same split, information, budget and attacks; document the precise algorithmic contribution",current_blockers=["component adaptations and literature screening do not establish complete-method superiority or novelty"]),
   dict(id="confirmation",action="After a valid development method and all parameters are frozen, execute the independently separated final protocol once",current_blockers=["prerequisites failed; reserved final data must remain closed"])])
 (R/"readiness_report.json").write_text(json.dumps(report,indent=2)+"\n")
 text=f"""Project completion audit

Software checks completed: {core["counts"].get("passed",0)} core tests passed, {core["counts"].get("skipped",0)} skipped; {report["focused_tests"]} focused research/completion tests passed. The core warning concerns a legacy checkpoint without per-node residual scales. The package builds from a clean staged source tree without network or dependency installation. A missing declared README was supplied.

Overall project complete: NO. Journal readiness: NOT ESTABLISHED. Software verification is not evidence that the research requirements are met.

The strongest documented radar-IMU motion pipeline was checked again using its saved predictions and reference arrays. Its 15 route windows contain 27,178 rows: pooled P95 discrepancy is{pooled["p95_mps"]:.6f}m/s. On 37,340 scheduled development requests, 36,046 are accepted ({100*36046/37340:.4f}%), with P95 discrepancy{scored["p95_mps"]:.6f}m/s and zero accepted recorded discrepancies above 2 m/s.
These aggregate values do not erase the retained failures: {json.dumps(failedwindows)}; original retained-block result {json.dumps(failedblocks)}. The latter is hash-bound historical evidence, not recomputed in this audit. These are retrospective reference disagreements, not physical speed-error certificates or a new untouched evaluation.
The separate7V policy track remains at 79.151943% nominal coverage. Its recent quality model has 13 one-second calibration blocks and is unbounded under its fixed 95% rank rule. Datasets, estimators and denominators are kept separate.

Remaining work, in dependency order:
1. Establish usable independent motion evidence and its uncertainty/timing/trust assumptions on separate driving data. Repair the actual estimate/observation model, then satisfy the existing accuracy, availability and attack requirements together. Neither extra pass markers nor a rejection-only filter of the failed7V estimates supplies this evidence.
2. Complete a fair published-method comparison and establish the specific method contribution. Existing adaptations and software authentication are not proof of novelty or superiority.
3. Freeze the final protocol only after development requirements pass, then evaluate independent recording/session groups. Do not open reserved final data to tune development failures.

Working software exists and builds. Completing these research requirements needs stronger measurement/method evidence and comparison experiments. This audit does not invent source calibration, relax targets, change a policy, train another candidate, score reserved measurements or write a manuscript.

Recomputed scope: saved route/request prediction metrics, not a full native model/source replay. Prior stage evidence remains intact. The core test runner forbids protected-payload opening and external socket connections; neither was attempted.
"""
 (R/"COMPLETION_REPORT.md").write_text(text)
 print(json.dumps(dict(full_project_complete=ready,core_passed=report["core_tests"]["passed"],core_skipped=report["core_tests"]["skipped"],focused_tests=report["focused_tests"],wheel_verified=pack["wheel_verified"],
  failed_route_windows=failedwindows,failed_retained_blocks=failedblocks,remaining_research_work=len(report["next_required_work"])),indent=2),flush=True)
 return 0 if ready else 2
if __name__=="__main__":sys.exit(main())
