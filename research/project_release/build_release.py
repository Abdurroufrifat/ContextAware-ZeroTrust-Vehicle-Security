from release_common import *
install_guard()
import importlib.metadata as md,platform
def read(relative):return json.loads(checked(P/relative).read_text(encoding="utf-8"))
tests=read("research/project_release/test_validation.json");app=read("research/project_release/application_replay.json")
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0 and app["passed"] and app["decisions"]==6800
canonical=read("research/vehicle_development_release/release_summary.json")
boreas=read("research/vehicle_boreas_rt_reference_study/contribution_assessment.json")
reference=read("research/vehicle_boreas_rt_reference_study/reference_qualification.json")
gateway=read("research/vehicle_command_gateway/summary.json")
sumo=read("research/vehicle_sumo_gateway_study/results.json")
secure=read("research/vehicle_secure_set_comparison/contribution_assessment.json")
temporal=read("research/vehicle_temporal_gateway_study/contribution_assessment.json")
scene=read("research/vehicle_kinematic_witness_study/contribution_assessment.json")
call=read("research/vehicle_sumo_gateway_study/call_alignment.json")
sources=[
 "pyproject.toml","src/coact_cps/cli.py","research/project_completion/COMPLETION_REPORT.md",
 "research/vehicle_development_release/release_summary.json","research/vehicle_development_release/FINAL_READINESS.json",
 "research/vehicle_development_release/file_manifest.json",
 "research/vehicle_boreas_rt_reference_study/contribution_assessment.json","research/vehicle_boreas_rt_reference_study/reference_qualification.json",
 "research/vehicle_boreas_rt_reference_study/file_manifest.json",
 "research/vehicle_command_gateway/summary.json","research/vehicle_command_gateway/file_manifest.json",
 "research/vehicle_sumo_gateway_study/results.json","research/vehicle_sumo_gateway_study/file_manifest.json",
 "research/vehicle_sumo_gateway_study/call_alignment.json","research/vehicle_sumo_gateway_study/published_comparison_scope.json",
 "research/vehicle_secure_set_comparison/contribution_assessment.json","research/vehicle_secure_set_comparison/source_ledger.json",
 "research/vehicle_secure_set_comparison/file_manifest.json",
 "research/vehicle_temporal_gateway_study/contribution_assessment.json","research/vehicle_temporal_gateway_study/file_manifest.json",
 "research/vehicle_kinematic_witness_study/contribution_assessment.json","research/vehicle_kinematic_witness_study/aggregate_results.json",
 "research/vehicle_kinematic_witness_study/CONTRACT.md","research/vehicle_kinematic_witness_study/file_manifest.json",
 "research/project_release/application_replay.json","research/project_release/test_validation.json",
 "research/project_release/research_session.py","research/project_release/test_research_session.py"]
entries=[
 dict(id="core",kind="Existing AI incident-diagnosis software",scope="Original core code and configured test suite; source-tree CLI retained",
  metrics=dict(core_passed=tests["core"]["counts"]["passed"],core_skipped=tests["core"]["counts"].get("skipped",0)),
  limitation="Not a new end-to-end learned autonomous-vehicle experiment"),
 dict(id="recorded_development",kind="Recorded-data development evidence",metrics=canonical,
  limitation="Original HeRCULES last-window and temporal neural-comparison failures retained; physical reference error unqualified"),
 dict(id="boreas",kind="Separate recorded-reference comparison",metrics=dict(frames=reference["physical_frames"],paired_cells=boreas["paired_cells"],improved_cells=boreas["improved_cells"],worse_cells=boreas["worse_cells"]),
  limitation="Variant hypothesis rejected; separate instrumentation does not establish hard reference accuracy or physical control"),
 dict(id="signed_gateway",kind="Synthetic cryptographic/numeric interface matrix",metrics=gateway["totals"],
  limitation="Signed source/history lies and database snapshot rollback remain retained counterexamples"),
 dict(id="sumo_components",kind="Software-in-the-loop",metrics=dict(controlled_trials=sumo["controlled_trials"],decisions=sumo["controlled_queries"],separate_native_idm_references=sumo["native_reference_trials"]),
  limitation="Static/clock component ablations and separate privileged native IDM references are not full recent security frameworks"),
 dict(id="s3e_scalar",kind="Offline published-method scalar adaptation comparison",metrics=dict(existing_trajectories=144,new_estimator_decisions=57600),
  limitation="Different source/initialization contracts; reflected scalar adaptation rather than full constrained-zonotope reproduction"),
 dict(id="temporal",kind="Fresh paired software-in-the-loop",metrics=dict(trials=temporal["fresh_trials"],decisions=temporal["fresh_decisions"],drift_errors_gt05=temporal["gradual_joint_selected_error_gt05"]),
  limitation="Abrupt attack rejection loses availability; gradual drift remains a demonstrated failure"),
 dict(id="scene_witness",kind="Fresh paired software-in-the-loop",metrics=dict(trials=scene["fresh_trials"],decisions=scene["fresh_decisions"],valid_scene_selected=scene["selected_with_valid_scene"],valid_scene_errors_gt05=scene["selected_error_gt05_with_valid_scene"],common_compromise_errors_gt05=scene["scene_common_compromise_selected_error_gt05"]),
  limitation="Assumes independently trustworthy scene range/absolute lead speed and aligned clocks; common scene compromise defeats policy"),
 dict(id="application",kind="Integration replay of existing records",metrics=dict(trajectories=app["trajectories"],decisions=app["decisions"],focused_tests_passed=tests["focused"]["counts"]["passed"]),
  limitation="Exact application integration check, not new independent driving evidence")]
index=dict(snapshot_date="2026-10-23",entries=entries,performance_metrics_pooled=False,
 source_bindings=[dict(path=n,bytes=(P/n).stat().st_size,sha256=sha(P/n)) for n in sources])
dump(R/"evidence_index.json",index)
status=dict(project="CO-ACT-CPS",snapshot_date="2026-10-23",
 completed_scope="Unified offline research application and verification of the documented frozen PC studies",
 application_implementation_complete=True,frozen_pc_evidence_scope_verified=True,
 application_replay_trajectories=app["trajectories"],application_replay_decisions=app["decisions"],
 focused_tests_passed=tests["focused"]["counts"]["passed"],core_tests_passed=tests["core"]["counts"]["passed"],core_tests_skipped=tests["core"]["counts"].get("skipped",0),
 specific_conditional_integration_result_supported=True,
 full_original_research_objectives_completed=False,
 independent_physical_reference_error_bound_qualified=reference["hard_reference_error_bound_established"],
 physical_vehicle_control_validation_complete=False,physical_scene_independence_established=scene["physical_scene_independence_established"],
 complete_recent_av_security_framework_natively_reproduced=False,
 new_algorithm_principle_or_new_learned_AV_model_established=False,human_effectiveness_established=False,
 complete_special_issue_compliance_certified=False,journal_readiness_certified=False,production_promotion_allowed=False,
 next_constraint="Scientific claims must stay within measured development/proxy/simulation evidence; missing physical/source qualification cannot be supplied by additional software pass markers",
 entry_point="COACT_CPS_RESEARCH.py",evidence_index="research/project_release/evidence_index.json")
dump(R/"completion_status.json",status)
claims=[
 dict(claim="Signed/context-bound report admission and explicit source/clock/error contracts are implemented",supported=True,evidence=["signed_gateway","application"],limit="No measurement-truth or durable anti-rollback guarantee"),
 dict(claim="Scene-backed admission preserves 99.5% selection in tested nominal and wheel/anchor drift groups",supported=True,evidence=["scene_witness"],limit="12,330 qualifying selected points are conditional on scene inputs/motion premises"),
 dict(claim="This policy is safe under compromise of every scene/speed domain",supported=False,evidence=["scene_witness"],limit="314 selected errors above0.5m/s in common-compromise cases"),
 dict(claim="Recorded-data predictors satisfy a certified absolute physical speed bound",supported=False,evidence=["recorded_development","boreas"],limit="Reference/source/clock qualifications and retained failed intervals remain"),
 dict(claim="A 2026 published S3E method has an executable numerical scalar comparator",supported=True,evidence=["s3e_scalar"],limit="Full method/framework and equal trust assumptions are not claimed"),
 dict(claim="A new learned AI model has been validated end-to-end in SUMO",supported=False,evidence=["core","scene_witness"],limit="Candidate reports are generated test inputs; original AI diagnosis track remains separate"),
 dict(claim="Human-centered effectiveness or journal novelty has been established by these tests",supported=False,evidence=["core","application"],limit="Software tests and machine explanations do not establish human effectiveness or global novelty")]
dump(R/"claim_matrix.json",claims)
requirements=[
 dict(id="physical_claims",status="unestablished",needed="Independent source/clock/calibration and absolute-reference error evidence plus justified untouched confirmation protocol",scope_option="Use explicit recorded-proxy and software-in-the-loop claims; physical testing has not been verified as mandatory in the call"),
 dict(id="novelty_scope",status="conditional_integration_result_demonstrated_global_novelty_unestablished",needed="Keep the concrete integration result and precise published-method reproduction boundaries; no global-first or new-principle claim"),
 dict(id="call_alignment",status="full_text_unavailable",needed="Official full call returned403; title/journal/indexed dates alone cannot establish every special-issue requirement"),
 dict(id="security_limit",status="retained_counterexamples",needed="No claim of defense when all trusted measurement domains are forged, or durable anti-rollback/restart protection")]
dump(R/"remaining_research_requirements.json",requirements)
call_record=dict(checked_date="2026-10-23",official_url=call["official_call_url"],journal="Engineering Applications of Artificial Intelligence",
 indexed_title=call["official_indexed_title"],full_call_text_retrieved=False,access_result="403 on fresh official-page request",
 indexed_deadline_from_existing_publisher_index_audit=call["official_indexed_submission_deadline"],
 human_study_mandatory_requirement_verified=False,physical_vehicle_test_mandatory_requirement_verified=False,
 full_call_compliance_established=False)
dump(R/"call_alignment.json",call_record)
versions={}
for name in ("numpy","pandas","torch","scikit-learn","pytest","traci","joblib"):
 try:versions[name]=md.version(name)
 except md.PackageNotFoundError:versions[name]=None
dump(R/"environment.json",dict(python=sys.version,platform=platform.platform(),packages=versions,
 executable=sys.executable,network_required_for_verification=False,gpu_required=False,
 source_tree_workflow=True,updated_vehicle_api_packaged_as_pip_wheel=False))
lines=["# CO-ACT-CPS consolidated research release","23 October 2026.","",
"The unified offline application and the documented PC evidence checks are complete. Full original research validation remains incomplete. The completion fields in completion_status.json distinguish these scopes.","",
"Run the root COACT_CPS_RESEARCH.py command. status prints the saved scope; verify runs the full evidence chain and application demo; demo replays a signed gradual-drift simulation case; core --help exposes the original AI incident-diagnosis CLI. The existing environment is D:/CO-ACT-CPS/.venv. All code and outputs remain under D:/CO-ACT-CPS.","",
"The ResearchSession API consumes signed packets, receiver time and risk context. It owns source/command authentication and the selected temporal or scene policy. It rejects an injected approved decision, duplicate query time, duplicate signing keys, state overwrites and physical-operation fields. Fifteen new boundary/interface tests and 6,800 exact decisions from 34 existing trajectories validate this integration.","",
"## Evidence index","",
"| Track | Evidence type | Key limit |","|---|---|---|"]
for e in entries:lines.append(f'| {e["id"]} | {e["kind"]} | {e["limitation"]} |')
lines+=["","Do not pool the studies' performance values. Repeated baseline runs, offline comparator predictions and application replays are not independent new driving trials.","",
"## Current result","",
"The scene-witness policy selects597/600 (99.5%) estimates in each tested 50/90ms nominal, abrupt speed-source attack and gradual-drift group, with zero selected errors above0.5m/s. The prior temporal policy admits153 errors above0.5m/s in each drift group. With scene data forged too, the new policy admits157 such errors per delay group. Scene dropout/bad tags reduce availability. Both benefit and failure remain visible.","",
"Across the full latest study,12,330 selected points have appraised valid scene/motion premises and no error above0.5m/s. This is a conditional simulation result. It assumes correctly associated, independently trustworthy absolute lead speed, bounded range and aligned times; real lead speed derived from the compromised ego speed can make the check circular.","",
"## Original and new application scope","",
"The original coact_cps diagnosis software remains available through core. It is not replaced by the vehicle admission logic. The shared root command is a workflow integration, not evidence of a newly trained end-to-end autonomous-driving AI model.","",
"Tests:510 focused passed;731 configured core passed;4 core skipped. The legacy calibration warning remains. The 34-trajectory application replay reproduces6,800 decisions exactly. The complete latest independent simulation replay remains28,800 decisions/144trials.","",
"## Completion boundary","",
"Physical control/reference/source qualification, full native recent AV-security framework reproduction, global algorithm novelty, human effectiveness and complete special-issue compliance are unestablished. The official call page again returned403; no mandatory physical test or human-study requirement was inferred from that error. No original threshold, heldout restriction or failed measurement result was relaxed.","",
"See claim_matrix.json, remaining_research_requirements.json and call_alignment.json. The current bounded integration result can be described precisely; broader claims require evidence that software cannot manufacture.","",
"The source-tree workflow is verified; no new installable wheel for the vehicle API is claimed. Dataset/model licenses remain separate from code licensing. No manuscript, commit, push or physical operation is included in this release.",""]
(R/"README.md").write_text("\n".join(lines),encoding="utf-8")
supervisor="""# Supervisor progress summary
The project now has one runnable research entry point for its original AI incident-diagnosis code and the vehicle-security extension. The vehicle API authenticates signed sensor/command reports, checks context and timing, and uses bounded motion or scene evidence to decide which speed estimate to accept.

The latest simulation study ran144 paired trials with28,800 decisions. With a trustworthy scene range/absolute-lead-speed channel, selection availability is99.5% in both tested delay groups under nominal inputs and coordinated speed-source drift, with no selected error above0.5m/s. The previous temporal policy accepted153 such drift errors per delay group. Forging the scene channel too still defeats the new method; those failures are retained.

The unified application reproduces6,800 saved decisions across34 trajectories. There are510 passing focused tests and731 passing core tests, with4 skips. Recorded HeRCULES and Boreas studies, their failed intervals/candidates and the published-method comparison retain their own cohorts and assumptions.

The implemented contribution is a reproducible integration of authenticated context-aware admission and bounded scene evidence, with measured benefits and explicit counterexamples. Physical sensor/reference guarantees, an end-to-end new learned vehicle model and human-effectiveness evidence are not established. These results support a conditional software-in-the-loop study; they do not establish vehicle deployment safety or journal acceptance.
"""
(R/"SUPERVISOR_SUMMARY.md").write_text(supervisor,encoding="utf-8")
print(json.dumps(status,indent=2))
