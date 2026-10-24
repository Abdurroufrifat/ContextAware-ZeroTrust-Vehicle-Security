from learned_common import *
install_guard()
tests=json.loads((R/"test_validation.json").read_text())
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==538 and tests["core"]["counts"]["passed"]==731
assert tests["core"]["counts"]["skipped"]==4
application=json.loads((R/"application_replay.json").read_text())
assert application["passed"] and application["trials"]==54 and application["decisions"]==10800
for name,h in application["application_source_sha256"].items():assert sha(P/name)==h
assert json.loads((R/"verification.json").read_text())["controlled_decisions_reconstructed"]==32400
assert json.loads((R/"report_verification.json").read_text())["passed"]
assessment=json.loads((R/"contribution_assessment.json").read_text())
assert assessment["totals"]["learned_shielded"]["selected_with_valid_scene"]==8748
assert assessment["totals"]["learned_shielded"]["errors_gt05_with_valid_scene"]==0
assert assessment["totals"]["learned_shielded"]["errors_gt05"]==292
assert assessment["totals"]["learned_unshielded"]["errors_gt05_with_valid_scene"]==6162
assert assessment["learning_improves_over_original_overall"] is False
bindings=json.loads((R/"dependency_bindings.json").read_text())
for name,h in bindings["source_files"].items():assert sha(P/name)==h
assert not (R/"file_manifest.json").exists(),"Package already frozen"
write("execution_record.json",dict(experiment_trials=162,experiment_decisions=32400,
 application_replay_decisions=10800,application_replays_are_new_driving_evidence=False,
 focused_tests=538,core_tests=731,core_skips=4,review_important_findings_resolved=True,
 rulings=["Existing authorized project; no git commit/push, manuscript or physical operation.",
 "No test tuning; learning did not beat the existing guard overall, which remains the default.",
 "Separate scene-premise scores; only two shared RNG seed clusters, no population inference.",
 "All common-scene-compromise errors retained; no reference, threshold or lag replacement.",
 "Harness import/state-root errors corrected without weakening the restrictive original API.",
 "Model selection/source integrity verified before loading after two real tamper regression tests."],
 deferred_minor="Generic arbitrary models can return float-convertible strings/bools; the public application exposes only frozen sklearn models and exact projection.",
 physical_vehicle_validation=False,journal_acceptance_certified=False))
paths=[q for q in R.rglob("*") if q.is_file() and
 not any(x in ("runtime","__pycache__",".pytest_cache") for x in q.relative_to(R).parts)
 and q.suffix!=".log" and q.name!="file_manifest.json"]
paths.append(P/"LEARNED_VEHICLE_RESULTS.md")
write("file_manifest.json",{str(q.relative_to(P)):sha(q) for q in sorted(paths)})
print("EXPERIMENT_FROZEN",len(paths),flush=True)
