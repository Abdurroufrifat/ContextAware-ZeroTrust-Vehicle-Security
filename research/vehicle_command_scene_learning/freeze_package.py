from cs_common import *
install_guard()
assert not (R/"file_manifest.json").exists(),"Already frozen"
tests=json.loads((R/"test_validation.json").read_text());assert all(v["exit_code"]==0 for v in tests.values())
assert tests["focused"]["counts"]["passed"]==572 and tests["core"]["counts"]["passed"]==731
a=json.loads((R/"application_replay.json").read_text());assert a["passed"] and a["decisions"]==10800
for name,h in a["source_bindings"].items():assert sha(P/name)==h
assert json.loads((R/"verification.json").read_text())["controlled_decisions_reconstructed"]==48600
assert json.loads((R/"report_verification.json").read_text())["passed"]
assessment=json.loads((R/"contribution_assessment.json").read_text())
assert all(assessment[k] for k in ("learned_better_original_nominal","learned_better_midpoint_nominal","learned_better_original_all_cases","learned_better_midpoint_all_cases"))
assert assessment["totals"]["refined_learned"]["valid_certificate_gt05"]==0
assert assessment["totals"]["refined_learned"]["errors_gt05"]==454
write("execution_record.json",dict(experiment_trials=243,experiment_decisions=48600,application_replay_decisions=10800,
 focused_tests=572,core_tests=731,core_skips=4,original_experiment_unchanged=True,
 specific_conditional_proof_and_learning_improvement_supported=True,
 physical_vehicle_control_validation=False,global_first_ever_claim=False,complete_call_compliance_certified=False,
 rulings=["User authorized direct work in existing D:/CO-ACT-CPS; no git commit/push, manuscript or real-vehicle operation.",
 "Initial development models corrected before fresh tests for elevated-risk feature exclusion and joint certificate labels; originals preserved.",
 "Scene-only certificate kept separate from refined scene-and-command execution certificate.",
 "Model/train selection frozen before new profiles/seeds; all counterexamples preserved; no test tuning.",
 "Three shared RNG seed clusters yield descriptive fixed-grid evidence, not a population confidence interval.",
 "Public-source search found candidate physical datasets but no fully qualified scene-command/error-bound interface. Do not manufacture a physical claim."]))
files=[p for p in R.rglob("*") if p.is_file() and not any(x in ("runtime","__pycache__",".pytest_cache") for x in p.relative_to(R).parts) and p.suffix!=".log" and p.name!="file_manifest.json"]
files.append(P/"COMMAND_SCENE_RESULTS.md")
write("file_manifest.json",{str(p.relative_to(P)):sha(p) for p in sorted(files)})
print("COMMAND_SCENE_EXPERIMENT_FROZEN",len(files),flush=True)
