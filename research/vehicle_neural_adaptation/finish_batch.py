from shared import *
import numpy as np
verification=read("independent_verification.json");assert verification["passed"]
raw=read("raw_feature_verification.json");assert raw["passed"]
tests=read("test_validation.json")
assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
review=read("code_review.json");assert review["completed"] and review["unresolved_important_findings"]==0
data=read("data_audit.json");training=read("training_summary.json");evaluation=read("evaluation_summary.json")
groups=["Library01_temporal_test","Library02_development_transfer"]
lines=["# Available-input neural development comparison","",
"2026-10-21. This bounded comparison is complete. The full project remains incomplete.","",
"## Implementation and scope","",
"Implemented an original 3,329-parameter DeepSets speed regressor using all finite 3..100 m radar points, their unit directions, raw Doppler and the same causal IMU/calibration context as the existing estimator. Four features per point; no point subsampling or fabricated RCS/SNR/power. This is an explicitly adapted comparator, not a reproduction of the pretrained RadarNet publication or proof against neural methods generally.","",
"The fixed 564-frame cohort contains 316 Library01 and 248 Library02 attempts. Library01 time boundaries at60%/80%, with2s removed on each side of both, produce188 train attempts/187 known labels,58 validation,61 late temporal-test and9 purged diagnostic frames. One train reference and one transfer reference remain unknown. Existing full-grid four unknown endpoints remain unchanged. Neural training only reads a separate train/validation file; Library02 and late-test labels are loaded only by the evaluator.","",
"Three independently trained seeds17/29/43 use150 fixed epochs, Adam0.001, batch16, Huber delta0.3. Selection minimizes validation MAE with earliest-epoch tie handling. Selected epochs: "+", ".join(str(r["seed"])+":"+str(r["selected_epoch"]) for r in training["runs"])+". No best-seed selection or post-test retuning. All state dictionaries,150-row histories, input/experiment hashes, predictions and bootstrap intervals are retained.","",
"Library02 was previously inspected in development. The later Library01 block was not used for this model's training or epoch selection but also is not a pristine project-level holdout. No unseen-route population claim. Errors below are disagreement with an uncertain pose-derived speed-magnitude proxy, not certified physical speed errors.","",
"## Fixed evaluation results","",
"| Development evaluation set | Model | Attempts/scored/unknown | MAE m/s | P95 m/s | >2 m/s |",
"|---|---|---:|---:|---:|---:|"]
for group in groups:
    result=evaluation["groups"][group]
    for name,m in result["metrics"].items():
        lines.append(f'| {group} | {name} | {m["attempts"]}/{m["scored"]}/{m["unknown_reference"]} | {m["mae_mps"]:.6f} | {m["p95_mps"]:.6f} | {m["above2"]} |')
lines+=["",
"The estimator has lower matched MAE than each frozen neural run on both evaluation sets. Seed variability is retained. This supports this particular available-input, small-data development comparison; it is not a state-of-the-art or full-security superiority claim. The author classical3D RANSAC has slightly lower temporal-test P95 than the estimator; no across-metric dominance is claimed.",
"",
"The estimator's61-frame temporal-test P95 is0.5602207093211773m/s, so the existing0.5 condition is not met there. The original full-query Library01 final302-query result (301 scores; P951.102297507859495m/s) also remains failed. The sparse16-frame cohort inside that original interval is reported separately in evaluation_summary.json and is never substituted for the full original query grid. Neural sparse-window P95 also exceeds0.5 for every seed. Neither adding a neural baseline nor changing the evaluation grain resolves that failure.",
"",
"Candidate-minus-neural MAE paired circular-block intervals use5 cohort frames/1000 draws with frozen seeds, conditional on the recorded sequence. They are not independent-route significance evidence. The training median sanity comparator, every seed, all purged rows, original unknowns and >2 counts are visible in matched_metrics.csv/predictions.npz.",
"",
"## Timing and verification","",
"Training used CPU only, two threads; total training seconds: "+str(training["total_training_seconds"])+". Feature preparation P95: "+str(data["feature_preparation_p95_ms"])+"ms, excluding archive input. Neural single-frame forward-only timings use one thread after the test suite completed; tensor padding timing is separately recorded. No sensor-to-policy or physical-vehicle deadline claim.",
""]
for name,t in evaluation["timing"].items():
    lines.append(f'- {name}: forward-only P95 {t["forward_only_p95_ms"]:.6f}ms; padding/tensor P95 {t["padding_tensor_p95_ms"]:.6f}ms.')
lines+=["",
"Independent verification reconstructs every model prediction with NumPy rather than the production model class; recomputes all metrics and bootstrap endpoints; checks all split boundaries, validation minima, checkpoint/source identities and prior results. Six fresh raw-frame feature oracles cover both sequences and do not import the production feature function.",
"",
str(tests["core"]["counts"]["passed"])+" core tests and "+str(tests["focused"]["counts"]["passed"])+" focused tests pass; "+str(tests["core"]["counts"].get("skipped",0))+" core tests are skipped. The existing legacy per-node calibration warning remains. No protected measurement payloads were opened, no gates/references changed and no physical witness promoted.",
"",
"## Remaining project work","",
"Numerical comparison with this available-channel neural adaptation is now complete. Native pretrained-paper comparison still needs genuine input/target compatibility; current missing-channel/encoding restrictions remain.",
"",
"Priority is the unresolved physical/reference and common-mode-security evidence: the original failed window, source/clock authenticity and bounds, coherent both-source bypasses, explicitly scoped novelty and justified independent final confirmation. A learned regressor does not authenticate sources or prove zero trust. Existing development conditions stay unchanged. These cannot be marked complete merely because software tests pass.",
"",
"No manuscript, human-review workflow, installation, commit, push or deployment was performed. Artifacts are local under research/vehicle_neural_adaptation."
]
(R/"FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
decision={"batch_complete":True,"available_input_neural_comparison_complete":True,"native_pretrained_paper_reproduction":False,
 "neural_seeds_reported":[17,29,43],"all_attempts_retained":564,"new_model_test_counts":{"temporal":61,"transfer":248,"transfer_scored":247},
 "candidate_lower_mae_than_all_three_fixed_runs_both_development_sets":True,"candidate_temporal_p95_condition_met":False,
 "original_failed_window_resolved":False,"reference_gap_resolved":False,"final_holdout_used":False,
 "overall_project_complete":False,"physical_witness_admitted":False,"production_changed":False}
write("RESEARCH_DECISION.json",decision)
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_ADAPTATION.md"
assert sha(backup)==data["old_hashes"][str(status)]
prefix="""Latest: available-input neural development comparison completed,2026-10-21.

Implemented original3,329-parameter DeepSets comparator using present geometry/Doppler and same causal IMU context; no missing channel fabrication. Three frozen seeds trained only on187 known Library01 training labels and selected on58 validation labels. Evaluation:61 late Library01 rows and248 Library02 development-transfer attempts/247 scores. Previously inspected sequences; not pristine final holdouts.
Matched MAE m/s: candidate0.175575 vs neural0.291237..0.392203 on late Library01; candidate0.076025 vs neural0.197434..0.417896 on Library02. All seeds/counts/intervals retained. This is an adapted small-data comparator, not native-paper or general neural superiority.
Candidate late Library01 P950.560221 exceeds the unchanged0.5 condition. Author RANSAC temporal P950.541123 is slightly lower. Original302-query/301-score P951.1022975 failure and all original unknowns remain. Reference/source/clock uncertainty and coherent-source attack limitations unresolved.
Independent NumPy prediction replay, raw feature oracles, split/selection/metric/source audits pass.731core/110focused tests pass,4core skips, legacy calibration warning retained.
Next research priority: resolve or explicitly bound physical/reference and common-mode/source-clock evidence, state specific novelty and justify independent final confirmation. Native pretrained-input compatibility remains unverified. Overall project complete:NO; physical witness disabled; reserved payloads remain closed.
Evidence: research/vehicle_neural_adaptation/FINDINGS.md, matched_metrics.csv, evaluation_summary.json, RESEARCH_DECISION.json, independent_verification.json.

Previous verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
progress=R/"PLAN_AND_PROGRESS.md"
progress.write_text(progress.read_text(encoding="utf-8").replace("- [ ]","- [x]"),encoding="utf-8")
with progress.open("a",encoding="utf-8") as f:
    f.write("\nAll planned steps completed:18 RED-to-GREEN tests; frozen preparation/training/evaluation; three seeds; independent NumPy/metric and raw-feature verification;110focused/731core passes,4skips; final code review recorded. No implementation changes after outcomes. Original failures and all reserved-data restrictions retained.\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}}
write("file_manifest.json",manifest)
write("completion.json",{"batch_complete":True,"overall_project_complete":False,"top_level_files":len(manifest),
 "neural_seeds":3,"focused_passed":tests["focused"]["counts"]["passed"],"core_passed":tests["core"]["counts"]["passed"],
 "core_skipped":tests["core"]["counts"].get("skipped",0),"current_status_sha256":sha(status)})
print(json.dumps(decision,indent=2))
