from common import *
import csv
test=json.loads((R/"test_validation.json").read_text())
verification=json.loads((R/"independent_verification.log").read_text(encoding="utf-16"))
assert verification["passed"] and all(test[name]["exit_code"]==0 for name in ("focused","core"))
assert test["focused"]["counts"]["passed"]==198 and test["core"]["counts"]["passed"]==731
write("independent_verification.json",verification)
data=json.loads((R/"results.json").read_text())
lines=["# Equal-input motion-component experiment","",
"Completed 21 October 2026. This is a fixed development ablation, not a manuscript or an exact reproduction of a published security framework.","",
"All four estimators received the same radar points, original Doppler and Xsens/calibration transform. Exactly564original frame identities were retained:316Library01 and248Library02. All four produced a numerical estimate on every row. The two missing reference labels remain unknown. No quality or outcome-based exclusion was used.","",
"The experiment separates two component choices: ordinary least squares versus the existing20-iteration Huber fit, and two horizontal velocity components versus three free components. The planar control imposes zero vertical velocity; sharing inputs does not remove that modeling assumption. Huber delta0.3 and all original data conventions were fixed before this run.","",
"## Recorded-reference disagreement","",
"| Development population | Method | Scored/attempts | MAE m/s | P95 m/s | >2 m/s |",
"| --- | --- | ---: | ---: | ---: | ---: |"]
for row in data["metrics"]:
    lines.append(f'| {row["group"]} | {row["method"]} | {row["scored"]}/{row["attempts"]} | {row["mae_mps"]:.6f} | {row["p95_mps"]:.6f} | {row["above2"]} |')
lines+=["",
"The61-frame temporal population is a subset of Library01, not a third independent recording. Library02 is previously inspected development transfer, not an untouched holdout. The table describes disagreement with the existing uncertain pose-derived reference, not known physical true error.","",
"## What this changes","",
"Huber reweighting reduced recorded MAE relative to ordinary least squares for both dimensional choices in all three displayed populations. The planar constraint gives only a small MAE difference between the two robust fits on the full sparse cohorts: planar0.098529 versus spatial0.100212m/s for Library01; planar0.076025 versus spatial0.078017m/s for Library02.",
"",
"In the temporal subset, spatial Huber has slightly lower MAE and P95 than planar Huber:0.174633/0.548746 versus0.175575/0.560221m/s. Both P95 values still exceed the unchanged0.5 condition. No replacement method was selected from these outcomes. The prior candidate remains unchanged.",
"",
"The comparison supports a descriptive effect of robust fitting on this cohort. It does not establish that the planar constraint is uniformly better, that this standard fitting method is novel, or that the whole AV security framework is superior.",
"",
"Spatial designs have normal-matrix condition>100 on196/316Library01 and128/248Library02 rows; all are retained. Planar designs have no such rows. This geometric diagnostic helps describe the model assumption but is not a physical-error certificate or a post-hoc rejection rule. Fixed20IRLS outputs and stationarity diagnostics are preserved; no extra convergence iterations were added.",
"",
"## Integrity and verification","",
"- Recomputed the historical planar candidate exactly on all564frames (maximum difference0). Original unrotated3DOLS was separately reproduced within1e-10m/s.",
f'- New transformed spatialOLS differs from historical unrotatedOLS by at most{data["transformed_spatial_ols_max_difference_from_historical"]:.12g}m/s. Rounded calibration transforms were not forced to be perfectly orthogonal.',
f'- Independent SciPy solver replay checked all2,256predictions; maximum difference{verification["max_solver_difference_mps"]:.12g}m/s.',
"- Six separate struct/quaternion raw-feature oracles and108scalar metric checks pass; all9paired cells and31input hashes were checked.",
"- Fourteen new behavioral tests failed before the fitting component existed and now pass. Combined checks:198focused and731configured core tests pass;4core tests skipped. Existing warning retained: legacy checkpoint lacks per-node residual calibration and uses unit scales.",
"- Automated scoped code review found no important actionable defect. Its limitations are saved in code_review.json.",
"",
"Reproduce calculations without rerunning the experiment: run verify_ablation.py with the existing project Python. run_ablation.py refuses to overwrite frozen results. Files and parameters are in PROTOCOL.md and input_lock.json.",
"",
"## Limits and remaining requirements","",
"The original full Library01 final302-query/301-score P951.102297507859495m/s failure is unchanged. This sparse cohort never replaces it. The prior external64large reference disagreements, source/clock/calibration gaps and coordinated-compromise counterexamples remain.",
"",
"This is an estimator-only comparison. Numerical output fraction is not policy availability, physical security or a physical timing result. No sensor-to-policy latency was measured. The methods share inputs but do not have identical computation costs.",
"",
"Specific novelty, a reproduced recent complete AV security framework, independently qualified physical reference and justified final confirmation remain unresolved. More tests of the same fitting alternatives cannot establish those missing facts. The project can support an explicitly limited offline component study; journal readiness remains unestablished.",
"",
"All prior inputs, original gates, current status, canonical release and production policies are unchanged. Reserved payloads stayed closed. No manuscript, external message, commit, push, installation or promotion was performed.",
"",
"Files: metrics.csv; paired_mae.csv; frames_and_predictions.npz; results.json; independent_verification.json; test_validation.json; code_review.json; input_lock.json; file_manifest.json."]
(R/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
assert all(sha(P/name)==digest for name,digest in json.loads((R/"input_lock.json").read_text()).items())
write("completion.json",{"development_ablation_complete":True,"attempts":564,"solver_predictions_verified":2256,
 "focused_passed":198,"core_passed":731,"core_skipped":4,"candidate_unchanged":True,"physical_validation_established":False,
 "novelty_established":False,"journal_ready":False,"production_changed":False,"reserved_payloads_opened":False})
manifest={path.name:sha(path) for path in sorted(R.iterdir()) if path.is_file() and path.name!="file_manifest.json"}
write("file_manifest.json",manifest)
assert all(sha(R/name)==digest for name,digest in manifest.items())
print(json.dumps({"completion":json.loads((R/"completion.json").read_text()),"package_files_verified":len(manifest)},indent=2))
