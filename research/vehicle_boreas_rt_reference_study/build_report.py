from common import *
install_guard()
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=json.loads((R/"results.json").read_text());v=json.loads((R/"verification.json").read_text());tests=json.loads((R/"test_validation.json").read_text())
assert v["passed"] and tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
pairs=r["paired"];improvements=sum(x["huber6_minus_huber3_mean_abs_error_mps"]<0 for x in pairs)
qualification={"dataset":"Boreas-RT","development_sequences":json.loads((R/"membership.json").read_text())["sequences"],
"physical_frames":96,"sparse_frames_per_sequence":32,"selected_timestamp_span_per_sequence_seconds":62,"acquired_bytes":258427140,
"raw_predictor_inputs":["Aeva XYZ/radial velocity/relative point time"],"reference_inputs":["post-processed GNSS","Applanix IMU","wheel encoder"],
"reference_file":"applanix/aeva_poses.csv","reference_target":"velocity of the Aeva sensor origin resolved in ENU; evaluate vector norm",
"predictor_raw_sensor_used_in_reference_generation_as_documented":False,"documented_separate_reference_instrument":True,
"statistical_error_independence_established":False,"calibration_and_common_clock_dependencies":True,
"point_time_nanoseconds_mapping":"inferred from raw ranges and documented 10 Hz rate; not explicitly confirmed by reviewed format documentation",
"hard_reference_error_bound_established":False,"physical_source_attestation_established":False,
"uniform_05mps_true_error_certificate_established":False,"physical_control_validation_established":False,
"independent_recorded_reference_comparison_completed":r["independent_physical_reference_comparison_completed"],"promotion_allowed":False}
write("reference_qualification.json",qualification)
assessment={"hypothesis":"Local within-scan linear trend improves over same-input constant Huber fit at scan-end and0/50/100ms post-scan delays",
"paired_cells":9,"improved_cells":improvements,"worse_cells":sum(x["huber6_minus_huber3_mean_abs_error_mps"]>0 for x in pairs),
"hypothesis_supported":improvements==9,"local_variant_promoted":False,
"specific_novelty_established":False,"reason":"Timing-aware principle has prior art; the fixed local variant is worse in all 9 paired cells.",
"defensible_current_contribution":"An auditable implementation and fixed external recorded-reference stress comparison with explicit uncertainty limits and retained failures; original software artifact, not a proven novel algorithm.",
"next_research_requirement":"A distinct method and falsifiable advantage over appropriate published methods; present data do not establish it."}
write("contribution_assessment.json",assessment)
table=["| Recording | Method | MAE at scan end (m/s) | P95 (m/s) | Disagreements >2m/s /32 |","|---|---|---:|---:|---:|"]
for seq in qualification["development_sequences"]:
 for m in [x for x in r["metrics"] if x["sequence"]==seq and x["age_us"]==0]:
  table.append(f'| {seq} | {m["method"]} | {m["mae_mps"]:.6f} | {m["p95_mps"]:.6f} | {m["above_2_count"]} |')
text="""# External physical-recording comparison and contribution test

## Outcome
Added 96 complete physical Aeva scans from three fixed public Boreas-RT development recordings, plus separate-instrument GNSS/INS reference, calibration and author error-plot files.288 reference-free fits yield 864 frame/method/query comparisons. Independent verification reconstructs all fits, comparisons, 27 metric cells and 9 paired cells.

The tested timing-aware six-parameter variant is worse in mean absolute reference disagreement in all 9 paired sequence/delay cells. It is rejected as a claimed improvement. Prior continuous-time Doppler estimation, acceleration constraints and uncertainty fusion also overlap the broad idea. A demonstrated novel algorithm has not been established. No failed result was replaced with a success claim.

## Reference qualification
The author's documented reference generation uses GNSS, Applanix IMU and wheel encoder; this estimator uses Aeva points only. The supplied sensor-origin aeva_poses.csv provides an external recorded measurement comparison without using its velocities in the estimator. Shared calibration/clock dependencies mean statistical independence is not proven. Author gt_errors.pdf files are self-reported error summaries, not hard uniform speed-error bounds. See reference_qualification.json and primary URLs in novelty_sources.json.

The binary point-time unit was not explicit in the reviewed public format description. Nanoseconds are inferred from raw offset ranges and the documented 10 Hz rate; SCHEMA_ADDENDUM.md records that inference before fitting. All scan-end/delay results are conditional on this mapping. This study does not establish physical 100 ms deadlines or a true-speed certificate.

The age_us field denotes delay after scan end. It is not the maximum age of every raw return: a full scan plus 100 ms delay can contain points nearly 200 ms old. The existing 100 ms gateway condition was not relaxed or applied as a physical certificate here.

## Sampling and methods
Three metadata-selected recordings, each 32 full scans spaced 2 s apart starting 120 s after its first Aeva frame; each sparse sample spans 62 s. This is neither complete-sequence validation nor an independent set of 864 driving trials. Total acquisition is 258,427,140 bytes across 132 objects. Official Boreas-RT odometry/localization test-list sequences were excluded; protected project holdouts remain unopened.

All three local fits use identical eligible point IDs: finite 8 measurement fields, 3..100 m range, deterministic cap 10,000 points; 20 IRLS passes and 0.3 m/s Huber loss threshold where used. The Huber threshold is not a calibrated error bound. OLS3/Huber3 assume constant sensor-frame velocity; Huber6time fits a linear sensor-frame trend and can amplify noise or poor geometry. It is an adaptation of an existing model family, not reproduction of a full published odometry/security system.

Query delays are 0, 50, 100 ms. Evaluation interpolates the external ENU velocity vector and then takes its norm; prediction files were frozen before any reference scoring. No alignment/lag search or reference-based tuning was performed.

## Recorded failures
At scan end, Huber3 P95 disagreement is 0.613777, 0.881181 and 42.814185 m/s for the three recordings. These component results fail the earlier 0.5 m/s study condition. The third recording also contains many >2 m/s disagreements, indicating that unrestricted 3D fitting does not generalize adequately here. All 32 frames remain in each cell's denominator. No security-admission or real vehicle-control success rate is claimed.

"""+'\n'.join(table)+"""

## Verification and corrections
32 new tests and 390 existing focused tests pass: 422 total. The core suite passes 731 with four skips and the existing legacy checkpoint residual-calibration warning.
The independent verifier uses struct decoding, SciPy QR-based least squares and csv/bisect reference interpolation, without importing boreas_core. It hashes all 132 payloads and checks all 864 comparisons and paired membership. Maximum fitted-coefficient difference is below 1e-12.

pre_review preserves the first complete predictions/results/code and reproduced review failures. Scan timing was separated from feature eligibility, and missing/hash-invalid/corrupt frames now yield explicit rejected rows. Reference integrity/schema failure yields unsupported scoring rows. Empty/singleton reference files are rejected through a validated parser; completion requires complete inventory, supported reference rows and actual scored results. pre_reference_io_fix preserves the earlier evaluator and all numerical rows, which are unchanged. Predictor wording was corrected: the earlier version hashed reference bytes for provenance but did not parse reference values; the final predictor reads only frame payloads. The review corrections change none of the 288 actual prediction rows.

pre_module_rename preserves the intermediate module-name collision found by the cumulative test run. New files use boreas_core.py/test_boreas_core.py to avoid importing the older reference-qualification module during combined tests. No original module was changed.

## Files and recheck
- PROTOCOL.md, SCHEMA_ADDENDUM.md, membership.json: pre-run design, schema inference and 96 frame identities.
- data_lock.json: all 132 downloaded object sizes/hashes.
- predictions.json and prediction_freeze.json: 288 fits frozen before scoring.
- comparisons.csv,metrics.csv,paired.csv: 864 comparisons, 27 summary cells and 9 matched cells.
- reference_qualification.json,contribution_assessment.json,novelty_sources.json: evidence/novelty boundaries.
- reference_comparison.png: per-recording mean disagreement versus post-scan delay.
- verify_study.py: independent scientific replay.
- verify_package.py and run_all.py: package and inherited release checks.
- All data and results are development evidence; all physical/control promotion flags remain false. No manuscript, submission, commit or push was produced.
"""
(R/"README.md").write_text(text,encoding="utf-8")
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for ax,seq in zip(axes,qualification["development_sequences"]):
 for method in ["ols3","huber3","huber6time"]:
  cells=sorted([x for x in r["metrics"] if x["sequence"]==seq and x["method"]==method],key=lambda x:x["age_us"])
  ax.plot([x["age_us"]/1000 for x in cells],[x["mae_mps"] for x in cells],marker="o",label=method)
 ax.set_yscale("log");ax.set_xlabel("Post-scan delay (ms)");ax.set_title(seq.replace("boreas-",""),fontsize=9);ax.grid(alpha=.2)
axes[0].set_ylabel("Mean absolute reference disagreement (m/s)");axes[0].legend(fontsize=8)
fig.suptitle("Fixed sparse physical-recording comparison; 32 scans per recording\nTiming unit inferred; no hard true-speed certificate",fontsize=10)
fig.savefig(R/"reference_comparison.png",dpi=170);plt.close(fig)
print(json.dumps({"novelty_hypothesis_supported":assessment["hypothesis_supported"],"improved_cells":improvements,"worse_cells":assessment["worse_cells"],
"external_reference_comparisons": 864,"focused_passed":tests["focused"]["counts"]["passed"],"core_passed":tests["core"]["counts"]["passed"]},indent=2))
