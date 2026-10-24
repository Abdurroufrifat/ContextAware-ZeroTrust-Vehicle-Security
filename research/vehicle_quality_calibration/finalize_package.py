import hashlib,importlib.metadata,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;PROJECT=R.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((R/"quality_summary.json").read_text());pairs=json.loads((R/"pair_summary.json").read_text());v=json.loads((R/"independent_verification.json").read_text());t=json.loads((R/"test_validation.json").read_text());d=json.loads((R/"quality_limit_diagnosis.json").read_text())
assert v["all_passed"] and all(x["exit_code"]==0 for x in t.values())
count=sum(x["count"] for x in t.values());assert count==142
assert s["calibration_block_count"]==13 and s["block_quantile"] is None and s["guarded_check"]["accepted"]==0
assert d["primary_summary_sha256"]==sha(R/"quality_summary.json")
assert d["optimistic_oracle_rejection_only_max_accepts"]==s["check_requests"]-s["unguarded_check"]["errors_above_2_mps"]==45
previous={}
for folder,manifest in [(R.parent/"vehicle_motion_diagnosis","file_manifest.json"),(R.parent/"vehicle_packet_map","file_manifest.json"),(R.parent/"vehicle_lidar_map","file_manifest.json"),(R.parent/"vehicle_gicp_comparison","file_manifest.json"),(R.parent/"vehicle_lidar_stream","file_manifest.json"),(R.parent/"vehicle_radar_context","source_timing_file_manifest.json"),(R.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
 expected=json.loads((folder/manifest).read_text())
 for name,h in expected.items():assert (folder/name).parent==folder and sha(folder/name)==h,(folder,name)
 previous[folder.name+"/"+manifest]=len(expected)
(R/"method_sources.json").write_text(json.dumps(dict(primary_sources=[
 dict(title="A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification",authors="Anastasios N. Angelopoulos; Stephen Bates",arxiv="2107.07511",url="https://arxiv.org/abs/2107.07511",submitted_year=2021),
 dict(title="Conformal prediction beyond exchangeability",authors="Rina Foygel Barber; Emmanuel J. Candes; Aaditya Ramdas; Ryan J. Tibshirani",arxiv="2202.13415",url="https://arxiv.org/abs/2202.13415",submitted_year=2022)],
 attribution="finite-sample split-conformal rank is a published construction; the fixed forest, geometry features and one-second block design here are research implementation choices, not a claimed theorem for dependent vehicle data",
 exchangeability_established=False,independent_blocks_established=False,physical_certificate=False),indent=2)+"\n")
text=f"""Built and checked one frozen learned motion-quality model using separate temporal observations from training/calibration data.

Data and split:
705 fixed query pairs were collected from 07_turning, 04_crossTraffic and 14_countryRoad; 700 produced valid geometry feature vectors. Only original train/calibration target rows were used. Original evaluation and final confirmation payloads were not decoded.
The original training segment supplied fitting rows. Each recording's original calibration segment was divided chronologically in half: early half sets the calibration multiplier, late half checks the fixed model. There were 429 valid/scorable fitting rows, 112 calibration rows and 117 check requests, of which 115 were valid/scorable. Missing references are retained as unknown.

Observation/model scope:
A fresh 20 ms raw firing window is matched against an earlier full 100 ms window ending 100 ms before the request. There are no current-source points in the target. The registration uses the previously pinned KISS-ICP core, fixed source/target voxel settings, identity initialization and no prior-motion deskew. This fixed temporal-pair model differs from the earlier continuous packet-map observer: the learned scale is not validated for that previous observer.
Fourteen LiDAR-only features describe overlap, residual quantiles, surface-normal residuals, reciprocal matches, local geometry, source shape, the estimator's own speed and interval. No CAN, INS/reference speed, recording identity or check labels enter features.
After all features and poses were frozen, offline references supplied absolute query-time speed discrepancies. One random forest was fitted to log1p discrepancy, with 200 trees, depth6, minimum leaf10, fixed seed and one CPU thread. Its predicted positive error scale is floored at0.05 m/s. Model parameters were not tuned on check outcomes.

Calibration result:
Normalized errors are discrepancy divided by predicted scale. Frames within each recording are grouped in one-second bins, and each bin contributes its maximum normalized error.
Only 13 calibration bins are available: 3 stationary cross-traffic, 2 turning, 8 country-road. At alpha0.05, rank ceil((13+1)*0.95)=14 lies beyond the thirteen observed maxima. The implementation correctly returns an unbounded estimate and abstains. At least19 bins are necessary for this rule to yield a finite observed order statistic; sufficient count alone would not establish exchangeability or independence.
The per-frame multiplier {s["framewise_diagnostic_quantile"]:.9f} is recorded as a diagnostic and never substituted for block calibration.
One-second grouping is a research design to reduce frame pseudo-replication. These adjacent blocks are not proven independent/exchangeable. Even sufficient data for a finite multiplier would not establish a hard per-estimate physical bound, freshness, attack-conditional validity or a generalization guarantee.

Check result:
The unguarded estimator has P95 discrepancy {s["unguarded_check"]["p95_error_mps"]:.9f} m/s across115 scorable requests, with72 errors above2 m/s. Two check requests have unknown error.
The forest's predicted-scale MAE on scorable check rows is {s["learned_scale_check_mae_mps"]:.9f} m/s.
The operative quality rule accepts zero of117 check requests because calibration is unbounded. Error metrics on zero accepts are undefined; no defense success is claimed.
A post-result arithmetic diagnosis gives an optimistic upper ceiling for ANY rejection-only filter of these unchanged estimates: it must reject the72 known large errors, so even granting every remaining request safe, at most45/117 (38.462%) can be accepted. This is not achieved safe coverage and is specific to this estimator/check set. Additional calibration bins alone cannot repair its accuracy or meet95% availability.

Verification:
142 tests passed. Independent verification checked all705 split/label/prediction records, recalculated18 frozen feature snapshots, independently refitted the forest using fitting rows only, replayed portable exported tree predictions and checked all13 block maxima/order statistics. Seven earlier result manifests remain unchanged.
No new audit of every raw payload was performed; the validated decoder and the prior eighteen-window source audit remain the provenance basis. No online end-to-end worker/queue freshness test is claimed by this pair study.
Model, portable forest, frozen inputs, labels, predictions, protocols and source references are saved locally. No dependencies, production policy, manuscript, deployment, commit or push changed.

Decision:
This research candidate fails calibration and availability requirements. Do not apply it to the previous observer or the security policy. The existing research policy remains224/283 nominal coverage (79.152%), below95%; physical acquisition authenticity/clock/error bounds remain unproved.
Stop quality-only tuning on this unchanged temporal-pair estimator. The next requirement is a materially better independent motion estimate/observation model, together with additional separate moving calibration sequences and honest uncertainty assumptions. Preserve final confirmation data until a jointly accurate, timely candidate is frozen. Full project complete:no.
"""
(R/"FINDINGS.md").write_text(text)
(R/"README.md").write_text("""Research-only temporal motion-quality calibration. Read FINDINGS.md and quality_limit_diagnosis.json.
Use D:/CO-ACT-CPS/.venv/Scripts/python.exe with run_tests.py or verify_quality.py to check saved work.
freeze_protocol.py, collect_pairs.py, calibrate_quality.py and finalize_package.py are one-time evidence runs; preserve prior evidence before starting a separate protocol.
The trained forest is exported both as a generated joblib file and portable forest.json. The latter allows prediction verification without loading a model pickle.
This model is for the fixed temporal-pair observation model, not the previous continuous packet-map observer. Calibration is insufficient and policy promotion is disabled.
Keep runtime out of git. Zero accepted requests establishes failed availability, not a successful defense.
""")
(R/".gitignore").write_text("runtime/\n__pycache__/\n")
(R/"environment.json").write_text(json.dumps(dict(versions={n:importlib.metadata.version(n) for n in ["numpy","scipy","scikit-learn","kiss-icp","small-gicp"]},dependencies_changed=False),indent=2)+"\n")
status=PROJECT/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_QUALITY_CALIBRATION.md"
assert not backup.exists();shutil.copyfile(status,backup);old=status.read_text(encoding="utf-8")
status.write_text("""Latest verified research batch: temporal motion-quality model/calibration (2026-10-21).
Built one fixed14-feature random-forest error-scale model from705 train/calibration temporal pairs. Original evaluation/final payloads were not decoded.
429 fitting labels,112 early-calibration labels,117 separate late-calibration check requests;115 scorable checks. All142 regression tests and independent verification passed.
Calibration failed:13 one-second maxima; finite-sample95% rank14 exceeds the available13, so the operative error limit is unbounded and all117 checks abstain.
Underlying check estimates have P95 discrepancy7.350409m/s and72 errors above2m/s. Even an optimistic perfect rejection-only filter can accept at most45/117 (38.462%), below95%; more calibration alone cannot fix this estimator.
The learned scale is not validated for the old continuous packet-map observer. No formal coverage/independent-block/physical clock or error certificate is established.
No promotion. Existing research policy nominal coverage remains79.151943%, below95%. Full project complete:no.
Next requirement: materially improve the independent motion estimate/observation model and add separate moving calibration sequences. Stop quality-only tuning of this unchanged pair estimator. Preserve final confirmation data until a jointly accurate and timely candidate is frozen.
Evidence: research/vehicle_quality_calibration/FINDINGS.md, quality_summary.json, quality_limit_diagnosis.json, independent_verification.json.
Previous status retained below as history.

"""+old,encoding="utf-8")
run=R/s["runtime_relative_path"]
completion=dict(batch_complete=True,full_project_complete=False,tests_passed=count,independent_verification_passed=True,
 pairs=705,calibration_blocks=13,calibration_passed=False,check_requests=117,check_accepted=0,prior_manifests_unchanged=previous,
 evaluation_payloads_decoded=False,protected_holdouts_accessed=False,model_promoted=False,production_policy_changed=False,
 runtime_files={str(p.relative_to(R)):sha(p) for p in run.iterdir() if p.is_file()},project_status_sha256=sha(status))
(R/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ("file_manifest.json","package_audit.json")}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(batch_complete=True,tests_passed=count,pairs=705,prior_manifests_unchanged=previous,manifest_files=len(manifest),calibration_passed=False),indent=2))
