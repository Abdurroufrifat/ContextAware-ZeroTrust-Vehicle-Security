import datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
p=dict(name="temporal_quality_block_calibration_v1",
 recordings=["07_turning","04_crossTraffic","14_countryRoad"],metadata="original training and calibration query times only; exclude evaluation and purged target rows",
 split="fit on original train; first half of original calibration separately per record for calibration, chronological second half for independent parameter-check; no split tuning",
 estimator="fixed pair: earlier100ms raw observations ending100ms before query, current20ms observations ending at query; identity initialization; no current observations in target; no ongoing state/map feedback",
 components="validated raw decoder; KISS-ICP1.3.0 core defaults used by prior diagnostics; source voxels0.5then1.5m; target voxel0.5m then1m/20-point hash map; no motion deskew because this pair has no prior motion state",
 features="14 predeclared LiDAR-only geometric, reciprocal correspondence, surface, rank, shape, own speed and interval features; no CAN, INS/reference values or recording identity in features",
 forest_parameters=dict(n_estimators=200,max_depth=6,min_samples_leaf=10,random_state=20261021,n_jobs=1),
 training_target="log1p absolute discrepancy to offline query-time INS; references loaded after all feature/pose outputs saved",
 positive_scale="max(0.05,expm1(mean predicted log error))",
 score="absolute query-time speed discrepancy divided by positive predicted scale",
 calibration="one-second blocks per recording starting at first scorable early-calibration row; take maximum normalized error in each block; alpha0.05, order ceil((nblocks+1)*0.95); absent order statistic means unbounded and abstention; incomplete blocks retained but no independence claim",
 accept="finite upper error estimate<=0.5m/s only; this is a research quality decision and never alone authorizes security use",
 age_limits_unchanged=True,candidate_agreement_mps=1,nominal_coverage_target_pct=95,
 check="report all held-late-calibration requests; no missing labels counted as success; unguarded and guarded results separate",
 stop_rule="one fixed model; do not shorten blocks, mix check/calibration, relax error/age limits, tune on check results or proceed to evaluation if model calibration/availability fails",
 timing="diagnostic pair feature cost only; raw decoder/worker/queue/query completion not replayed as a new online observer; no end-to-end freshness claim",
 evidence="freeze code and protocol before collecting outcomes; audit samples first valid role sample and every50th per-record query; portable forest exported for independent prediction replay",
 method_sources=[dict(title="A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification",url="https://arxiv.org/abs/2107.07511"),
 dict(title="Conformal prediction beyond exchangeability",url="https://arxiv.org/abs/2202.13415")],
 limitations="split conformal requires exchangeability assumptions; one-second maxima are a conservative research design, not a proof of independent blocks; correlated trajectories, distribution shift and adversarial conditional errors remain unvalidated",
 physical_speed_authorized=False,production_policy_changed=False,final_holdouts_accessed=False)
file=R/"protocol_v1.json";assert not file.exists()
file.write_text(json.dumps(p,indent=2)+"\n")
(R/"protocol_freeze.json").write_text(json.dumps(dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_new_pair_outcomes=True,
 implementation_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ("quality_core.py","collect_pairs.py","calibrate_quality.py")}),indent=2)+"\n")
print("Quality calibration protocol frozen")
