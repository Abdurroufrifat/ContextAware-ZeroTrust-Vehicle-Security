import datetime,hashlib,json
from pathlib import Path
from packet_core import CONFIG
R=Path(__file__).resolve().parent
protocol=dict(name="latest_packet_window_map_v1",development_recordings=["07_turning","04_crossTraffic","14_countryRoad"],
 sensor="raw VLS128 single return, retained validated factory calibration and bridge firing epoch",bootstrap_ms=100,
 observation_window_ms=20,cadence_minimum_ms=20,raw_receipt_lookback_padding_ms=2,pending_retention_ms=220,
 scheduling="single serialized decode plus registration worker; at max(next_due,previous_completion), process only most recent receipt/firing window; discard expired pending packets; fixed queries observe latest completed job, including failures",
 reset="new full-window bootstrap when last accepted state end is over 200ms old; separate coordinate epoch; failed matches do not update pose/map",
 motion="SE3 constant twist scaled separately for actual pose gap and firing span; no CAN/INS initialization or online values",
 geometry=dict(min_points=150,min_correspondences=100,max_correspondence_m=1.5,min_overlap=.5,max_inlier_euclidean_p95_m=.3,max_fixed_correspondence_jacobian_condition=1000),
 numeric=dict(min_interval_seconds=.005,max_interval_seconds=.2,max_speed_mps=60),
 age_limit_ns=100000000,age_rule="both inferred endpoint midpoint and conservative first-firing midpoint must be within age limit at query; native completion must precede query",
 uncertainty="local fixed-correspondence geometry screens only; no calibrated error bound or physical acquisition certificate; abstain on failed screens",
 configuration=CONFIG,backend="KISS-ICP 1.3.0 components; custom packet scheduler, variable-interval motion scaling and guarded map updates; not full official-pipeline reproduction",
 original_targets=dict(nominal_coverage_pct=95,p95_discrepancy_mps=.5,accepted_errors_above_2_mps=0,candidate_agreement_mps=1),
 scoring="all outputs persisted before offline INS; integer timestamp interpolation, bracket <=50ms, no extrapolation; missing references remain unscorable",
 timing="measure deserialize/decode/window selection, preprocessing/registration/map-quality/state update; exclude bag disk I/O, audit artifact persistence and report scoring; measured CPU replay, no runtime guarantee",
 audit_sample="first two jobs and every hundredth job per recording; save input plus pre-update map/source snapshot",
 stop_rule="one fixed run on all three development recordings; retain failures and abstentions; no parameter tuning after outcomes; no promotion or final holdout access",
 physical_speed_authorized=False,production_policy_changed=False)
p=R/"protocol_v1.json";assert not p.exists()
p.write_text(json.dumps(protocol,indent=2)+"\n")
freeze=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_dataset_outcomes=True,
 implementation_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ["packet_core.py","run_packet_benchmark.py"]})
(R/"protocol_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
print("Frozen",freeze["sha256"])
