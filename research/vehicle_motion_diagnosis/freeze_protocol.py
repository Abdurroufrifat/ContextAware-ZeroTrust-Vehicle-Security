import datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
protocol=dict(name="training_calibration_geometry_floor_diagnosis_v1",
 hypothesis="downsampling/map insertion may make the fixed 0.3m residual screen reject the exact known pose; test sampling separately from motion and optimizer behavior",
 development_recordings=["07_turning","04_crossTraffic","14_countryRoad"],allowed_roles=["train","calibration"],
 anchors="25%, 50%, 75% indices within each recording and allowed role; metadata identifiers/times only; eighteen anchors total",
 raw_topic="/bus/vls128_roof/eth_scan/bus_to_host",
 data_rule="decode raw payloads only within 202ms receipt support before each allowed anchor; stop each bag at last calibration anchor; exclude future receipts/firings and preserve original epoch",
 observation_windows_ms=dict(parent=100,current=20,previous_start=200,previous_end=100),
 identity_controls="source is a voxelized exact subset of same-parent points, deliberately sharing observations; no motion compensation or unknown motion is involved in this correspondence control",
 target_stages=["raw_parent","voxel_half_m","default_voxel_map","source_preserving_voxel_map"],
 source_voxel_sequence_m=[.5,1.5],target_voxel_m=.5,map_voxel_m=1,map_max_points_per_voxel=20,
 known_pose_controls=[dict(translation_x_m=0,yaw_rad=0),dict(translation_x_m=.04,yaw_rad=0),dict(translation_x_m=.16,yaw_rad=.005)],
 registration="frozen pre-voxelized source, official KISS-ICP1.3.0 core, identity initialization, max distance6m/kernel2m, 500 iterations, convergence1e-4, four threads; actual reinsertion target frozen and scored",
 metrics="nearest-neighbor untrimmed inlier P95 at exact pose and fitted pose, inlier bound1.5m, residual screen0.3m, overlap>=0.5, >=100 correspondences; known pose translation/rotation error; translation error divided by0.02s labelled equivalent speed error, never measured vehicle speed",
 alternative_residual="diagnostic point-to-plane P95 at exact identity using 16-neighbor PCA normals on default map; no replacement threshold selected",
 temporal_control="six near-stationary cross-traffic anchors matched to earlier 100ms window with no shared current points; offline INS speed checked after every fit is frozen; identity is approximate, not exact pose truth",
 input_snapshots="all 18 anchor arrays, all 108 known-transform source/actual-target pairs and six temporal pairs persisted with hashes",
 stop_rule="one prespecified diagnostic run; report all controls; do not tune parameters or change observer/policy based on these outcomes in this batch",
 prohibitions="no evaluation raw decoding, final holdout payload access, promotion, uncertainty-certificate claim, runtime-freshness claim, manuscript or production change",
 uncertainty="these controls diagnose failure mechanisms; they are not data for a generalization bound; calibrated observer uncertainty remains an open requirement")
p=R/"protocol_v1.json";assert not p.exists()
p.write_text(json.dumps(protocol,indent=2)+"\n")
(R/"protocol_freeze.json").write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_outcomes=True,
 implementation_hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ("controls.py","run_diagnosis.py")}),indent=2)+"\n")
print("Diagnostic protocol frozen")
