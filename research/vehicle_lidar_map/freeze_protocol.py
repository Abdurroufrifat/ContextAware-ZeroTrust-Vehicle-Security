import hashlib,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
p=dict(name="causal_full_cloud_KISS_ICP_v1",development_recordings=["07_turning","04_crossTraffic","14_countryRoad"],
 configuration=json.loads((R/"backend_environment.json").read_text())["config"],
 configuration_rule="published defaults, resource cap four CPU threads; no outcome-dependent tuning",
 observations="full 360 degree processed LiDAR; retain original header, per-point offsets and receipt; filter future firings; no CAN/INS values online",
 time_model="upstream v1.3.0 deskews to max retained firing offset; pose endpoint is derived under constant interscan motion; interval midpoint is inferred speed time, not a fresh physical acquisition",
 processing="continuous single serial worker; deserialize, validation, preprocessing, registration, map update, kinematics included; bag disk I/O and artifact persistence excluded; no real-time guarantee",
 map_rule="reset each recording, initialize identity; official pipeline registers against earlier scans then updates map; no reference initialization",
 fixed_queries=283,query_rule="latest completed output only; do not substitute an older result if latest numeric estimate fails",
 numeric_screen=">=100 valid returns, finite pose, 0.05<=interval_seconds<=0.2, speed<=60 m/s; not a geometric quality certificate",
 timing_screen="completed<=query and 0<=query minus velocity interval midpoint<=100000000 ns",
 offline_scoring="INS loaded only after all poses and queries frozen; query-time and interval-midpoint errors; <=50000 us interpolation bracket, no extrapolation",
 target=dict(coverage_pct=95,p95_error_mps=.5,errors_above_2_mps=0),
 stop_rule="one fixed complete replay; retain all outcomes; no policy fusion or promotion without supported quality/freshness evidence",
 physical_speed_authorized=False,production_policy_changed=False,holdouts_accessed=False)
path=R/"protocol_v1.json";assert not path.exists()
path.write_text(json.dumps(p,indent=2)+"\n")
(R/"protocol_freeze.json").write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_dataset_outcomes=True),indent=2))
print("Protocol frozen")
