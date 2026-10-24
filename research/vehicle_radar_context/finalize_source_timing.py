"""Persist source feasibility and version regenerated test logs; no decision changes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/"lidar_point_timing_audit.json").read_text())
v=json.loads((ROOT/"lidar_timing_independent_verification.json").read_text())
tests=json.loads((ROOT/"test_validation_v2.json").read_text())
assert v["all_passed"] and all(x["exit_code"]==0 for x in tests.values())
count=sum(int(re.search(r"Ran (\d+) tests","\n".join(x["tail"])).group(1)) for x in tests.values())
assert count==92
mp=ROOT/"balanced_file_manifest.json";old=json.loads(mp.read_text())
allowed={"test_validation_v2.json","tests_v2_radar.log","tests_v2_CAN_context.log","tests_v2_authentication.log"}
changed={}
for name,sha in old.items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual!=sha:
        assert name in allowed,("Unexpected balanced package change",name)
        changed[name]=dict(previous_sha256=sha,current_sha256=actual)
if changed:
    snapshot=ROOT/"balanced_file_manifest_before_point_timing.json"
    if not snapshot.exists(): snapshot.write_bytes(mp.read_bytes())
    for name,item in changed.items(): old[name]=item["current_sha256"]
    mp.write_text(json.dumps(old,indent=2)+"\n")
(ROOT/"balanced_test_revalidation.json").write_text(json.dumps(dict(all_passed=True,current_tests=count,
    initial_balanced_batch_tests=84,regenerated_test_artifacts=changed,
    note="Shared test discovery now includes eight new point-time tests. Four generated logs/reports supersede the old versions; historical manifest hashes are retained. Estimator code, source data and comparison results are unchanged."),indent=2)+"\n")
source=json.loads((ROOT/"velodyne_source_provenance.json").read_text())
lines=["# Independent-source timing feasibility","",
"Status: timing investigation and point-time parser complete. No new velocity or security-policy result is claimed.","",
"The corrected radar track still fails the unchanged 100 ms deadline. Original processed headers remain 106 ms earlier than raw packet headers.","",
"## LiDAR timestamp model","",
"The pinned UniBwTAS Velodyne driver defines the encoded point-time field as a firing offset in seconds from the scan header. The research parser adds each offset to the unchanged header using integer epoch arithmetic. It never substitutes publication or query time.",
"",
"Repository: https://github.com/UniBwTAS/velodyne",
"Tree: "+source["tree_sha"],"",
"The published fork matches the extended field layout in these bags. The exact recording build is not attested. Its UDP-bridge receipt epoch plus firing-order offsets does not certify physical acquisition age or archived sensor authenticity.","",
"## Verified availability","",
f'Independently reread {r["total_scans"]:,} clouds, checked {r["total_valid_points"]:,} valid numeric point times and inspected {r["raw_packets"]:,} raw packet headers. No image pixels or point coordinates were inspected. INS was not an input.',"",
"| Development record | Queries | Queries with >=30 fresh cloud points | Queries with fresh raw packets |","|---|---:|---:|---:|"]
for name,m in r["records"].items():
    lines.append(f'| {name} | {m["evaluation_queries"]} | {m["queries_with_30_fresh_points"]} | {v["raw_results"][name]["queries_with_fresh_raw_packets"]} |')
lines+=["",
f'Cloud-point timestamp feasibility is {r["time_model_feasible_queries"]}/283; raw-packet header feasibility is {v["raw_packet_time_feasible_queries"]}/283. These are availability counts, not velocity accuracy, policy coverage or security success rates.',"",
"On the stationary cross-traffic record, 56 queries have no eligible cloud points. Inspected examples show the latest cloud's newest valid point is about 100.06–100.37 ms old while the next cloud has not arrived. These points are correctly rejected. Full-rotation publication introduces an availability gap relative to the query schedule.",
"",
"Raw LiDAR packets have eligible original headers at all 283 query times, with actual receipt times respected and future headers excluded. This supports investigating streaming decoding without waiting for a full cloud. Raw packet geometry and velocity accuracy were not established here.",
"",
"The first header-only audit used the documentation's generic bus/vls128/eth_scan/bus_to_host topic. Bag metadata identified the actual bus/vls128_roof/eth_scan/bus_to_host topic. The dedicated LiDAR protocol explicitly uses that topic; the first frozen audit is preserved.",
"",
"## Validation and limits","",
f'All {count} relevant tests passed: 32 radar/timing, 20 CAN context and 40 authentication. Eight new tests cover epoch precision, endianness, row padding, the inclusive age boundary, future/unreceived data, nonfinite values, unknown time semantics and malformed layouts. Independent source rereading reproduced every cloud/query count.',
"",
"The previous sensor-balanced policy remains at 224/283 nominal accepts (79.152%). Its inputs and decisions are unchanged. The 95% coverage requirement remains unmet.",
"",
"Next implement a streaming raw-LiDAR decoder and CAN-independent motion estimation with explicit time support. Measure accuracy, runtime, missingness, attack behavior and fusion effects before using it in policy evaluation. A validated acquisition-clock model and independent confirmation are still required.",
"",
"Production is unchanged; physical witness use and promotion remain disabled. Reserved payloads remain unopened, unhashed and unscored. No manuscript, deployment, commit or GitHub push was performed."]
(ROOT/"SOURCE_TIMING_FINDINGS.md").write_text("\n".join(lines)+"\n")
status=ROOT.parent.parent/"CURRENT_PROJECT_STATUS.md"
snapshot=status.with_name("PROJECT_STATUS_BEFORE_SOURCE_TIMING.md")
if not snapshot.exists(): snapshot.write_bytes(status.read_bytes())
status.write_text(f"""# Current CO-ACT-CPS status

Latest completed batch: independent-source timing audit and strict LiDAR point-time parser.

- Audited {r["total_scans"]:,} development clouds and independently checked {r["total_valid_points"]:,} valid point times.
- Inspected {r["raw_packets"]:,} raw LiDAR packet headers.
- Point-time availability: {r["time_model_feasible_queries"]}/283 queries. Raw-packet header availability: {v["raw_packet_time_feasible_queries"]}/283.
- Complete-cloud publication leaves 56 stationary-control queries without points inside the unchanged 100 ms deadline. Raw packets have eligible original headers at these query times.
- All {count} relevant tests passed, including eight new timing tests.
- Original headers and encoded offsets are preserved. The bridge-epoch model is not a certified acquisition clock; the exact recording driver build is not attested.
- Previous sensor-balanced policy remains 224/283 nominal accepts (79.152%) with unchanged source data and decisions.
- Corrected radar timing still fails. Coverage and certified acquisition freshness remain unresolved.

New results:
research/vehicle_radar_context/SOURCE_TIMING_FINDINGS.md
research/vehicle_radar_context/lidar_point_timing_audit.json
research/vehicle_radar_context/lidar_timing_independent_verification.json
research/vehicle_radar_context/lidar_timing.py

Next target: streaming raw-LiDAR decoding and CAN-independent motion estimation. Availability alone does not establish usable geometry, velocity accuracy or policy coverage. Published-method comparisons and frozen independent confirmation remain unfinished.

Full project complete: no. Production unchanged; physical witness use/promotion disabled. Reserved payloads unopened/unhashed/unscored. No manuscript, deployment, commit or GitHub push.
""")
completion=dict(batch_complete=True,full_project_complete=False,relevant_tests_passed=count,
    independent_timing_audit_passed=True,cloud_point_time_feasible_queries=r["time_model_feasible_queries"],
    raw_packet_time_feasible_queries=v["raw_packet_time_feasible_queries"],total_queries=283,
    existing_policy_nominal_coverage_pct=79.15194346289752,velocity_accuracy_established=False,
    acquisition_age_certified=False,production_policy_changed=False,promotion_allowed=False,
    next_target="streaming raw-LiDAR decoder and independently scored motion estimation")
(ROOT/"source_timing_completion.json").write_text(json.dumps(completion,indent=2)+"\n")
names=["audit_independent_sources.py","source_timing_protocol.json","source_timing_protocol_freeze.json",
    "independent_source_timing_audit.json","lidar_timing.py","test_lidar_timing.py","audit_lidar_points.py",
    "lidar_point_timing_protocol_v1.json","lidar_point_timing_freeze.json","lidar_point_timing_audit.json",
    "verify_lidar_timing.py","lidar_timing_independent_verification.json","SOURCE_TIMING_FINDINGS.md",
    "source_timing_completion.json","velodyne_driver_metadata.json","velodyne_source_provenance.json",
    "balanced_test_revalidation.json","test_validation_v2.json","tests_v2_radar.log","tests_v2_CAN_context.log",
    "tests_v2_authentication.log","finalize_source_timing.py"]
(ROOT/"source_timing_file_manifest.json").write_text(json.dumps(
    {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2)+"\n")
print(json.dumps(completion,indent=2))
