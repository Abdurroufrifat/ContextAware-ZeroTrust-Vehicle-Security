"""Persist the failed frozen candidate honestly; leave the previous policy untouched."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/"stream_benchmark_summary.json").read_text())
verify=json.loads((ROOT/"independent_verification.json").read_text())
tests=json.loads((ROOT/"test_validation.json").read_text())
decoder=json.loads((ROOT/"decoder_validation.json").read_text())
assert verify["all_passed"] and all(v["exit_code"]==0 for v in tests.values()) and all(v["all_passed"] for v in decoder.values())
test_count=sum(v["count"] for v in tests.values());assert test_count==100
runtime=ROOT/summary["runtime_relative_path"]
with (runtime/"estimates.jsonl").open() as f: estimates=[json.loads(l) for l in f]
geometry=[r for r in estimates if r["estimate"]["accepted"]]
ages=[(r["completed_ns"]-r["lower_support_midpoint_ns"])/1e6 for r in geometry]
latency=dict(minimum_completed_lower_support_age_ms=min(ages),median_completed_lower_support_age_ms=float(np.median(ages)),
    p95_completed_lower_support_age_ms=float(np.quantile(ages,.95)),maximum_completed_lower_support_age_ms=max(ages))
(ROOT/"timing_failure_diagnosis.json").write_text(json.dumps(latency,indent=2)+"\n")
point_count=sum(v["matched_raw_points"] for v in decoder.values())
largest_error=max(v["maximum_geometry_discrepancy_m"] for v in decoder.values())
lines=["# Streaming LiDAR development benchmark","",
"Result: this frozen candidate fails both speed accuracy and the 100 ms report-age requirement. It is excluded from the existing policy. Completing the implementation and tests does not complete the full project.","",
"## What was implemented","",
"A vectorized VLS-128 single-return decoder preserves original packet headers and computes per-laser firing offsets using the pinned upstream driver and factory calibration. Unsupported models, modes, malformed bank order and bad packets are rejected. The retained upstream reference contains its original notices.",
"",
"A LiDAR-only planar point-to-plane estimator compares the newest 0–20 ms of returns with observations aged 100–120 ms. The result is an interval-motion estimate. Its interval midpoint and lower support midpoint are recorded explicitly; old observations are never described as new acquisitions.",
"",
"Each packet's deserialization/decoding and each query's preprocessing/registration is modeled on one serialized CPU worker. Bag I/O and saving research artifacts are excluded from this workload model. The report-age gate includes queue wait and measured processing time. This is a development replay model, not a production hardware timing guarantee.",
"",
"The factory calibration and fixed method parameters were selected before motion outcomes. Neither CAN values nor INS enters motion estimation. Query times and identifiers are reused from the existing 283-query development protocol. INS is loaded only after all estimates are saved.",
"",
"## Decoder validation","",
f'Matched all {point_count:,} valid points in one complete recorded cloud from each of the three development recordings. Maximum XYZ discrepancy is {largest_error:.9f} m, and maximum timestamp discrepancy is {max(v["maximum_timestamp_discrepancy_ns"] for v in decoder.values())} ns. This checks source-code agreement; it is not independent proof of physical sensor calibration or acquisition freshness.',
"",
"## Frozen results","",
"| Recording | Queries | Geometric estimates | Deadline-eligible estimates | Registration/preprocessing P95 (ms) |",
"|---|---:|---:|---:|---:|"]
for name,v in summary["records"].items():
    lines.append(f'| {name} | {v["queries"]} | {v["geometry_accepted"]} | {v["deadline_eligible"]} | {v["registration_p95_ms"]:.3f} |')
g=summary["metrics"]["geometry_accepted"]
lines+=["",
f'Ignoring the deadline yields {g["accepted"]}/283 estimates ({g["availability_pct"]:.3f}%). Their query-time P95 discrepancy is {g["p95_query_time_discrepancy_mps"]:.6f} m/s; maximum discrepancy is {g["maximum_query_time_discrepancy_mps"]:.6f} m/s. Four exceed 2 m/s.',
"",
f'Scoring at the declared interval midpoint gives P95 {verify["midpoint_time_p95_discrepancy_mps"]:.6f} m/s and still four errors above 2 m/s. One of the 112 interval-time references is unavailable and is not filled; all 112 have query-time references.',
"",
f'After including measured processing and queue time, zero estimates meet the age gate. Among geometrically accepted outputs, the minimum age of the lower interval-support midpoint at completion is {latency["minimum_completed_lower_support_age_ms"]:.3f} ms, median {latency["median_completed_lower_support_age_ms"]:.3f} ms and P95 {latency["p95_completed_lower_support_age_ms"]:.3f} ms. The unchanged limit is 100 ms.',
"",
"## Accuracy failure examples","",
"| Record/index | Estimated speed (m/s) | Query reference (m/s) | Discrepancy (m/s) | Registration residual P95 (m) |",
"|---|---:|---:|---:|---:|"]
for v in verify["geometry_large_errors"]:
    lines.append(f'| {v["recording"]}/{v["index"]} | {v["speed"]:.6f} | {v["query_reference"]:.6f} | {v["query_error"]:.6f} | {v["residual_p95_m"]:.6f} |')
lines+=["",
"The small geometric residuals and high overlap did not distinguish correct motion from these incorrect estimates. Static-scene support, narrow observation windows and scene ambiguity need further investigation. No object-motion labels were used to prove which mechanism caused each failure. Reducing processing time alone would not fix the observed accuracy failures.",
"",
"## Verification and project status","",
f'All {test_count} relevant tests passed: eight streaming tests, 32 radar/timing tests, 20 CAN-context tests and 40 authentication tests. An independent audit checked all 283 input windows, timing arithmetic and hashes, 112 accepted poses and registration-quality calculations, and the offline discrepancies.',
"",
"The independent verification scope is saved input traces and source-cloud decoding agreement. It does not authenticate archived acquisition or establish a physical error bound. No attack-protection improvement is claimed for this failed candidate.",
"",
"The previous sensor-balanced policy remains unchanged at 224/283 nominal accepts (79.152%). The 95% coverage target, validated acquisition timing, fair published-method comparisons, novelty evidence and independent confirmation remain unfinished. Reserved payloads remain unopened, unhashed and unscored.",
"",
"## Next research decision","",
"Do not promote this candidate or tune thresholds to make the current records pass. A replacement must address scene ambiguity and processing cost, with parameters frozen before new outcomes. Evaluate an established motion method or a source with validated acquisition timing before adding another policy variant. Faster decoding alone is insufficient.",
"",
"No manuscript, deployment, commit or GitHub push was performed."]
(ROOT/"FINDINGS.md").write_text("\n".join(lines)+"\n")
(ROOT/"README.md").write_text("""# Streaming LiDAR research prototype

Frozen development-only VLS-128 decoding and planar interval-motion benchmark.

The decoder matches the checked published clouds. The motion candidate fails accuracy and timing requirements and is not eligible for policy use. Read FINDINGS.md and stream_benchmark_summary.json before interpreting the test results.

Key files:
- protocol_v1.json and protocol_freeze.json: fixed candidate configuration
- stream_decoder.py: supported raw packet decoder
- motion.py: planar interval registration
- validate_decoder_bags.py: independent comparison to archived processed clouds
- run_stream_benchmark.py: causal single-worker replay and offline scoring
- verify_benchmark.py: independent saved-input timing/pose/reference audit
- test_validation.json: this batch's test results, stored separately from earlier experiments

Runtime point subsets and packet samples are ignored by git. Use the existing three allowlisted development bags only. No acquisition authentication, hard physical error bound, production integration or publication novelty is established.
""")
status=ROOT.parent.parent/"CURRENT_PROJECT_STATUS.md";snapshot=status.with_name("PROJECT_STATUS_BEFORE_LIDAR_STREAM.md")
if not snapshot.exists(): snapshot.write_bytes(status.read_bytes())
status.write_text(f"""# Current CO-ACT-CPS status

Latest completed batch: implemented and evaluated a frozen streaming LiDAR decoder and interval-speed prototype.

Decoder:
- All {point_count:,} checked points matched the published processed clouds across three development recordings.
- Largest XYZ difference {largest_error:.9f} m; largest timestamp difference 12 ns.

Motion prototype: FAILED REQUIREMENTS.
- 112/283 outputs pass geometric quality checks.
- Query-time P95 discrepancy 1.32245 m/s; four errors exceed 2 m/s, maximum 7.25408 m/s.
- Interval-midpoint scoring still has four errors above 2 m/s.
- Zero outputs meet the unchanged 100 ms age deadline after measured processing and queue time.
- This candidate is excluded from the existing policy. No security improvement is claimed.
- All {test_count} relevant software tests passed; independent input/pose/timing/reference verification passed.

Current best existing research policy remains unchanged: 224/283 nominal accepts (79.152%), below the 95% target. Acquisition freshness is still uncertified.

Latest files:
research/vehicle_lidar_stream/FINDINGS.md
research/vehicle_lidar_stream/stream_benchmark_summary.json
research/vehicle_lidar_stream/independent_verification.json
research/vehicle_lidar_stream/completion.json

Next work requires a method/source change addressing scene ambiguity and processing cost; do not tune thresholds to manufacture success. Published-method comparisons, novelty assessment and frozen independent confirmation remain unfinished.

Full project complete: no. Production unchanged; physical witness use/promotion disabled. Reserved payloads unopened/unhashed/unscored. No manuscript, deployment, commit or GitHub push.
""")
completion=dict(implementation_batch_complete=True,full_project_complete=False,tests_passed=test_count,
    decoder_checked_points=point_count,decoder_validation_passed=True,independent_trace_audit_passed=True,
    method_accuracy_requirement_passed=False,method_deadline_requirement_passed=False,method_promoted=False,
    production_policy_changed=False,reserved_payloads_accessed=False,current_policy_nominal_coverage_pct=79.15194346289752)
(ROOT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
names=["protocol_v1.json","protocol_freeze.json","implementation_notes_v1.json","stream_decoder.py","motion.py",
    "factory_VLS128.yaml","factory_calibration.json","source_provenance.json","source_probe.json","probe_source.py",
    "validate_decoder_bags.py","decoder_validation.json","run_stream_benchmark.py","stream_benchmark_summary.json",
    "verify_benchmark.py","independent_verification.json","test_stream.py","freeze_and_test.py","run_tests.py",
    "test_validation.json","tests_stream.log","tests_radar_timing.log","tests_CAN_context.log","tests_authentication.log",
    "timing_failure_diagnosis.json","FINDINGS.md","README.md","completion.json","finalize_stream.py"]
(ROOT/"file_manifest.json").write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2)+"\n")
print(json.dumps(dict(completion=completion,timing_failure=latency),indent=2))
