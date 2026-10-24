"""Record all outcomes and the predeclared stop decision; no policy promotion."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/"comparison_summary.json").read_text())
v=json.loads((ROOT/"independent_verification.json").read_text())
diag=json.loads((ROOT/"native_rejection_diagnosis.json").read_text())
tests=json.loads((ROOT/"test_validation.json").read_text())
assert v["all_passed"] and all(t["exit_code"]==0 for t in tests.values())
n=sum(t["count"] for t in tests.values());assert n==104
sources=dict(paper=dict(author="Kenji Koide",year=2024,title="small_gicp: Efficient and parallel algorithms for point cloud registration",
    journal="Journal of Open Source Software",volume=9,issue=100,pages="6948",doi="10.21105/joss.06948",
    primary_url="https://joss.theoj.org/papers/10.21105/joss.06948"),
    implementation=dict(package="small-gicp",version="1.0.1",repository="https://github.com/koide3/small_gicp",
        package_release_year=2026,paper_year=2024),
    scope="General registration backend comparison on fixed development point pairs. Not a full published autonomous-vehicle defense or full odometry-system reproduction.")
(ROOT/"method_sources.json").write_text(json.dumps(sources,indent=2)+"\n")
(ROOT/"requirements-backend.txt").write_text("small-gicp==1.0.1\n")
lines=["# Compiled registration comparison","",
"Result: none of the frozen native variants qualifies for policy use. The designated GICP candidate fails the predeclared trigger for full raw replay. All results, including failed variants, are retained.","",
"## Scope and sources","",
"Compared ICP, point-to-plane ICP and GICP through the hash-pinned small_gicp 1.0.1 Windows/Python 3.12 binary. This is a 2026 software release of a library described in Koide's 2024 JOSS paper, not a 2026 method paper.",
"",
"Primary paper: https://joss.theoj.org/papers/10.21105/joss.06948",
"Implementation: https://github.com/koide3/small_gicp","",
"All variants use the exact 283 saved input pairs and support timestamps from the frozen streaming experiment. There are 265 nonempty input pairs. They share identity initialization, one CPU thread, 0.25 m downsampling, 16-neighbor preprocessing, 1.5 m matching distance and a 20-iteration budget. No CAN values or INS prior enters registration.",
"",
"The fixed native quality guard requires convergence, at least 100 inliers, 50% overlap, Euclidean inlier P95 no greater than 0.3 m, observable reference planar geometry, speed no greater than 60 m/s and yaw no greater than 2 rad/s. This guard uses a different residual from the original prototype's trimmed point-to-plane residual. The old/new acceptance counts therefore do not represent identical quality metrics.",
"",
"## Fixed-input outcomes","",
"| Method | Geometric accepts / 283 | Accepted P95 speed discrepancy (m/s) | Accepted errors >2 m/s | Method-only P95 time (ms) |",
"|---|---:|---:|---:|---:|"]
for method,m in s["metrics"].items():
    p95="Unavailable: no accepted outputs" if m["p95_query_discrepancy_mps"] is None else f'{m["p95_query_discrepancy_mps"]:.6f}'
    lines.append(f'| {method} | {m["accepted"]} | {p95} | {m["large_discrepancies_above_2_mps"]} | {m["method_p95_ms"]:.3f} |')
lines+=["",
"The original baseline's 112 accepts, four large errors and P95 1.322452 m/s were reproduced. Each native variant accepts zero outputs under its frozen guard. Zero accepted errors here is abstention, not successful attack detection or sufficient availability.",
"",
"## Why the native methods were rejected","",
"| Native method | Converged outputs, before guard | Ungated diagnostic P95 speed discrepancy (m/s) | Ungated diagnostic errors >2 m/s |",
"|---|---:|---:|---:|"]
for method,d in diag.items():
    lines.append(f'| {method} | {d["native_converged"]} | {d["ungated_diagnostic_p95_error_mps"]:.6f} | {d["ungated_diagnostic_large_errors"]} |')
lines+=["",
"All converged native outputs fail the 0.3 m Euclidean P95 quality check. The unfiltered GICP results still contain 17 large errors, so removing the quality gate would not establish an accurate source. Ungated figures are diagnostics only and never authorize policy acceptance.",
"",
"The results show that substituting a compiled registration backend alone does not solve the problem on these narrow, sparse interval pairs. Scene ambiguity, partial-view sampling and motion assumptions remain possible causes; this comparison does not prove which mechanism caused every error, nor that the full raw recordings are unusable.",
"",
"## Timing and stopping decision","",
"Method-only timing includes each backend's preprocessing, registration and common host quality calculations. File loading, upstream raw decoding and creation of the original point subsets are excluded. The original baseline is timed again on the same cached inputs for this comparison. These figures must not be substituted for end-to-end vehicle latency.",
"",
"The GICP raw-replay trigger was frozen before dataset outcomes: accepted P95 speed discrepancy <=0.5 m/s, zero accepted errors >2 m/s, and at least 95% acceptance among nonempty inputs. It failed. No full raw replay was run for these native variants; no native deadline or policy-coverage improvement is claimed.",
"",
"## Verification and remaining project work","",
f'All {n} relevant tests passed: four compiled-adapter tests and the existing 100 streaming/radar/context/authentication tests. Verification checked all 283 input hashes, 1,132 outputs, 795 native poses and quality calculations, original baseline consistency, and the installed binary against the pinned wheel.',
"",
"The current sensor-balanced policy remains unchanged at 224/283 nominal accepts (79.152%). Its 95% coverage requirement and validated acquisition timing remain unresolved. Final independent confirmation stays closed.",
"",
"Next research work must change the observation model rather than just exchange backends: examine stable scene support and a causally maintained map or broader independent observations, specify the time support and failure assumptions, then freeze a new evaluation design. Do not relax the deadline, reuse CAN as independent truth or tune these held development outcomes until a target passes. Published AV-security comparisons and novelty evidence remain incomplete.",
"",
"No manuscript, deployment, commit or GitHub push was performed. Physical witness use and promotion remain disabled; reserved payloads remain unopened, unhashed and unscored."]
(ROOT/"FINDINGS.md").write_text("\n".join(lines)+"\n")
(ROOT/"README.md").write_text("""# Fixed-input compiled registration comparison

Read FINDINGS.md before interpreting software test results.

This module compares a pinned small_gicp backend's ICP, PLANE_ICP and GICP cores on the existing frozen streaming LiDAR point pairs. No native variant passed the predeclared research quality guard, and the designated full raw-replay trigger failed.

The 2024 JOSS publication describes the general-purpose library. This project uses its 2026 package release; it does not claim a 2026 method paper or full vehicle-defense reproduction.

Key artifacts: protocol_v1.json, comparison_summary.json, native_rejection_diagnosis.json, independent_verification.json, backend_install.json, method_sources.json and completion.json. Runtime outputs and the wheel are ignored by git. Backend license notices are retained.

Original source data, comparisons and production policy are unchanged. Acquisition authenticity and physical error bounds are not established.
""")
status=ROOT.parent.parent/"CURRENT_PROJECT_STATUS.md"
backup=status.with_name("PROJECT_STATUS_BEFORE_COMPILED_COMPARISON.md")
if not backup.exists(): backup.write_bytes(status.read_bytes())
status.write_text(f"""# Current CO-ACT-CPS status

Latest completed batch: frozen comparison of compiled ICP, point-to-plane ICP and GICP on the same 283 streaming LiDAR input pairs.

- The original planar prototype's results were reproduced unchanged.
- All three native variants accept zero outputs under the frozen quality guard.
- Ungated GICP converges on 212 queries but still has 17 errors above 2 m/s and P95 discrepancy 6.2881 m/s. These diagnostic outputs never authorize acceptance.
- GICP fails the predeclared full raw-replay trigger; no native end-to-end timing or coverage improvement is claimed.
- All {n} relevant tests passed. Verification checked 283 inputs, 1,132 outputs, 795 native poses/quality checks and the pinned installed binary.
- This is a general-purpose registration-core comparison, not full reproduction of a published vehicle security or odometry system.

Current best existing research policy remains 224/283 nominal accepts (79.152%), below the 95% target. Original corrected radar timing still fails; physical acquisition age remains uncertified.

Latest files:
research/vehicle_gicp_comparison/FINDINGS.md
research/vehicle_gicp_comparison/comparison_summary.json
research/vehicle_gicp_comparison/native_rejection_diagnosis.json
research/vehicle_gicp_comparison/independent_verification.json

Next research decision: change the observation model, examining static-scene support and a causal map or broader independent observations. Merely replacing the registration backend is unsupported by these results. Freeze a new design before new outcomes; do not relax age/coverage/security targets.

Published AV-defense comparisons, novelty assessment and independent confirmation remain unfinished. Full project complete: no. Production unchanged; physical witness use/promotion disabled. Reserved payloads unopened/unhashed/unscored. No manuscript, deployment, commit or GitHub push.
""")
completion=dict(comparison_batch_complete=True,full_project_complete=False,tests_passed=n,
    verification_passed=True,variants_tested=["ICP","PLANE_ICP","GICP"],native_variants_passing_quality=0,
    full_raw_replay_trigger_passed=False,full_raw_replay_performed=False,
    end_to_end_native_deadline_evaluated=False,production_policy_changed=False,promotion_allowed=False,
    current_policy_nominal_coverage_pct=79.15194346289752)
(ROOT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
names=["protocol_v1.json","protocol_freeze.json","backend_install.json","backend_motion.py","test_backend_motion.py",
    "freeze_and_test.py","install_backend.py","inspect_backend.py","run_comparison.py","comparison_summary.json",
    "diagnose_native_outputs.py","native_rejection_diagnosis.json","verify_comparison.py","independent_verification.json",
    "run_tests.py","test_validation.json","tests_compiled_backend.log","tests_stream.log","tests_radar_timing.log",
    "tests_CAN_context.log","tests_authentication.log","method_sources.json","requirements-backend.txt","FINDINGS.md",
    "README.md","completion.json","finalize_comparison.py"]
names+=sorted(p.name for p in ROOT.glob("BACKEND_*") if p.is_file())
(ROOT/"file_manifest.json").write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},indent=2)+"\n")
print(json.dumps(completion,indent=2))
