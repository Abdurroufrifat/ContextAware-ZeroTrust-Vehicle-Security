import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent.parent;B=P/"research"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((R/"readiness_report.json").read_text())
assert report["core_tests"]["passed"]==731 and report["core_tests"]["skipped"]==4 and report["focused_tests"]==146
assert not report["full_project_complete"] and report["packaging"]["wheel_verified"]
source=R/"check_readiness.py"
text=source.read_text()
for a,b in {"Its15":"Its 15","its15":"its 15","contain27,178":"contain 27,178","On37,340":"On 37,340","requests,36,046":"requests, 36,046","track remains at79.151943%":"track remains at 79.151943%","has13 one-second":"has 13 one-second","fixed95%":"fixed 95%","above2m/s":"above 2 m/s","below95%":"below 95%","meet95%":"meet 95%","above2mps":"above2mps"}.items():text=text.replace(a,b)
source.write_text(text)
# Documentation formatting only; recompute the same saved metrics to refresh generated prose.
from check_readiness import main
assert main()==2
report=json.loads((R/"readiness_report.json").read_text())
for name,h in report["source_inputs_sha256"].items():assert sha(P/name)==h,(name,"input changed")
previous={}
for folder,manifest in [(B/"vehicle_quality_calibration","file_manifest.json"),(B/"vehicle_motion_diagnosis","file_manifest.json"),(B/"vehicle_packet_map","file_manifest.json"),(B/"vehicle_lidar_map","file_manifest.json"),(B/"vehicle_gicp_comparison","file_manifest.json"),(B/"vehicle_lidar_stream","file_manifest.json"),(B/"vehicle_radar_context","source_timing_file_manifest.json"),(B/"vehicle_radar_context","balanced_file_manifest.json")]:
 expected=json.loads((folder/manifest).read_text())
 for n,h in expected.items():assert (folder/n).parent==folder and sha(folder/n)==h,(folder,n)
 previous[folder.name+"/"+manifest]=len(expected)
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_COMPLETION_AUDIT.md"
assert not backup.exists();shutil.copyfile(status,backup);old=status.read_text(encoding="utf-8")
status.write_text("""Latest: consolidated software and research-completion audit (2026-10-21).

Software: full configured core suite passed 731 tests, skipped 4, with one legacy residual-calibration warning. Selected research and completion suites passed 146 tests. A staged, offline wheel build passed; the previously missing README declared by pyproject.toml was supplied. No model or production policy changed.

Strongest documented NavINST motion pipeline: saved predictions reproduce 27,178 route rows and 37,340 scheduled development requests. Availability is 96.5345%; accepted overall P95 reference discrepancy is 0.385366 m/s with zero accepted recorded discrepancies over 2 m/s. These are retrospective reference disagreements, not certified physical errors.
Original failures remain: evaluation_urban04_0 route-window P95 0.512122 m/s, and retained calibration_0 block P95 0.505660 m/s. The latter is hash-bound historical evidence, not freshly recomputed in this audit. Aggregates do not erase subset failures.
The separate 7V policy track remains at 79.151943% nominal coverage. Its failed temporal-pair quality model is not a replacement for the stronger NavINST model or a validated security witness.

Overall project complete: NO. Journal readiness: NOT ESTABLISHED.
Three remaining research work packages:
1. A usable independent motion/measurement method, justified reference/source/time evidence, adequate separate calibration, and simultaneous original accuracy, availability and threat-case success.
2. Compatible published-method comparison and a specific supported algorithmic contribution.
3. After development requirements pass and the method is frozen, an independently separated final confirmation.
These are substantive research requirements, not additional installation commands. Rejection-only tuning of the failed 7V pair estimator cannot supply the needed availability.

Reserved Urban02 NIGHT, RAV4 and vehicle_final_evaluation payloads remain excluded; none was opened or hashed by the completion checks. No manuscript, human study, deployment, commit or push occurred.
Start here: research/project_completion/COMPLETION_REPORT.md and readiness_report.json.
Prior status retained below as history.

"""+old,encoding="utf-8")
(R/".gitignore").write_text("runtime/\n__pycache__/\n")
(R/"README.md").write_text("""Completion audit, not a new estimator or a successful research result.
COMPLETION_REPORT.md states what is verified and what remains. readiness_report.json keeps machine-readable gates and exact source fingerprints.
run_core_checks.py runs the configured core tests with offline/protected-path guards and project-local temporary directories.
run_focused_checks.py checks selected research modules and the completion predicate; it is separate from the full core suite.
check_readiness.py recomputes saved strongest-pipeline metrics and exits with status2 when research completion remains blocked. This status is intentional, not a software test failure.
check_packaging.py builds a clean staged wheel without network/dependency installation. Its one-time staging assertion prevents accidental overwriting.
The missing root README was repaired. Existing models, gates and experimental results remain unchanged.
Reserved measurement paths are rejected before opening. The old signed-axis full verifier was not invoked because its historical manifest includes protected paths; this audit uses an explicit safe local manifest and saved prediction/reference arrays.
Do not equate software test success with a calibrated physical source, successful threat evaluation, method novelty or journal acceptance.
""")
completion=dict(audit_complete=True,full_project_complete=False,core_tests_passed=731,core_tests_skipped=4,focused_tests_passed=146,
 package_build_verified=True,strongest_pipeline_saved_metrics_recomputed=True,retained_block_source_recomputed=False,
 prior_result_manifests_unchanged=previous,models_changed=False,original_thresholds_changed=False,production_policy_changed=False,
 reserved_measurement_payloads_opened=False,manuscript_created=False,
 root_files_sha256={n:sha(P/n) for n in ("CURRENT_PROJECT_STATUS.md","README.md")},
 wheel_sha256={str(p.relative_to(R)):sha(p) for p in (R/"runtime"/"wheels").glob("*.whl")})
(R/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ("file_manifest.json","package_audit.json")}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(dict(audit_complete=True,full_project_complete=False,core_tests_passed=731,core_tests_skipped=4,focused_tests_passed=146,
 wheel_verified=True,prior_result_manifests_unchanged=previous,manifest_files=len(manifest)),indent=2))
