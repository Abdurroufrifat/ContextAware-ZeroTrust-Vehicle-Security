from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent.parent
assert json.loads((R/"independent_verification.json").read_text())["passed"]
status=P/"CURRENT_PROJECT_STATUS.md";backup=R/"PROJECT_STATUS_BEFORE_SOURCE_PILOT.md"
assert not backup.exists()
shutil.copyfile(status,backup)
prefix="""Latest: new independent-source and published-baseline development pilot (2026-10-21).

Implemented and ran two native classical radar baselines on50 fixed public MSC-RAD4R URBAN_D0 training frames:267,125points,9.27MB. All50 files obtained after recovery of two temporary network failures. Corrected only a NumPy scalar-shape compatibility issue in a subclass; original author source/outcomes preserved.
All731 core tests pass,4skip, one existing legacy calibration warning. New14 decoder/time/compatibility tests pass. Offline hash/schema/metric replay passed.
Pilot does not resolve the accuracy blocker: unsigned3D baseline P95 disagreement7.376175m/s against40 matched offline smoothed position-derived reference speeds;35/40 exceed2m/s. Ten requests have no frozen reference match. Kellner signed-versus-unsigned diagnostic is not a validated physical-error comparison. No policy accepts these estimates.
Reference uncertainty, acquisition clock bounds, flag semantics and axis/sign ancestry remain insufficient. No superiority or journal readiness established. Native baseline execution is now available; a fair multi-route candidate comparison and deep-model reproduction are not complete.
Recommended next source: HeRCULES20Hz radar plus separate50Hz INS velocities and calibration. Official access requires the user's email/name/affiliation via its request form. No form submitted.
Full project complete:NO. Original three research work packages remain. Physical witness disabled; production unchanged; original failed gates retained; reserved payloads unopened. No manuscript, deployment or push.
Evidence: research/vehicle_independent_source_qualification/FINDINGS.md, pilot_summary.json, test_validation.json, independent_verification.json.
Prior verified status retained below.

"""
status.write_text(prefix+backup.read_text(encoding="utf-8"),encoding="utf-8")
# Explicit list confined to this module. No recursive access to any parent/protected payload.
files=[p for p in R.iterdir() if p.is_file() and p.name not in {"file_manifest.json","completion.json","package_audit.json"}]
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(R/"file_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
completion={"batch_complete":True,"overall_project_complete":False,"physical_witness_admitted":False,"files":len(files),"core_tests_passed":731,"core_tests_skipped":4,"new_tests_passed":14,"pcd_files":50,"decoded_points":267125,"next_source":"HeRCULES official access request; then freeze development qualification","current_status_sha256":hashlib.sha256(status.read_bytes()).hexdigest(),"reserved_payloads_opened":False}
(R/"completion.json").write_text(json.dumps(completion,indent=2),encoding="utf-8")
print(json.dumps(completion,indent=2))
