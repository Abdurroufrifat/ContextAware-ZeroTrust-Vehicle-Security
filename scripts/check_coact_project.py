"""Audit software/evidence integrity while reporting research failures separately."""
from pathlib import Path
import os,subprocess,sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest_path=ROOT/'PROJECT_MANIFEST.json'
if manifest_path.is_file():
 manifest=json.loads(manifest_path.read_text())
 for name,expected in manifest.items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
   raise SystemExit('Project file verification failed: '+name)
 print('PROJECT SNAPSHOT INTEGRITY: PASS ('+str(len(manifest))+' files)',flush=True)
checks=[ROOT/'scripts/audit_vehicle_session_evidence.py',ROOT/'research/vehicle_final_evaluation/verify_final.py',ROOT/'research/vehicle_temporal_policy/verify_temporal.py',ROOT/'research/vehicle_cadence_policy/verify_cadence.py',ROOT/'research/vehicle_reference_check/verify_reference.py',ROOT/'research/vehicle_radar_estimator/verify_estimator.py',ROOT/'research/vehicle_radar_multidrive/verify_multidrive.py',ROOT/'research/vehicle_radar_tail_audit/verify_tail.py',ROOT/'research/vehicle_radar_quality_guard/verify_quality.py',ROOT/'research/vehicle_continuous_radar/verify_continuous.py',ROOT/'research/vehicle_availability_audit/verify_availability.py',ROOT/'research/vehicle_masked_history/verify_masked.py',ROOT/'research/vehicle_causal_filter/verify_filter.py',ROOT/'research/vehicle_night_training/verify_night.py',ROOT/'research/vehicle_context_boosting/verify_context.py',ROOT/'research/vehicle_route_validation/verify_routes.py',ROOT/'research/vehicle_matched_baselines/verify_matched.py',ROOT/'research/vehicle_failure_diagnosis/verify_diagnosis.py',ROOT/'research/vehicle_trace_audit/verify_trace.py',ROOT/'research/vehicle_magnitude_ablation/verify_magnitude.py',ROOT/'research/vehicle_paired_robustness/verify_paired.py',ROOT/'research/vehicle_signed_magnitude_fusion/verify_fusion.py',ROOT/'research/vehicle_auxiliary_imu/verify_auxiliary.py',ROOT/'research/vehicle_livox_collection/verify_collection.py',ROOT/'research/vehicle_multisource_imu/verify_multisource.py',ROOT/'research/vehicle_signed_axis_correction/verify_axis.py',ROOT/'research/vehicle_remaining_failure_audit/verify_remaining.py',ROOT/'research/vehicle_frame_reference_audit/verify_frame_reference.py']
for path in checks:
 if not path.is_file():raise SystemExit('Required project file missing: '+str(path))
 env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'scripts'),env.get('PYTHONPATH','')])
 p=subprocess.run([sys.executable,str(path)],cwd=ROOT,env=env)
 if p.returncode:raise SystemExit('Project verification failed in '+path.name+'; copy the error output for repair.')
print('CONSOLIDATED SOFTWARE AND EVIDENCE CHECK: PASS')
print('LATEST RESEARCH DECISION: FRAME_REFERENCE_AUDIT_COMPLETE_NO_PROMOTION; MODEL UNCHANGED; ACCURACY FAILURES REMAIN; WITNESS DISABLED.')
print('See research/vehicle_frame_reference_audit/FRAME_REFERENCE_DECISION.md.')
