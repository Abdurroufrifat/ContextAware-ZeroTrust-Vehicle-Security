"""Run short authentication controls and record missing real measurement evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
import auth_guard

HERE=Path(__file__).resolve().parent
CODE=('auth_guard.py','run_evidence.py','test_auth.py','test_report.py','protocol.json','input_evidence/run_report.json')


def digest(p):return hashlib.sha256(auth_guard.safe(p).read_bytes()).hexdigest()


def build_report(prior,test_count):
    s=prior['summary'];v=prior['verification']
    checks=[v['verification_passed'] is True,v['input_lock_sha256']==s['input_lock_sha256'],
            v['baseline_input_snapshot_sha256']==s['baseline_input_snapshot_sha256'],
            v['verified_scenes']==s['scene_count']==len(s['scenes']),
            v['verified_event_rows']==s['baseline_pose_rows']*len(s['scenarios']),
            v['verified_decisions']==v['verified_event_rows']*len(s['policy_names']),
            v['promotion_allowed'] is False,s['production_policy_changed'] is False]
    if not all(checks):raise ValueError('Prior attack evidence metadata is inconsistent')
    nominal=s['scenario_metrics']['nominal']['age_and_agreement']['all']
    coordinated=s['scenario_metrics']['coordinated_bias_plus3']['age_and_agreement']['during']
    return dict(protocol_id='authenticated-report-boundary-v1',
                component_validation=dict(tests_run=test_count,tests_passed=test_count,
                                          scope='message integrity, identity/context binding, receipt/measurement age and persistent sequence checks in local simulation'),
                prior_attack_evidence=dict(scenes=s['scene_count'],pose_rows=s['baseline_pose_rows'],
                                           verified_decisions=v['verified_decisions'],
                                           input_lock_sha256=s['input_lock_sha256'],
                                           nominal_coverage_percent=nominal['coverage_percent'],
                                           nominal_p95_disagreement_mps=nominal['p95_reference_disagreement_mps'],
                                           coordinated_accepted_disagreement_gt2=coordinated['accepted_disagreement_gt2_count']),
                actual_measurement_evidence={k:'not_established' for k in (
                    'native_origin_signatures_and_key_provisioning','separately_protected_measurement_endpoint',
                    'independent_sensor_domain_and_calibration','authenticated_physical_measurement_time',
                    'hardware_or_equivalent_attestation','rollback_protected_receiver_state')},
                limits=dict(protects_against_compromised_signer=False,
                            protects_against_replay_database_snapshot_rollback=False,
                            establishes_physical_measurement_truth=False,
                            establishes_independent_source_trust=False),
                interpretation='Component tests use ephemeral simulation keys. No real CAN records are signed or certified. Missing evidence grants no physical-speed authorization.',
                project_promotion_allowed=False,production_policy_changed=False,physical_witness_enabled=False)


def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)


def run(output):
    output=auth_guard.safe(output)
    if output==HERE or HERE in output.parents and output.name in ('input_evidence','_temporary'):
        raise ValueError('Output conflicts with component inputs or temporary state')
    release=HERE/'sha256.json'
    if release.exists():
        for name,expected in json.loads(release.read_text()).items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid release path')
            if digest(HERE/relative)!=expected:raise ValueError('Component release hash mismatch: '+name)
    frozen={n:digest(HERE/n) for n in CODE}
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'input_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())!=frozen:raise ValueError('Frozen component/input changed')
    elif any(output.iterdir()):
        raise ValueError('Nonempty output without a component lock')
    else:write(lock,frozen)
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'test_results.log').write_text(log.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():
        write(output/'validation_failure.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
        raise ValueError('Component validation failed; see test_results.log')
    if frozen!={n:digest(HERE/n) for n in CODE}:raise ValueError('Component/input changed during validation')
    prior=json.loads((HERE/'input_evidence/run_report.json').read_text())
    report=build_report(prior,result.testsRun)
    report['component_input_lock_sha256']=digest(lock)
    target=output/'validation_report.json'
    if target.exists() and json.loads(target.read_text())!=report:
        raise ValueError('Existing validation report differs; refusing replacement')
    write(target,report)
    print(json.dumps(dict(tests_passed=result.testsRun,
                          prior_scenes=report['prior_attack_evidence']['scenes'],
                          physical_evidence='not_established',project_promotion_allowed=False),sort_keys=True))
    print('EVIDENCE_AUTH_COMPONENT_VALIDATED_NO_PROMOTION')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'local_validation')
    args=parser.parse_args()
    try:run(args.output_dir)
    except (ValueError,OSError,KeyError) as e:
        print('EVIDENCE_VALIDATION_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
