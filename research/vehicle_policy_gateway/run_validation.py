"""Short offline gateway validation with ephemeral synthetic reports only."""
import argparse,hashlib,io,json,os,secrets,sys,tempfile,unittest
from pathlib import Path
import policy_gateway as p
import validate_audit
HERE=Path(__file__).resolve().parent
TMP=HERE/'_temporary';TMP.mkdir(exist_ok=True)
CODE=('policy_gateway.py','run_validation.py','validate_audit.py','test_gateway.py','test_audit.py','protocol.json','input_evidence/scene_full_report.json')
DEPENDENCIES=('vehicle_evidence_auth/auth_guard.py','vehicle_scene_comparison/scene_core.py')


def digest(path):return hashlib.sha256(p.auth.safe(path).read_bytes()).hexdigest()


def read(path):return json.loads(p.auth.safe(path).read_text(encoding='utf-8'))


def write(path,data):
    path=p.auth.safe(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def freeze():
    return dict(protocol_id='offline-authenticated-policy-gateway-v1',source_sha256={name:digest(HERE/name) for name in CODE},
                dependency_sha256={name:digest(HERE.parent/name) for name in DEPENDENCIES})


def simulation_controls():
    cases=('nominal','candidate_bias','candidate_tamper','compromised_signers','missing_anchor','elevated_context','lockdown_context','unestablished_assumption','duplicate_replay','wrong_capture')
    results={}
    with tempfile.TemporaryDirectory(dir=TMP) as folder:
        root=Path(folder)
        for index,case in enumerate(cases):
            context=p.auth.Context('demo-vehicle','demo-session','demo-'+str(index));ka=secrets.token_bytes(32);kb=secrets.token_bytes(32)
            state=root/('state-'+str(index)+'.sqlite');p.auth.create_state(state,context)
            guard=p.auth.Guard(state,context,[p.auth.KeyRule('candidate','ka','wheel_speed',ka),p.auth.KeyRule('anchor','kb','protected_speed',kb)])
            policies=[p.SourcePolicy('candidate','wheel_speed','candidate','domain-A',None,0,2000000),p.SourcePolicy('anchor','protected_speed','anchor','domain-B',250,0,2000000,case!='unestablished_assumption')]
            gateway=p.Gateway(guard,policies)
            def packet(role,value):
                body=dict(schema=1,source_id=role,key_id='ka' if role=='candidate' else 'kb',vehicle_id=context.vehicle_id,session_id=context.session_id,capture_id='other-capture' if role=='anchor' and case=='wrong_capture' else context.capture_id,channel='wheel_speed' if role=='candidate' else 'protected_speed',seq=1,measured_us=1000000,speed_mmps=value)
                return p.Packet(json.dumps(p.auth.sign(body,ka if role=='candidate' else kb)),1000000)
            candidate=packet('candidate',8000 if case in ('candidate_bias','compromised_signers') else 5200)
            anchor=packet('anchor',8000 if case=='compromised_signers' else 5000)
            if case=='candidate_tamper':
                message=json.loads(candidate.wire);message['body']['speed_mmps']=8000;candidate=p.Packet(json.dumps(message),1000000)
            if case=='missing_anchor':anchor=None
            risk='elevated' if case=='elevated_context' else 'lockdown' if case=='lockdown_context' else 'normal'
            if case=='duplicate_replay':gateway.evaluate(candidate,anchor,1000000,risk)
            results[case]=gateway.evaluate(candidate,anchor,1000000,risk)
    expected={'nominal':5200,'candidate_bias':5000,'candidate_tamper':5000,'compromised_signers':8000,'missing_anchor':None,'elevated_context':5000,'lockdown_context':None,'unestablished_assumption':None,'duplicate_replay':None,'wrong_capture':None}
    if any(results[name]['research_speed_mmps']!=value for name,value in expected.items()):raise ValueError('Simulation control changed')
    if any(r[key] is not False for r in results.values() for key in ('physical_speed_authorized','measurement_truth_established','independent_source_trust_established','physical_witness_enabled','promotion_allowed')):raise ValueError('Unsupported physical/source claim')
    return results


def run(output):
    output=p.auth.safe(output)
    if output==HERE or output in HERE.parents:raise ValueError('Output overlaps source module')
    for protected in (HERE/'input_evidence',TMP):
        if output==protected or protected in output.parents or output in protected.parents:raise ValueError('Output overlaps inputs/temporary state')
    release=HERE/'sha256.json'
    if release.exists():
        for name,expected in read(release).items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid release path')
            if digest(HERE/relative)!=expected:raise ValueError('Release hash mismatch: '+name)
    frozen=freeze();output.mkdir(parents=True,exist_ok=True);lock=output/'input_lock.json'
    if lock.exists():
        if read(lock)!=frozen:raise ValueError('Frozen gateway/evidence changed; no automatic replacement')
    elif any(output.iterdir()):raise ValueError('Nonempty output without lock')
    else:write(lock,frozen)
    audit=validate_audit.validate(read(HERE/'input_evidence/scene_full_report.json'))
    log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'test_results.log').write_text(log.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():
        write(output/'validation_failure.json',dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
        raise ValueError('Gateway tests failed; see test_results.log')
    controls=simulation_controls()
    if freeze()!=frozen:raise ValueError('Gateway/evidence changed during validation')
    report=dict(protocol_id=frozen['protocol_id'],component_input_lock_sha256=digest(lock),tests_run=result.testsRun,tests_passed=result.testsRun,
                prior_audit=audit,synthetic_control_count=len(controls),synthetic_controls=controls,
                state_semantics='Per-report authenticated sequences commit independently, including reports later denied by policy. No cache or multi-source transaction.',
                remaining_limits=['compromised signer/anchor can authenticate false measurements','old replay database snapshots can restore admission','declared domains/calibration and risk input are provisioned premises; real evidence not established','prior full speed metrics do not automatically transfer to new signed integer-wire gateway'],
                actual_protected_source_evidence='not_established',full_native_speed_experiment_repeated=False,
                production_policy_changed=False,physical_witness_enabled=False,project_promotion_allowed=False)
    target=output/'validation_report.json'
    if target.exists() and read(target)!=report:raise ValueError('Existing validation report differs; refusing replacement')
    write(target,report)
    print(json.dumps(dict(tests_passed=result.testsRun,synthetic_controls=len(controls),prior_paired_tables_checked=audit['paired_tables_checked'],physical_witness_enabled=False,project_promotion_allowed=False),sort_keys=True))
    print('POLICY_GATEWAY_VALIDATED_NO_PROMOTION');print('Saved: '+str(target))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=HERE/'local_validation')
    args=parser.parse_args()
    try:run(args.output_dir)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print('GATEWAY_VALIDATION_STOPPED: '+str(error),file=sys.stderr);sys.exit(1)
