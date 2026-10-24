"""Fixed offline gateway benchmark. Timings describe the execution host only."""
import argparse,gzip,hashlib,json,os,platform,sqlite3,sys,tempfile
from pathlib import Path
import benchmark_core as b
import verify_benchmark as verify
HERE=Path(__file__).resolve().parent
TMP=HERE/'_temporary';TMP.mkdir(exist_ok=True)
N=128
CODE=('benchmark_core.py','run_benchmark.py','verify_benchmark.py','test_benchmark.py','protocol.json')
DEPENDENCIES=('vehicle_policy_gateway/policy_gateway.py','vehicle_policy_gateway/protocol.json','vehicle_evidence_auth/auth_guard.py')


def safe(path):return b.p.auth.safe(path)
def digest(path):return hashlib.sha256(safe(path).read_bytes()).hexdigest()
def read(path):return json.loads(safe(path).read_text(encoding='utf-8'))


def atomic(path,data):
    path=safe(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def write(path,data):atomic(path,(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8'))


def freeze():
    protocol=read(HERE/'protocol.json')
    if protocol['cases']!=list(b.CASES) or protocol['policies']!=list(b.POLICIES) or protocol['requests_per_case_policy']!=N:raise ValueError('Protocol/code inventory differs')
    return dict(protocol_id=protocol['id'],requests_per_case_policy=N,source_sha256={n:digest(HERE/n) for n in CODE},dependency_sha256={n:digest(HERE.parent/n) for n in DEPENDENCIES})


def environment():
    return dict(python=platform.python_version(),os=platform.system(),architecture=platform.machine(),logical_cpus=os.cpu_count(),sqlite=sqlite3.sqlite_version,
                timing_clock='time.perf_counter_ns',storage_medium='not characterized',hardware_scope='execution host only; not a vehicle or deadline certification')


def summary(metrics,runtime,lock):
    return dict(protocol_id='gateway-protocol-latency-benchmark-v1',input_lock_sha256=lock,requests_per_case_policy=N,
                cases=list(b.CASES),policies=list(b.POLICIES),phases=list(b.GROUPS),total_decisions=18*3*N,case_metrics=metrics,
                execution_environment=runtime,reference_kind='known generated scalar speed; not independent physical truth',
                control_interpretation='Unsigned and authentication-only component controls have less information/authorization logic than the gateway; not full published AV defense baselines.',
                timing_scope='Decision call includes parsing, HMAC and replay-state transactions where used; excludes signing, setup, attack generation, explicit snapshot restore, logging and verification. All rejections included.',
                negative_controls=dict(compromised_signer_gateway_large_disagreements=metrics['signed_coordinated_bias']['integrated_gateway']['during']['accepted_synthetic_error_gt2_count'],
                                       restored_state_gateway_replayed_evidence_admissions=metrics['database_snapshot_rollback']['integrated_gateway']['all']['replayed_evidence_used_count']),
                scenario_conformance_is_not_security_success=True,latency_deadline_claimed=False,held_out_driving_evaluation=False,native_vehicle_data_replayed=False,
                real_source_protection_established=False,physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)


def run(output):
    output=safe(output)
    if output==HERE or output in HERE.parents or TMP==output or TMP in output.parents:raise ValueError('Output overlaps source/temporary state')
    frozen=freeze();output.mkdir(parents=True,exist_ok=True);lock=output/'input_lock.json'
    if lock.exists():
        stored=read(lock);verify.compare(stored['code_and_parameters'],frozen,'frozen_code_changed')
    else:
        if any(output.iterdir()):raise ValueError('Nonempty output without lock')
        write(lock,dict(code_and_parameters=frozen,execution_environment=environment()))
    if (output/'summary.json').exists():checked=verify.verify(output)
    else:
        if any(p.name!='input_lock.json' for p in output.iterdir()):raise ValueError('Partial benchmark preserved; use a new output directory')
        metrics={};lines=[]
        for number,case in enumerate(b.CASES,1):
            metrics[case]={}
            for policy in b.POLICIES:
                reducer=b.Metrics()
                with tempfile.TemporaryDirectory(dir=TMP) as folder:
                    for event in b.trial(case,policy,Path(folder),N):
                        reducer.add(event);lines.append(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
                metrics[case][policy]=reducer.report()
            print('Benchmarked synthetic cases: '+str(number)+'/18',flush=True)
        if freeze()!=frozen:raise ValueError('Code/parameters changed during benchmark')
        atomic(output/'decisions.jsonl.gz',gzip.compress(''.join(lines).encode('utf-8'),mtime=0));del lines
        write(output/'summary.json',summary(metrics,read(lock)['execution_environment'],digest(lock)))
        write(output/'result_hashes.json',{n:digest(output/n) for n in ('input_lock.json','summary.json','decisions.jsonl.gz')})
        checked=verify.verify(output)
    write(output/'verification.json',checked);write(output/'run_report.json',dict(summary=read(output/'summary.json'),verification=checked))
    print(json.dumps(checked,sort_keys=True),flush=True);print('GATEWAY_BENCHMARK_COMPLETE_NO_PROMOTION',flush=True)
    print('Saved: '+str(output/'run_report.json'),flush=True)
    return checked


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=HERE/'local_benchmark');args=parser.parse_args()
    try:run(args.output_dir)
    except (ValueError,OSError,KeyError,TypeError,EOFError) as error:
        print('GATEWAY_BENCHMARK_STOPPED: '+str(error),file=sys.stderr);sys.exit(1)
