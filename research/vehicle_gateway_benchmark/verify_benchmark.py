"""Separate scenario oracle and scalar reductions; timing/MAC keys are not replayed."""
import gzip,json,math
import benchmark_core as b


def expected(case,policy,i,n):
    during=n//4<=i<3*n//4;v=5000+20*i;candidate=v+200;anchor=v+(250 if i%2 else 0)
    if during and case in ('candidate_bias','candidate_tamper','signed_coordinated_bias'):candidate+=3000
    if during and case in ('candidate_replay','database_snapshot_rollback'):candidate=5000+20*(n//4-1)+200
    if policy=='unsigned_candidate_control':return candidate,'candidate'
    if policy=='authenticated_candidate_control':
        blocked=during and (case in ('candidate_tamper','candidate_replay','candidate_context','receiver_clock_rollback') or (case=='database_snapshot_rollback' and i!=n//4))
        return (None,'abstain') if blocked else (candidate,'candidate')
    if case in ('anchor_assumption_missing','anchor_key_revoked'):return None,'abstain'
    if during:
        if case in ('anchor_tamper','anchor_replay','anchor_context','anchor_missing','risk_lockdown','unsupported_operation','anchor_measurement_stale','receiver_clock_rollback'):return None,'abstain'
        if case=='database_snapshot_rollback':return (candidate,'candidate') if i==n//4 else (None,'abstain')
        if case in ('candidate_bias','candidate_tamper','candidate_replay','candidate_context','risk_elevated'):return anchor,'anchor'
    return candidate,'candidate'


def compare(actual,expected,path='result'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError(path+': keys differ')
        for key in expected:compare(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError(path+': list differs')
        for i,value in enumerate(expected):compare(actual[i],value,path+'.'+str(i))
    elif isinstance(expected,float):
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):raise ValueError(path+': numeric value differs')
    elif type(actual) is not type(expected) or actual!=expected:raise ValueError(path+': value differs')


def percentile(data,q):
    values=sorted(data)
    if not values:return None
    at=q*(len(values)-1);lo=math.floor(at);hi=math.ceil(at)
    return values[lo] if hi==lo else values[lo]*(hi-at)+values[hi]*(at-lo)


def reduce(rows):
    n=len(rows);accepted=[r for r in rows if r['output_mmps'] is not None]
    errors=[abs(r['output_mmps']-r['reference_mmps']) for r in accepted];times=[r['latency_ns'] for r in rows]
    k=len(errors);coverage=100*k/n if n else None;p95=percentile(errors,.95);bad=sum(e>2000 for e in errors)
    return dict(requests=n,accepted=k,coverage_percent=coverage,mae_synthetic_error_mps=sum(errors)/(k*1000) if k else None,
                p95_synthetic_error_mps=p95/1000 if p95 is not None else None,max_synthetic_error_mps=max(errors)/1000 if k else None,
                accepted_synthetic_error_gt2_count=bad,selection_counts={name:sum(r['selection']==name for r in rows) for name in ('candidate','anchor','abstain')},
                replayed_evidence_used_count=sum(r['replayed_evidence_used'] for r in rows),restricted_context_output_count=sum(r['restricted_context_output'] for r in rows),
                latency_samples=len(times),median_decision_latency_ms=percentile(times,.5)/1000000 if times else None,
                p95_decision_latency_ms=percentile(times,.95)/1000000 if times else None,max_decision_latency_ms=max(times)/1000000 if times else None,
                diagnostic_gates_pass=bool(n and coverage>=95 and p95 is not None and p95<=500 and bad==0))


def verify(output):
    import run_benchmark as run
    output=run.safe(output);lock=run.read(output/'input_lock.json');compare(lock['code_and_parameters'],run.freeze(),'frozen_code_parameters')
    hashes=run.read(output/'result_hashes.json')
    if set(hashes)!={'input_lock.json','summary.json','decisions.jsonl.gz'}:raise ValueError('Benchmark artifact inventory differs')
    for name,value in hashes.items():
        if run.digest(output/name)!=value:raise ValueError('Benchmark artifact hash mismatch: '+name)
    metrics={};count=0;n=lock['code_and_parameters']['requests_per_case_policy']
    with gzip.open(output/'decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for case in b.CASES:
            metrics[case]={}
            for policy in b.POLICIES:
                rows=[]
                for i in range(n):
                    line=stream.readline()
                    if not line:raise ValueError('Truncated benchmark ledger')
                    row=json.loads(line);phase='before' if i<n//4 else 'during' if i<3*n//4 else 'after'
                    for name,value in [('case',case),('policy',policy),('index',i),('phase',phase),('reference_mmps',5000+20*i),('physical_speed_authorized',False)]:compare(row[name],value,name)
                    value,selection=expected(case,policy,i,n);compare(row['output_mmps'],value,'oracle.output');compare(row['selection'],selection,'oracle.selection')
                    if type(row['latency_ns']) is not int or row['latency_ns']<0:raise ValueError('Invalid measured latency')
                    active=phase=='during';replay=False
                    if active:
                        if policy=='unsigned_candidate_control':replay=case in ('candidate_replay','database_snapshot_rollback')
                        else:replay=case=='database_snapshot_rollback' and i==n//4
                    compare(row['replayed_evidence_used'],replay,'replayed_evidence_used')
                    restricted=bool(active and case in ('risk_lockdown','unsupported_operation') and value is not None)
                    compare(row['restricted_context_output'],restricted,'restricted_context_output')
                    rows.append(row);count+=1
                metrics[case][policy]={g:reduce(rows if g=='all' else [r for r in rows if r['phase']==g]) for g in b.GROUPS}
        if stream.readline():raise ValueError('Extra benchmark ledger row')
    expected_summary=run.summary(metrics,lock['execution_environment'],run.digest(output/'input_lock.json'))
    compare(run.read(output/'summary.json'),expected_summary,'summary')
    compare(lock['code_and_parameters'],run.freeze(),'unchanged_code_parameters')
    return dict(verification_passed=True,verified_decisions=count,verified_scenarios=len(b.CASES),verified_policies=len(b.POLICIES),verified_policy_phase_cells=18*3*4,
                input_lock_sha256=expected_summary['input_lock_sha256'],method='Separate deterministic protocol outcome oracle and scalar synthetic-error/timing reduction',
                discarded_mac_keys_reverified=False,latency_remeasured=False,real_source_protection_established=False,promotion_allowed=False)
