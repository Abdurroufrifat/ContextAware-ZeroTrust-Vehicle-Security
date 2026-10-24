"""Audit recorded rule reasons, without changing or rerunning gateway decisions."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import run_benchmark as base
import verify_benchmark as scalar

HERE = Path(__file__).resolve().parent
CODE = ('audit_explanations.py', 'test_explanations.py', 'explanation_protocol.json')
FIELDS = {'case','policy','index','phase','reference_mmps','output_mmps','selection',
          'candidate_status','anchor_status','reason','latency_ns','replayed_evidence_used',
          'restricted_context_output','physical_speed_authorized'}
LIMITS = ('Authentication does not establish measurement truth. The protected-source '
          'assumption is unverified. Restoring old replay state may admit old evidence. '
          'This is a synthetic research result with no physical-speed authority.')
TEXT = {
    'unsigned_value_control': 'The unsigned control used the candidate value without authentication or policy checks.',
    'authentication_only_control': 'The authentication-only control used the candidate only if its report passed the report guard; anchor and risk policy were not applied.',
    'research_candidate_within_budget': 'The research gateway selected the candidate because the reported difference plus the assumed anchor uncertainty met the fixed 500 mm/s budget.',
    'candidate_outside_budget_fallback': 'The research gateway selected the assumed anchor because the candidate exceeded the fixed 500 mm/s budget.',
    'candidate_unavailable_fallback': 'The research gateway selected the assumed anchor because the candidate report was unavailable or failed a report check.',
    'elevated_context_anchor_only': 'The research gateway used only the assumed anchor because the receiver context was elevated.',
    'anchor_assumption_not_supplied': 'The research gateway withheld a speed because the anchor assumption was not explicitly supplied.',
    'physical_operation_unavailable': 'The gateway withheld a speed because the requested physical operation is unavailable.',
    'lockdown_context': 'The research gateway withheld a speed because the receiver context was lockdown.',
}


def expected_trace(case, policy, i, n):
    """Protocol oracle for recorded reasons/statuses, separate from gateway code."""
    active = n//4 <= i < 3*n//4
    cs = 'authenticated'
    anchor = 'authenticated'
    if active:
        if case == 'candidate_tamper': cs = 'bad_tag'
        elif case == 'candidate_context': cs = 'context_mismatch'
        elif case == 'receiver_clock_rollback': cs = 'clock_rollback'
        elif case in ('candidate_replay','database_snapshot_rollback'):
            if case == 'database_snapshot_rollback' and i == n//4: cs = 'authenticated'
            else: cs = 'measurement_stale' if i-(n//4-1)>5 else 'replay_or_reorder'
    if policy == 'unsigned_candidate_control':
        return 'unsigned_value_control','not_checked','not_checked'
    if policy == 'authenticated_candidate_control':
        return 'authentication_only_control',cs,'not_checked'
    if active and case == 'unsupported_operation':
        return 'physical_operation_unavailable','not_checked','not_checked'
    if active and case == 'risk_lockdown':
        return 'lockdown_context','not_checked','not_checked'
    if case == 'anchor_key_revoked': anchor = 'revoked_key'
    elif active:
        if case == 'anchor_tamper': anchor = 'bad_tag'
        elif case == 'anchor_context': anchor = 'context_mismatch'
        elif case == 'anchor_missing': anchor = 'missing'
        elif case == 'anchor_measurement_stale': anchor = 'measurement_stale'
        elif case == 'receiver_clock_rollback': anchor = 'clock_rollback'
        elif case in ('anchor_replay','database_snapshot_rollback'):
            if case == 'database_snapshot_rollback' and i == n//4: anchor = 'authenticated'
            else: anchor = 'measurement_stale' if i-(n//4-1)>5 else 'replay_or_reorder'
    if anchor != 'authenticated': return 'anchor_'+anchor,cs,anchor
    if case == 'anchor_assumption_missing': return 'anchor_assumption_not_supplied',cs,anchor
    if active and case == 'risk_elevated': return 'elevated_context_anchor_only',cs,anchor
    if cs != 'authenticated': return 'candidate_unavailable_fallback',cs,anchor
    if active and case == 'candidate_bias': return 'candidate_outside_budget_fallback',cs,anchor
    return 'research_candidate_within_budget',cs,anchor


def statement(reason):
    if reason in TEXT: return TEXT[reason]
    statuses = {'bad_tag','context_mismatch','missing','measurement_stale','clock_rollback',
                'replay_or_reorder','revoked_key'}
    if reason.startswith('anchor_') and reason[7:] in statuses:
        return 'The research gateway withheld a speed because the anchor report check returned '+reason[7:]+'.'
    raise ValueError('No validated explanation template')


def analyze(rows, n):
    if type(n) is not int or n<8 or n>4096 or n%4:
        raise ValueError('Invalid frozen request count')
    if len(rows) != len(base.b.CASES)*len(base.b.POLICIES)*n:
        raise ValueError('Explanation ledger denominator differs')
    details=[]; by_policy={}; by_reason={}; errors=Counter(); lookup={}
    for offset,row in enumerate(rows):
        case=base.b.CASES[offset//(3*n)]
        policy=base.b.POLICIES[(offset//n)%3]
        i=offset%n
        phase='before' if i<n//4 else 'during' if i<3*n//4 else 'after'
        if type(row) is not dict or set(row)!=FIELDS: raise ValueError('Unexpected decision schema')
        for key in ('case','policy','phase','selection','candidate_status','anchor_status','reason'):
            if type(row[key]) is not str: raise ValueError('Invalid text field: '+key)
        for key in ('index','reference_mmps','latency_ns'):
            if type(row[key]) is not int or row[key]<0: raise ValueError('Invalid integer: '+key)
        if row['output_mmps'] is not None and (type(row['output_mmps']) is not int or row['output_mmps']<0):
            raise ValueError('Invalid output scalar')
        for key in ('replayed_evidence_used','restricted_context_output','physical_speed_authorized'):
            if type(row[key]) is not bool: raise ValueError('Invalid boolean: '+key)
        for key,value in [('case',case),('policy',policy),('index',i),('phase',phase),('reference_mmps',5000+20*i)]:
            if row[key]!=value: raise ValueError('Decision ordering/reference differs: '+key)
        reason,cs,anchor=expected_trace(case,policy,i,n)
        output,selection=scalar.expected(case,policy,i,n)
        expected=dict(reason=reason,candidate_status=cs,anchor_status=anchor,output_mmps=output,
                      selection=selection,physical_speed_authorized=False)
        mismatch=[key for key,value in expected.items() if row[key]!=value]
        errors.update(mismatch)
        for table,key in ((by_policy,policy),(by_reason,reason)):
            cell=table.setdefault(key,dict(decisions=0,mismatched_decisions=0))
            cell['decisions']+=1;cell['mismatched_decisions']+=bool(mismatch)
        details.append(dict(case=case,policy=policy,index=i,phase=phase,selection=row['selection'],
                            recorded_reason=row['reason'],expected_reason=reason,
                            explanation_valid=not mismatch,mismatched_fields=mismatch,
                            statement=statement(reason) if not mismatch else 'The recorded decision explanation failed validation; do not rely on it.',
                            evidence_limits=LIMITS,explanation_type='rule_trace',
                            measurement_truth_established=False,physical_speed_authorized=False))
        lookup[(case,policy,i)]=(row,bool(mismatch))
    paired=[]
    for case in base.b.CASES[1:]:
        for policy in base.b.POLICIES:
            counts=Counter()
            for i in range(n//4,3*n//4):
                nominal=lookup[('nominal',policy,i)][0];changed=lookup[(case,policy,i)][0]
                counts['paired_active_requests']+=1
                for key in ('reason','selection','output_mmps'):
                    counts[key+'_changed_count']+=changed[key]!=nominal[key]
            paired.append(dict(case=case,policy=policy,**dict(counts)))
    negative={}
    for case,key in [('signed_coordinated_bias','signed_coordinated_bias_gateway'),('database_snapshot_rollback','database_snapshot_rollback_gateway')]:
        values=[lookup[(case,'integrated_gateway',i)] for i in range(n//4,3*n//4)]
        negative[key]=dict(requests=len(values),mismatched_explanations=sum(m for _,m in values),
                           accepted_synthetic_error_gt2_count=sum(r['output_mmps'] is not None and abs(r['output_mmps']-r['reference_mmps'])>2000 for r,_ in values),
                           replayed_evidence_admissions=sum(r['replayed_evidence_used'] for r,_ in values))
    bad=sum(bool(d['mismatched_fields']) for d in details)
    summary=dict(protocol_id='gateway-rule-explanation-audit-v1',decisions=len(rows),
                 matching_decisions=len(rows)-bad,mismatched_decisions=bad,
                 fidelity_percent=100*(len(rows)-bad)/len(rows),explanation_validation_passed=bad==0,
                 mismatch_field_counts=dict(errors),by_policy=by_policy,by_reason=by_reason,
                 paired_scenario_responses=paired,preserved_counterexamples=negative,
                 interpretation='Matching explanations show frozen synthetic rule-trace conformance, not attack detection success, physical truth, learned-model attribution or human usefulness.',
                 human_trust_evaluated=False,learned_model_attribution_evaluated=False,
                 published_framework_reproduced=False,native_vehicle_data_replayed=False,
                 physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)
    return summary,details


def freeze():
    return {name:base.digest(HERE/name) for name in CODE}


def audit(benchmark, output):
    benchmark=base.safe(benchmark);output=base.safe(output)
    if (output==HERE or output in HERE.parents or output==benchmark or benchmark in output.parents
            or output in benchmark.parents or output==base.TMP or base.TMP in output.parents):
        raise ValueError('Audit output overlaps source, benchmark or temporary state')
    if output.exists() and any(output.iterdir()) and not (output/'input_lock.json').is_file():
        raise ValueError('Nonempty output without audit lock; preserving existing files')
    frozen=freeze()
    checked=scalar.verify(benchmark)
    inputs={name:base.digest(benchmark/name) for name in ('input_lock.json','result_hashes.json','summary.json','decisions.jsonl.gz')}
    locked=dict(audit_source_sha256=frozen,benchmark_source_sha256=inputs)
    with gzip.open(base.safe(benchmark/'decisions.jsonl.gz'),'rt',encoding='utf-8') as stream:
        rows=[json.loads(line) for line in stream]
    n=base.read(benchmark/'input_lock.json')['code_and_parameters']['requests_per_case_policy']
    summary,details=analyze(rows,n)
    source_check=dict(benchmark_verification_passed=checked['verification_passed'],
                      benchmark_input_lock_sha256=checked['input_lock_sha256'],
                      benchmark_decisions=checked['verified_decisions'],
                      latency_remeasured=False,discarded_mac_keys_reverified=False)
    encoded=gzip.compress(''.join(json.dumps(d,sort_keys=True,allow_nan=False)+'\n' for d in details).encode('utf-8'),mtime=0)
    if freeze()!=frozen or inputs!={name:base.digest(benchmark/name) for name in inputs}:
        raise ValueError('Audit inputs changed during analysis')
    if (output/'input_lock.json').exists():
        scalar.compare(base.read(output/'input_lock.json'),locked,'audit_input_lock')
        hashes=base.read(output/'result_hashes.json')
        if set(hashes)!={'input_lock.json','summary.json','explanations.jsonl.gz'}: raise ValueError('Audit output inventory differs')
        for name,value in hashes.items():
            if base.digest(output/name)!=value: raise ValueError('Audit output hash mismatch: '+name)
        scalar.compare(base.read(output/'summary.json'),summary,'audit_summary')
        if (output/'explanations.jsonl.gz').read_bytes()!=encoded: raise ValueError('Audit explanations differ')
        scalar.compare(base.read(output/'run_report.json'),dict(summary=summary,source_verification=source_check),'audit_report')
    else:
        output.mkdir(parents=True,exist_ok=True)
        base.write(output/'input_lock.json',locked)
        base.write(output/'summary.json',summary)
        base.atomic(output/'explanations.jsonl.gz',encoded)
        base.write(output/'result_hashes.json',{name:base.digest(output/name) for name in ('input_lock.json','summary.json','explanations.jsonl.gz')})
        base.write(output/'run_report.json',dict(summary=summary,source_verification=source_check))
    return dict(summary=summary,source_verification=source_check)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    default=HERE/'local_benchmark' if (HERE/'local_benchmark').exists() else HERE/'lab_benchmark'
    parser.add_argument('--benchmark-dir',type=Path,default=default)
    parser.add_argument('--output-dir',type=Path,default=HERE/'local_explanation_audit')
    args=parser.parse_args()
    try:
        result=audit(args.benchmark_dir,args.output_dir)
        summary=result['summary']
        print(json.dumps({k:summary[k] for k in ('decisions','matching_decisions','mismatched_decisions','explanation_validation_passed')},sort_keys=True))
        print('EXPLANATION_AUDIT_COMPLETE_NO_PROMOTION' if summary['explanation_validation_passed'] else 'EXPLANATION_AUDIT_MISMATCHES_NO_PROMOTION')
        print('Saved: '+str(base.safe(args.output_dir)/'run_report.json'))
        if not summary['explanation_validation_passed']: sys.exit(1)
    except (ValueError,OSError,KeyError,TypeError,EOFError) as error:
        print('EXPLANATION_AUDIT_STOPPED: '+str(error),file=sys.stderr);sys.exit(1)
