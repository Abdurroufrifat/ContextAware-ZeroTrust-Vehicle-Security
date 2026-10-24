"""Descriptive matched-scene audit; no new attack evaluation or parameter fitting."""
import math,statistics,sys

CASES=('nominal','wheel_bias_plus3','wheel_drift_to3','wheel_replay_1s_retimed','wheel_replay_1s_original_receipt','wheel_dropout_hold','coordinated_bias_plus3','coordinated_u0','coordinated_u075','protected_copy_missing','protected_copy_stale','protected_copy_compromised','protected_bound_violated','unestablished_real_evidence')
OLD=('legacy_agreement','conditional_anchor_only','conditional_bounded_candidate')
NEW=('cusum_unprotected','ewma_unprotected','cusum_protected','ewma_protected')
POLICIES=OLD+NEW
GROUPS=('all','before','during','after','unavailable')
PAIRS=(('legacy_agreement','cusum_unprotected'),('legacy_agreement','ewma_unprotected'),('conditional_bounded_candidate','conditional_anchor_only'),('conditional_bounded_candidate','cusum_protected'),('conditional_bounded_candidate','ewma_protected'))
STATUSES=('monitored','warmup','invalid_calibration','numerical_failure','missing','unavailable')
COUNTS=('requests','accepted','scored_accepted','accepted_unscorable','accepted_disagreement_gt2_count','trace_bound_violation_count')


def require(condition,message):
    if not condition:raise ValueError(message)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def gate(x):
    return bool(x['requests'] and x['coverage_percent'] is not None and x['coverage_percent']>=95 and
                x['p95_reference_disagreement_mps'] is not None and x['p95_reference_disagreement_mps']<=.5 and
                x['accepted_disagreement_gt2_count']==0 and x['accepted_unscorable']==0)


def validate_cell(x,monitor):
    for key in COUNTS:require(type(x[key]) is int and x[key]>=0,'Invalid count: '+key)
    n=x['requests'];a=x['accepted'];k=x['scored_accepted']
    require(k<=a<=n and x['accepted_unscorable']==a-k,'Acceptance/scoring partition')
    require(x['accepted_disagreement_gt2_count']<=k and x['trace_bound_violation_count']<=a,'Diagnostic counts')
    require((n==0 and x['coverage_percent'] is None) or (n>0 and close(x['coverage_percent'],100*a/n)),'Coverage denominator')
    codes=x['selection_counts'];require(set(codes)=={'abstain','candidate','protected_copy'},'Selection names')
    require(all(type(v) is int and v>=0 for v in codes.values()),'Selection count type')
    require(sum(codes.values())==n and codes['candidate']+codes['protected_copy']==a,'Selection partition')
    for key in ('mae_reference_disagreement_mps','p95_reference_disagreement_mps','max_reference_disagreement_mps'):
        require((not k and x[key] is None) or (k>0 and finite(x[key]) and x[key]>=0),'Scored metric: '+key)
    if k:
        require(x['mae_reference_disagreement_mps']<=x['max_reference_disagreement_mps']+1e-10,'MAE exceeds maximum')
        require(x['p95_reference_disagreement_mps']<=x['max_reference_disagreement_mps']+1e-10,'P95 exceeds maximum')
        require((x['max_reference_disagreement_mps']>2)==(x['accepted_disagreement_gt2_count']>0),'Large-disagreement maximum/count')
    trace=x['max_deviation_from_original_wheel_mps']
    require(trace is None or (finite(trace) and trace>=0),'Trace diagnostic')
    require(type(x['diagnostic_gates_pass']) is bool and x['diagnostic_gates_pass']==gate(x),'Gate flag disagrees')
    if monitor:
        m=x['monitor']
        require(all(type(m[key]) is int and m[key]>=0 for key in ('requests','alarms')+STATUSES),'Monitor count')
        require(m['requests']==n and sum(m[key] for key in STATUSES)==n,'Monitor status partition')
        require(m['alarms']<=m['monitored'],'Alarm exceeds monitored count')
        rate=m['alarm_percent_of_monitored']
        require((not m['monitored'] and rate is None) or (m['monitored']>0 and close(rate,100*m['alarms']/m['monitored'])),'Alarm denominator')
    return gate(x)


def validate_groups(groups,monitor):
    require(set(groups)==set(GROUPS),'Phase inventory')
    for x in groups.values():validate_cell(x,monitor)
    for key in COUNTS:require(groups['all'][key]==sum(groups[g][key] for g in GROUPS[1:]),'Phase count: '+key)
    for key in ('abstain','candidate','protected_copy'):
        require(groups['all']['selection_counts'][key]==sum(groups[g]['selection_counts'][key] for g in GROUPS[1:]),'Phase selection count')
    if monitor:
        for key in ('requests','alarms')+STATUSES:
            require(groups['all']['monitor'][key]==sum(groups[g]['monitor'][key] for g in GROUPS[1:]),'Phase monitor count')
    n=groups['all']['scored_accepted']
    if n:
        mean=math.fsum(groups[g]['mae_reference_disagreement_mps']*groups[g]['scored_accepted'] for g in GROUPS[1:] if groups[g]['scored_accepted'])/n
        require(close(mean,groups['all']['mae_reference_disagreement_mps']),'Phase weighted MAE')
        maximum=max(groups[g]['max_reference_disagreement_mps'] for g in GROUPS[1:] if groups[g]['scored_accepted'])
        require(close(maximum,groups['all']['max_reference_disagreement_mps']),'Phase maximum')
    # Pooled P95 is deliberately not reconstructed from per-phase or per-scene P95s.


def validate_report(report,monitor=True):
    s=report['summary'];v=report['verification'];names=NEW if monitor else OLD;scenes=s['scene_count'];rows=s['baseline_pose_rows']
    require(type(scenes) is int and scenes>0 and type(rows) is int and rows>=0,'Report counts')
    require(s['protocol_id']==('classical-monitor-comparison-v2-numerics' if monitor else 'conditional-protected-copy-v1'),'Protocol')
    require(s['cases']==list(CASES) and s['policy_names']==list(names),'Frozen case/policy inventory')
    require(v['verification_passed'] is True and v['verified_scenes']==scenes,'Prior completion flag')
    require(v['verified_event_rows']==rows*14 and v['verified_decisions']==rows*14*len(names),'Prior decision counts')
    require(v['verified_policy_windows']==(scenes+1)*14*len(names)*5,'Prior window count')
    for key in ('input_lock_sha256','baseline_input_snapshot_sha256'):require(s[key]==v[key] and len(s[key])==64,'Prior lock/snapshot identity')
    for key in ('real_source_protection_established','promotion_allowed'):
        require(s[key] is False and v[key] is False,'Unsupported completion/protection flag')
    require(s['physical_witness_enabled'] is False and s['production_policy_changed'] is False,'Production flags')
    require(set(s['case_metrics'])==set(CASES),'Case metrics inventory')
    for case in CASES:
        require(set(s['case_metrics'][case])==set(names),'Policy metrics inventory')
        for p in names:
            validate_groups(s['case_metrics'][case][p],monitor)
            require(s['case_metrics'][case][p]['all']['requests']==rows,'Missing pooled requests')
    return dict(metadata_and_arithmetic_passed=True,pooled_metric_cells_checked=14*len(names)*5,
                source_reports_completion_flag_checked=True,raw_event_reconstruction_repeated_here=False)


def outcome(candidate,comparator):
    return 'both_pass' if candidate and comparator else 'candidate_only' if candidate else 'comparator_only' if comparator else 'both_fail'


class Pairs:
    def __init__(self):
        self.cells={case:[{g:dict(scenes=0,both_pass=0,candidate_only=0,comparator_only=0,both_fail=0,empty_request_scenes=0) for g in GROUPS} for pair in PAIRS] for case in CASES}
    def add(self,case,index,group,a,b,requests):
        x=self.cells[case][index][group];x['scenes']+=1;x[outcome(a,b)]+=1;x['empty_request_scenes']+=requests==0
    def report(self):
        result={}
        for case,pairs in self.cells.items():
            result[case]=[]
            for groups in pairs:
                result[case].append({g:dict(x,candidate_pass_scenes=x['both_pass']+x['candidate_only'],comparator_pass_scenes=x['both_pass']+x['comparator_only'],candidate_minus_comparator_percentage_points=100*(x['candidate_only']-x['comparator_only'])/x['scenes'] if x['scenes'] else None) for g,x in groups.items()})
        return result


class Calibration:
    def __init__(self):self.models={case:{t:[] for t in ('unprotected','protected')} for case in CASES}
    def add(self,case,track,model):
        n=model['n'];mu=model['mean'];sd=model['sd']
        require(type(n) is int and n>=0,'Calibration n')
        require((n==0 and mu is None) or (n>0 and finite(mu)),'Calibration mean')
        require((n<2 and sd is None) or (n>=2 and finite(sd) and sd>=0),'Calibration sd')
        require(type(model['valid']) is bool and model['valid']==(n>=2 and sd>0),'Calibration validity')
        self.models[case][track].append(model.copy())
    def report(self):
        result={}
        for case,tracks in self.models.items():
            result[case]={}
            for track,models in tracks.items():
                positive=[x['sd'] for x in models if x['sd'] is not None and x['sd']>0]
                result[case][track]=dict(scenes=len(models),valid_scenes=sum(x['valid'] for x in models),short_prefix_scenes=sum(x['n']<2 for x in models),zero_sd_scenes=sum(x['sd']==0 for x in models),subnormal_sd_scenes=sum(x<sys.float_info.min for x in positive),min_positive_sd=min(positive) if positive else None,median_positive_sd=statistics.median(positive) if positive else None,max_positive_sd=max(positive) if positive else None)
        return result
