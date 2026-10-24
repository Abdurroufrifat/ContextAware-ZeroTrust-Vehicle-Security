"""Deterministic software-only fixtures; public keys must never be deployed."""
from pathlib import Path
import copy, hashlib, itertools, json, sys
from collections import Counter
from dataclasses import replace
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from coact_cps.vehicle.policy import SOURCES, PolicyConfig, VehiclePolicyEngine, VehicleRequest, canonical_json, seal_source, strict_loads
KEYS = {s: hashlib.sha256(('PUBLIC-OFFLINE-FIXTURE-' + s).encode()).digest() for s in SOURCES}
CAP_KEY = hashlib.sha256(b'PUBLIC-OFFLINE-CAPABILITY-FIXTURE').digest()
ACL = {('fleet-service','av-1','vehicle.telemetry','read'), ('fleet-service','av-1','vehicle.planner','update_route')}
CONFIG = PolicyConfig(**json.loads((PROJECT/'configs/vehicle_zero_trust_stage1.json').read_text(encoding='utf-8'))['policy'])
COMPARATORS = ('acl_only','authenticated_context_only','full_policy')
NOW = 100000

def profiles():
    g = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))['profile_grid']
    return [dict(profile_id=f'p{i:03d}',speed_mps=s,acceleration_mps2=a,yaw_rate_rps=y) for i,(s,a,y) in enumerate(itertools.product(g['speed_mps'],g['acceleration_mps2'],g['yaw_rate_rps']))]

def engine():
    return VehiclePolicyEngine(KEYS,CAP_KEY,ACL,CONFIG)

def request():
    return VehicleRequest('fixture-request','fixture-nonce','fleet-service','av-1','vehicle.telemetry','read')

def signals(p):
    return {'identity':dict(subject_id='fleet-service',credential_valid=True,revoked=False,attestation_ok=True),
            'gnss':{'speed_mps':p['speed_mps']},'wheel':{k:p[k] for k in ('speed_mps','acceleration_mps2','yaw_rate_rps')},
            'imu':{k:p[k] for k in ('acceleration_mps2','yaw_rate_rps')},'can':{'anomaly_score':.05},'v2x':{'inconsistency_score':.05,'uncertainty':.05}}

def frames(p,seq=1,now=NOW,changes=None):
    s=signals(p)
    for name,values in (changes or {}).items(): s[name].update(values)
    return {name:seal_source(name,'av-1','fixture-frame',now,seq,values,KEYS[name]) for name,values in s.items()}

def alter(wire,source,resign=True,**changes):
    packet=strict_loads(wire);packet['body'].update(changes)
    return seal_source(**packet['body'],key=KEYS[source]) if resign else canonical_json(packet)

def ingress_catalog():
    rows=[]
    def add(case,kind='attack',expected='DENY',threat='source_fault'):
        rows.append(dict(id=case,kind=kind,expected_full=expected,threat=threat))
    add('benign','benign','ALLOW','none')
    for source in SOURCES:
        for fault in ('missing','stale','future','replay','tamper','bad_type'): add(f'{fault}__{source}',threat=fault)
    for case in ('ttl_at_limit','skew_at_limit'): add(case,'boundary','ALLOW','none')
    for case in ('ttl_over_limit','skew_over_limit','clock_rollback','wrong_vehicle','wrong_frame','malformed','unauthorized_scope','revoked','failed_attestation','single_speed_7','single_acceleration_5','single_yaw_1'): add(case,threat=case)
    for case,expected in [('moderate_read','RESTRICT'),('risk_at_allow','RESTRICT'),('risk_at_restrict','CHALLENGE'),('risk_at_deny','DENY')]: add(case,'control',expected,'none')
    for case in ('common_speed_1','common_speed_7','restart_replay'): add(case,'limitation','ALLOW',case)
    return rows

def meta(catalog,case):
    return next(c for c in catalog if c['id']==case)

def identity_scope(req,bodies):
    i=bodies['identity']['signals']
    return (req.subject_id,req.vehicle_id,req.resource,req.operation) in ACL and req.resource!='vehicle.actuator' and i['subject_id']==req.subject_id and i['credential_valid'] is True and i['revoked'] is False and i['attestation_ok'] is True

def evaluate(e,req,wires,now,comparator):
    if comparator=='acl_only':
        allowed=(req.subject_id,req.vehicle_id,req.resource,req.operation) in ACL and req.resource!='vehicle.actuator'
        return ('ALLOW','stream',['request_acl_satisfied']) if allowed else ('DENY','none',['scope_not_authorized'])
    d=e.evaluate(req,wires,now)
    # Preserve the complete production admission, parser, clock, identity, scope
    # and sequence state. This ablation overrides only the context-risk verdict.
    if comparator=='authenticated_context_only' and d.context_digest:
        bodies={s:strict_loads(w)['body'] for s,w in wires.items()}
        if identity_scope(req,bodies): return 'ALLOW','stream',['authenticated_context_and_acl_satisfied_risk_omitted']
    return d.outcome,d.mode,list(d.reasons)

def ingress_trial(p,case,comparator):
    if comparator not in COMPARATORS: raise ValueError('unknown comparator')
    c=meta(ingress_catalog(),case);e=engine();req=request();now=NOW;wires=frames(p)
    if case.startswith('replay__') or case in ('restart_replay','clock_rollback'):
        if evaluate(e,req,wires,now,comparator)[0]!='ALLOW': raise ValueError('invalid clean admission fixture')
        wires=frames(p,seq=2)
        if case=='restart_replay': e=engine();wires=frames(p)
        elif case=='clock_rollback': now-=1;wires=frames(p,seq=2,now=now)
    if '__' in case:
        fault,source=case.split('__')
        if fault=='missing': del wires[source]
        elif fault in ('stale','future'): wires[source]=alter(wires[source],source,timestamp_ms=now-501 if fault=='stale' else now+1)
        elif fault=='replay': wires[source]=frames(p)[source]
        elif fault in ('bad_type','tamper'):
            s=copy.deepcopy(signals(p)[source]);field='credential_valid' if source=='identity' else next(iter(s));s[field]=1 if source=='identity' else True
            wires[source]=alter(wires[source],source,signals=s,resign=fault!='tamper')
    elif case in ('ttl_at_limit','ttl_over_limit'): wires=frames(p,now=now-(500 if case=='ttl_at_limit' else 501))
    elif case in ('skew_at_limit','skew_over_limit'): wires['gnss']=alter(wires['gnss'],'gnss',timestamp_ms=now-(50 if case=='skew_at_limit' else 51))
    elif case=='wrong_vehicle': wires['gnss']=alter(wires['gnss'],'gnss',vehicle_id='av-other')
    elif case=='wrong_frame': wires['gnss']=alter(wires['gnss'],'gnss',frame_id='other-frame')
    elif case=='malformed': wires['gnss']='{"body":NaN}'
    elif case=='unauthorized_scope': req=replace(req,resource='vehicle.secret')
    elif case in ('revoked','failed_attestation'): wires=frames(p,changes={'identity':{'revoked':True} if case=='revoked' else {'attestation_ok':False}})
    elif case=='single_speed_7': wires=frames(p,changes={'gnss':{'speed_mps':p['speed_mps']+7}})
    elif case=='single_acceleration_5': wires=frames(p,changes={'imu':{'acceleration_mps2':p['acceleration_mps2']+5}})
    elif case=='single_yaw_1': wires=frames(p,changes={'imu':{'yaw_rate_rps':p['yaw_rate_rps']+1}})
    elif case in ('moderate_read','risk_at_allow','risk_at_restrict','risk_at_deny'):
        score={'moderate_read':.3,'risk_at_allow':.25,'risk_at_restrict':.5,'risk_at_deny':.75}[case];wires=frames(p,changes={'can':{'anomaly_score':score}})
    elif case in ('common_speed_1','common_speed_7'):
        shift=1 if case.endswith('_1') else 7;wires=frames(p,changes={s:{'speed_mps':p['speed_mps']+shift} for s in ('gnss','wheel')})
    outcome,mode,reasons=evaluate(e,req,wires,now,comparator);granted=outcome in ('ALLOW','RESTRICT');adverse=c['kind'] in ('attack','limitation')
    return dict(profile_id=p['profile_id'],case_id=case,comparator=comparator,kind=c['kind'],threat=c['threat'],outcome=outcome,mode=mode,reasons=reasons,granted=granted,attack=adverse,attack_success=adverse and granted,expected_full=c['expected_full'],contract_passed=(outcome==c['expected_full']) if comparator=='full_policy' else None)

def capability_catalog():
    rows=[dict(id=c,kind=k,expected_authorized=a,expected_mode=m) for c,k,a,m in [
        ('fresh','benign',True,'stream'),('lease_249','boundary',True,'stream'),('lease_250','attack',False,'none'),('token_tamper','attack',False,'none'),('request_rebind','attack',False,'none'),('token_reuse','attack',False,'none'),('source_replay','attack',False,'none'),('revoked','attack',False,'none'),('context_conflict','attack',False,'none'),('clock_rollback','attack',False,'none'),('downgrade','control',True,'snapshot'),('snapshot_no_upgrade','control',True,'snapshot'),('restart_token_reuse','limitation',True,'stream'),('common_speed_7','limitation',True,'stream')]]
    rows.extend(dict(id='missing__'+s,kind='attack',expected_authorized=False,expected_mode='none') for s in SOURCES)
    return rows

def capability_trial(p,case):
    c=meta(capability_catalog(),case);e=engine();req=request()
    d=e.evaluate(req,frames(p,changes={'can':{'anomaly_score':.3}} if case=='snapshot_no_upgrade' else None),NOW);token=e.issue_capability(d)
    now=NOW+(249 if case=='lease_249' else 250 if case=='lease_250' else 10);wires=frames(p,seq=2,now=now)
    if case in ('token_reuse','restart_token_reuse'):
        if not e.consume_capability(token,req,wires,now).authorized: raise ValueError('invalid first capability-use fixture')
        now+=1;wires=frames(p,seq=3,now=now)
        if case=='restart_token_reuse': e=engine()
    elif case=='token_tamper':
        packet=strict_loads(token);packet['body']['expires_ms']+=1000;token=canonical_json(packet)
    elif case=='request_rebind': req=replace(req,nonce='other-nonce')
    elif case=='source_replay': wires=frames(p)
    elif case=='revoked': wires=frames(p,seq=2,now=now,changes={'identity':{'revoked':True}})
    elif case=='context_conflict': wires=frames(p,seq=2,now=now,changes={'gnss':{'speed_mps':p['speed_mps']+7}})
    elif case=='clock_rollback':
        e.evaluate(replace(req,request_id='later'),frames(p,seq=2,now=now+1),now+1);wires=frames(p,seq=3,now=now)
    elif case=='downgrade': wires=frames(p,seq=2,now=now,changes={'can':{'anomaly_score':.3}})
    elif case.startswith('missing__'): del wires[case.split('__')[1]]
    elif case=='common_speed_7': wires=frames(p,seq=2,now=now,changes={s:{'speed_mps':p['speed_mps']+7} for s in ('gnss','wheel')})
    result=e.consume_capability(token,req,wires,now);adverse=c['kind'] in ('attack','limitation')
    return dict(profile_id=p['profile_id'],case_id=case,kind=c['kind'],authorized=result.authorized,mode=result.mode,reasons=list(result.reasons),expires_ms=result.expires_ms,attack=adverse,attack_success=adverse and result.authorized,contract_passed=result.authorized==c['expected_authorized'] and result.mode==c['expected_mode'])

def reduce_rows(rows):
    adverse=[r for r in rows if r['attack']];contracts=[r['contract_passed'] for r in rows if r['contract_passed'] is not None]
    def grant(r): return r.get('granted',r.get('authorized',False))
    return dict(trials=len(rows),grants=sum(grant(r) for r in rows),attack_trials=len(adverse),attack_grants=sum(grant(r) for r in adverse),attack_grant_fraction=sum(grant(r) for r in adverse)/len(adverse) if adverse else None,outcome_counts=dict(sorted(Counter(r.get('outcome','AUTHORIZED' if r.get('authorized') else 'DENIED') for r in rows).items())),contracts_checked=len(contracts),all_contracts_passed=all(contracts) if contracts else None)
