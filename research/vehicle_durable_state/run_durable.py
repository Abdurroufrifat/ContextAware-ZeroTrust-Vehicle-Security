"""Same fixed software cases, compared with/without trusted durable state."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent
PREVIOUS=ROOT.parent/'vehicle_security_benchmark'
sys.path.insert(0,str(PREVIOUS));sys.path.insert(0,str(PROJECT/'src'))
import benchmark_core as b
from coact_cps.vehicle.durable_policy import DurableVehiclePolicy
METHODS=('memory','durable')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def lines(rows): return ''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rows)

def expected_keys():
    return {(phase,method,p['profile_id'],case['id']) for phase,cases in [('ingress',b.ingress_catalog()),('capability',b.capability_catalog())] for method in METHODS for p in b.profiles() for case in cases}

def check_lock():
    protocol=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json')
    if any(protocol[k] for k in ('models_fitted','thresholds_changed','witness_enabled','reserved_final_opened','promotion_allowed')): raise ValueError('scope changed')
    if lock['protocol_sha256']!=sha(ROOT/'protocol.json'): raise ValueError('protocol changed')
    for name,h in {**protocol['input_sha256'],**lock['code_sha256']}.items():
        if sha(PROJECT/name)!=h: raise ValueError('locked input changed: '+name)

def freeze():
    f=ROOT/'analysis_lock.json'
    if f.exists(): raise ValueError('refusing to replace frozen analysis')
    names=['src/coact_cps/vehicle/durable_policy.py','research/vehicle_durable_state/run_durable.py','research/vehicle_durable_state/process_worker.py']
    write(f,dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256={n:sha(PROJECT/n) for n in names},initial_freeze_before_full_scoring=True,external_registration=False))
    check_lock()

def invoke(phase,p,case,method):
    if method=='memory':
        return b.ingress_trial(p,case,'full_policy') if phase=='ingress' else b.capability_trial(p,case)
    with tempfile.TemporaryDirectory(prefix='coact-durable-') as directory:
        db=Path(directory)/'replay.sqlite'
        DurableVehiclePolicy.provision(db,b.KEYS,b.CAP_KEY,b.ACL,b.CONFIG)
        def factory(): return DurableVehiclePolicy(db,b.KEYS,b.CAP_KEY,b.ACL,b.CONFIG)
        with patch.object(b,'engine',factory):
            return b.ingress_trial(p,case,'full_policy') if phase=='ingress' else b.capability_trial(p,case)

def execute():
    check_lock();rows=[]
    previous_ingress={(r['profile_id'],r['case_id']):r for r in [json.loads(v) for v in (PREVIOUS/'ingress_trials.jsonl').read_text(encoding='utf-8').splitlines()] if r['comparator']=='full_policy'}
    previous_caps={(r['profile_id'],r['case_id']):r for r in [json.loads(v) for v in (PREVIOUS/'capability_trials.jsonl').read_text(encoding='utf-8').splitlines()]}
    for phase,cases in [('ingress',b.ingress_catalog()),('capability',b.capability_catalog())]:
        for p in b.profiles():
            for c in cases:
                for method in METHODS:
                    raw=invoke(phase,p,c['id'],method)
                    old=(previous_ingress if phase=='ingress' else previous_caps)[p['profile_id'],c['id']]
                    if method=='memory' and raw!=old: raise ValueError('original benchmark replay changed')
                    row=dict(raw,phase=phase,method=method)
                    if phase=='ingress':
                        row.pop('comparator');row['original_expected']=row.pop('expected_full')
                        expected='DENY' if method=='durable' and c['id']=='restart_replay' else c['expected_full']
                        row['expected_for_method']=expected;row['contract_passed']=row['outcome']==expected
                    else:
                        row['original_expected']=c['expected_authorized']
                        expected=False if method=='durable' and c['id']=='restart_token_reuse' else c['expected_authorized']
                        row['expected_for_method']=expected
                        row['contract_passed']=row['authorized']==expected and row['mode']==('none' if not expected else c['expected_mode'])
                    rows.append(row)
    return rows

def counts(rows):
    adverse=[r for r in rows if r['attack']]
    def grant(r): return r.get('granted',r.get('authorized',False))
    return dict(trials=len(rows),grants=sum(grant(r) for r in rows),adverse_trials=len(adverse),adverse_grants=sum(grant(r) for r in adverse),all_contracts_passed=all(r['contract_passed'] for r in rows))

def summary(rows):
    groups={}
    for phase,cases in [('ingress',b.ingress_catalog()),('capability',b.capability_catalog())]:
        groups[phase]={}
        for method in METHODS:
            subset=[r for r in rows if r['phase']==phase and r['method']==method]
            groups[phase][method]=dict(all=counts(subset),by_case={c['id']:counts([r for r in subset if r['case_id']==c['id']]) for c in cases})
    return dict(status='DURABLE_STATE_COMPLETE_NO_WITNESS_PROMOTION',profiles=len(b.profiles()),total_trials=len(rows),groups=groups,all_contracts_passed=all(r['contract_passed'] for r in rows),storage_trust_required=True,physical_claim=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze: freeze()
    rows=execute();result=summary(rows)
    (ROOT/'trials.jsonl').write_text(lines(rows),encoding='utf-8');write(ROOT/'results.json',result)
    if not result['all_contracts_passed']: raise ValueError('retained software contract failure')
    print('Durable-state fixed matrix:',len(rows),'trials; original results preserved.')
if __name__=='__main__': main()
