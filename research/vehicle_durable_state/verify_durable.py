"""Read-only independent accounting plus exact replay of the fixed experiment."""
import argparse,hashlib,json,sys,unittest
from pathlib import Path
import run_durable as r
ROOT=r.ROOT;PROJECT=r.PROJECT;b=r.b
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_counts(rows):
    grants=adverse=adverse_grants=0;ok=True
    for row in rows:
        grant=row['outcome'] in ('ALLOW','RESTRICT') if row['phase']=='ingress' else row['authorized']
        bad=row['kind'] in ('attack','limitation')
        grants+=int(grant);adverse+=int(bad);adverse_grants+=int(bad and grant);ok=ok and row['contract_passed']
    return dict(trials=len(rows),grants=grants,adverse_trials=adverse,adverse_grants=adverse_grants,all_contracts_passed=ok)

def verify_rows(rows):
    expected=r.expected_keys();lookup={};changed=[]
    catalogs={'ingress':{c['id']:c for c in b.ingress_catalog()},'capability':{c['id']:c for c in b.capability_catalog()}}
    for row in rows:
        key=(row['phase'],row['method'],row['profile_id'],row['case_id'])
        if key not in expected or key in lookup: raise ValueError('unexpected or duplicate trial')
        lookup[key]=row;c=catalogs[row['phase']][row['case_id']]
        adverse=c['kind'] in ('attack','limitation')
        if row['kind']!=c['kind'] or row['attack'] is not adverse: raise ValueError('changed adverse trial classification')
        if row['phase']=='ingress':
            if row['outcome'] not in ('ALLOW','RESTRICT','CHALLENGE','DENY'): raise ValueError('unknown outcome')
            granted=row['outcome'] in ('ALLOW','RESTRICT')
            if row['granted'] is not granted or row['original_expected']!=c['expected_full']: raise ValueError('ingress semantics changed')
            expected_outcome='DENY' if row['method']=='durable' and row['case_id']=='restart_replay' else c['expected_full']
            contract=row['outcome']==expected_outcome
        else:
            granted=row['authorized']
            expected_outcome=False if row['method']=='durable' and row['case_id']=='restart_token_reuse' else c['expected_authorized']
            if row['original_expected'] is not c['expected_authorized']: raise ValueError('capability original expectation changed')
            contract=granted==expected_outcome and row['mode']==('none' if not expected_outcome else c['expected_mode'])
        if type(granted) is not bool or row['attack_success'] is not (adverse and granted): raise ValueError('hidden or invented adverse grant')
        if row['expected_for_method']!=expected_outcome or row['contract_passed'] is not contract or not contract: raise ValueError('candidate contract mismatch')
    if set(lookup)!=expected: raise ValueError('incomplete fixed trial grid')
    for (phase,method,profile,case),left in lookup.items():
        if method!='memory': continue
        right=lookup[phase,'durable',profile,case]
        fields=('outcome','mode','reasons','granted','attack_success') if phase=='ingress' else ('authorized','mode','reasons','expires_ms','attack_success')
        different=any(left[f]!=right[f] for f in fields)
        allowed=(phase,case) in (('ingress','restart_replay'),('capability','restart_token_reuse'))
        if different!=allowed: raise ValueError('unexpected behavioral change or missing restart fix')
        if different:
            lg=left['granted'] if phase=='ingress' else left['authorized'];rg=right['granted'] if phase=='ingress' else right['authorized']
            if not lg or rg: raise ValueError('restart replay improvement not observed')
            changed.append((phase,profile,case))
    return changed

def verify_totals(result,rows):
    if result['profiles']!=36 or result['total_trials']!=len(rows) or not result['all_contracts_passed']: raise ValueError('summary metadata differs')
    if set(result['groups'])!={'ingress','capability'}: raise ValueError('summary phase missing')
    for phase,catalog in [('ingress',b.ingress_catalog()),('capability',b.capability_catalog())]:
        if set(result['groups'][phase])!=set(r.METHODS): raise ValueError('summary comparator missing')
        for method in r.METHODS:
            group=result['groups'][phase][method];subset=[v for v in rows if v['phase']==phase and v['method']==method]
            if group['all']!=independent_counts(subset): raise ValueError('summary total differs')
            if set(group['by_case'])!={c['id'] for c in catalog}: raise ValueError('summary case missing')
            for case,cell in group['by_case'].items():
                if cell!=independent_counts([v for v in subset if v['case_id']==case]): raise ValueError('summary cell differs')

def prior_integrity(strict=False):
    verified=missing=protected=0
    for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
        if name=='FINAL_PROJECT_README.md': continue
        low=name.lower()
        if low.startswith('research/vehicle_final_evaluation/') or 'urban02' in low or 'rav4' in low:
            protected+=1;continue
        p=PROJECT/name
        if not p.is_file(): missing+=1;continue
        if sha(p)!=h: raise ValueError('preceding file changed: '+name)
        verified+=1
    if strict and missing: raise ValueError(f'incomplete project: {missing} prior files unavailable')
    return verified,missing,protected

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete-project',action='store_true');args=parser.parse_args()
    r.check_lock()
    for name,h in read(ROOT/'sha256.json').items():
        if sha(PROJECT/name)!=h: raise ValueError('durable update artifact changed: '+name)
    rows=[json.loads(line) for line in (ROOT/'trials.jsonl').read_text(encoding='utf-8').splitlines()]
    changes=verify_rows(rows);result=read(ROOT/'results.json');verify_totals(result,rows)
    replay=r.execute()
    if rows!=replay or result!=r.summary(replay): raise ValueError('exact policy replay differs')
    sys.path.insert(0,str(PROJECT/'tests'))
    import test_vehicle_durable_state,test_vehicle_zero_trust,test_replay,test_benchmark
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_vehicle_durable_state,test_vehicle_zero_trust,test_replay,test_benchmark))
    tests=unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful(): raise ValueError('regression failure')
    verified,missing,protected=prior_integrity(args.require_complete_project)
    print(f'DURABLE STATE VERIFIED: {len(rows)} fixed trials; {tests.testsRun} regression tests.')
    print(f'Restart replay grants removed: {len(changes)} / 72. Common signed-sensor attacks remain.')
    print('Adverse ingress grants: 108 -> 72 / 1836; capability grants: 72 -> 36 / 576.')
    print(f'Prior files: {verified} verified, {missing} unavailable, {protected} protected paths not opened.')
    print('Trusted local state required; witness disabled; reserved final closed. DURABLE_STATE_COMPLETE_NO_PROMOTION')
if __name__=='__main__': main()
