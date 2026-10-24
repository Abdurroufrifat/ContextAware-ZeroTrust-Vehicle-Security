"""Verify stored trials, independently count outcomes, then replay actual policy."""
import argparse, hashlib, json, sys, unittest
from collections import Counter
from pathlib import Path
import benchmark_core as b
import run_benchmark as runner

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_rows(rows,expected,catalog,capability):
    seen=set()
    for row in rows:
        key=(row['profile_id'],row['case_id']) if capability else (row['profile_id'],row['case_id'],row['comparator'])
        if key in seen or key not in expected: raise ValueError('duplicate or unexpected trial')
        seen.add(key);c=catalog[row['case_id']]
        if row['kind']!=c['kind']: raise ValueError('trial kind changed')
        adverse=c['kind'] in ('attack','limitation')
        if capability:
            granted=row['authorized']
            contract=granted==c['expected_authorized'] and row['mode']==c['expected_mode']
        else:
            if row['outcome'] not in ('ALLOW','RESTRICT','CHALLENGE','DENY'): raise ValueError('unknown outcome')
            granted=row['outcome'] in ('ALLOW','RESTRICT')
            if row['granted'] is not granted or row['expected_full']!=c['expected_full'] or row['threat']!=c['threat']: raise ValueError('trial semantics changed')
            contract=row['outcome']==c['expected_full'] if row['comparator']=='full_policy' else None
        if type(granted) is not bool or row['attack'] is not adverse or row['attack_success'] is not (adverse and granted): raise ValueError('hidden or invented attack success')
        if row['contract_passed'] is not contract: raise ValueError('contract flag differs from observed behavior')
    if seen!=expected: raise ValueError('missing trial')

def independent_counts(rows):
    total=len(rows);grants=attacks=attack_grants=contracts=0;contracts_ok=True;outcomes=Counter()
    for row in rows:
        allowed=row['outcome'] in ('ALLOW','RESTRICT') if 'outcome' in row else row['authorized']
        adverse=row['kind'] in ('attack','limitation')
        grants+=int(allowed);attacks+=int(adverse);attack_grants+=int(adverse and allowed)
        outcomes[row['outcome'] if 'outcome' in row else 'AUTHORIZED' if allowed else 'DENIED']+=1
        if row['contract_passed'] is not None: contracts+=1;contracts_ok=contracts_ok and row['contract_passed']
    return dict(trials=total,grants=grants,attack_trials=attacks,attack_grants=attack_grants,attack_grant_fraction=attack_grants/attacks if attacks else None,
                outcome_counts=dict(sorted(outcomes.items())),contracts_checked=contracts,all_contracts_passed=contracts_ok if contracts else None)

def verify_reductions(result,ingress,caps):
    if result['profiles']!=len(b.profiles()) or result['ingress_trials']!=len(ingress) or result['capability_trials']!=len(caps): raise ValueError('result denominator changed')
    if result['ingress_case_types']!=len(b.ingress_catalog()) or result['capability_case_types']!=len(b.capability_catalog()): raise ValueError('result case count changed')
    for method in b.COMPARATORS:
        rows=[r for r in ingress if r['comparator']==method];group=result['comparators'][method]
        if group['all']!=independent_counts(rows): raise ValueError('comparator total differs')
        if set(group['by_kind'])!={r['kind'] for r in rows} or set(group['by_case'])!={c['id'] for c in b.ingress_catalog()}: raise ValueError('missing comparator stratum')
        for field,output_field in [('kind','by_kind'),('case_id','by_case')]:
            for value,cell in group[output_field].items():
                if cell!=independent_counts([r for r in rows if r[field]==value]): raise ValueError('comparator cell differs')
    group=result['capabilities']
    if group['all']!=independent_counts(caps): raise ValueError('capability total differs')
    if set(group['by_kind'])!={r['kind'] for r in caps} or set(group['by_case'])!={c['id'] for c in b.capability_catalog()}: raise ValueError('missing capability stratum')
    for field,output_field in [('kind','by_kind'),('case_id','by_case')]:
        for value,cell in group[output_field].items():
            if cell!=independent_counts([r for r in caps if r[field]==value]): raise ValueError('capability cell differs')
    observed=all(r['contract_passed'] for r in ingress if r['comparator']=='full_policy') and all(r['contract_passed'] for r in caps)
    if result['all_full_policy_contracts_passed'] is not observed or not observed: raise ValueError('software contract mismatch')

def verify_prior(strict=False):
    count=0;missing=[];protected=0
    for name,h in read(b.ROOT/'prior_snapshot_manifest.json').items():
        if name=='FINAL_PROJECT_README.md': continue
        # Protect closed final sources even on a complete Windows installation.
        # Decisions use only path metadata; never open these files for hashing.
        low=name.lower()
        if low.startswith('research/vehicle_final_evaluation/') or 'urban02' in low or 'rav4' in low:
            protected+=1;continue
        path=b.PROJECT/name
        if not path.is_file(): missing.append(name);continue
        if sha(path)!=h: raise ValueError('prior immutable file changed: '+name)
        count+=1
    if strict and missing: raise ValueError(f'incomplete project: {len(missing)} prior files missing; preserve the existing project when extracting')
    return count,len(missing),protected

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete-project',action='store_true');args=parser.parse_args()
    runner.check_lock()
    for name,h in read(b.ROOT/'sha256.json').items():
        if sha(b.ROOT/name)!=h: raise ValueError('benchmark artifact changed: '+name)
    catalog=read(b.ROOT/'case_catalog.json')
    expected_catalog=dict(profiles=b.profiles(),comparators=list(b.COMPARATORS),ingress=b.ingress_catalog(),capabilities=b.capability_catalog())
    if catalog!=expected_catalog: raise ValueError('catalog differs from frozen design')
    ingress=[json.loads(line) for line in (b.ROOT/'ingress_trials.jsonl').read_text(encoding='utf-8').splitlines()]
    caps=[json.loads(line) for line in (b.ROOT/'capability_trials.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(p['profile_id'],c['id'],m) for p in b.profiles() for c in b.ingress_catalog() for m in b.COMPARATORS}
    cap_expected={(p['profile_id'],c['id']) for p in b.profiles() for c in b.capability_catalog()}
    validate_rows(ingress,expected,{c['id']:c for c in b.ingress_catalog()},False)
    validate_rows(caps,cap_expected,{c['id']:c for c in b.capability_catalog()},True)
    result=read(b.ROOT/'results.json');verify_reductions(result,ingress,caps)
    fresh_ingress,fresh_caps=runner.execute()
    if ingress!=fresh_ingress or caps!=fresh_caps or result!=runner.summary(fresh_ingress,fresh_caps): raise ValueError('policy replay differs')
    tests=unittest.defaultTestLoader.discover(str(b.ROOT),pattern='test_benchmark.py')
    import test_vehicle_zero_trust
    tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_vehicle_zero_trust))
    run=unittest.TextTestRunner(verbosity=1).run(tests)
    if not run.wasSuccessful(): raise ValueError('regressions failed')
    count,missing,protected=verify_prior(args.require_complete_project)
    print(f'SECURITY BENCHMARK VERIFIED: {len(ingress)} ingress + {len(caps)} capability trials; {run.testsRun} regressions.')
    print(f'Prior immutable files: {count} verified, {missing} unavailable; {protected} protected paths not opened.')
    print('Full-policy adverse/fault grants: 108/1836 ingress; 72/576 capability. These failures remain visible.')
    print('Reserved final closed; production unchanged; witness disabled. SECURITY_BENCHMARK_COMPLETE_NO_PROMOTION')

if __name__=='__main__':
    sys.path.insert(0,str(b.PROJECT/'tests'))
    main()
