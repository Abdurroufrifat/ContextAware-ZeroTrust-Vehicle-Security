"""Freeze once before scoring; subsequently reproduce the fixed development matrix."""
import argparse, hashlib, json
from pathlib import Path
import benchmark_core as b
ROOT=b.ROOT
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def check_lock():
    protocol=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json')
    if protocol['promotion_allowed'] or protocol['witness_enabled'] or protocol['reserved_final_opened']: raise ValueError('scope changed')
    if sha(ROOT/'protocol.json')!=lock['protocol_sha256']: raise ValueError('protocol changed')
    for name,h in protocol['input_sha256'].items():
        if sha(b.PROJECT/name)!=h: raise ValueError('production input changed: '+name)
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError('frozen benchmark changed: '+name)

def freeze():
    path=ROOT/'analysis_lock.json'
    if path.exists(): raise ValueError('refusing to overwrite existing freeze')
    write(path,dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256={n:sha(ROOT/n) for n in ('benchmark_core.py','run_benchmark.py')},
                    before_first_full_scoring=True,external_registration=False))
    check_lock()

def execute():
    check_lock()
    ingress=[b.ingress_trial(p,c['id'],method) for p in b.profiles() for c in b.ingress_catalog() for method in b.COMPARATORS]
    caps=[b.capability_trial(p,c['id']) for p in b.profiles() for c in b.capability_catalog()]
    return ingress,caps

def summary(ingress,caps):
    out=dict(status='DEVELOPMENT_BENCHMARK_COMPLETE_NO_PROMOTION',physical_claim=False,independent_published_baseline_comparison=False,
             profiles=len(b.profiles()),ingress_case_types=len(b.ingress_catalog()),capability_case_types=len(b.capability_catalog()),
             ingress_trials=len(ingress),capability_trials=len(caps),comparators={},capabilities={},limitations={})
    for method in b.COMPARATORS:
        rows=[r for r in ingress if r['comparator']==method]
        out['comparators'][method]={'all':b.reduce_rows(rows),'by_kind':{k:b.reduce_rows([r for r in rows if r['kind']==k]) for k in sorted({r['kind'] for r in rows})},
                                    'by_case':{c['id']:b.reduce_rows([r for r in rows if r['case_id']==c['id']]) for c in b.ingress_catalog()}}
    out['capabilities']={'all':b.reduce_rows(caps),'by_kind':{k:b.reduce_rows([r for r in caps if r['kind']==k]) for k in sorted({r['kind'] for r in caps})},
                         'by_case':{c['id']:b.reduce_rows([r for r in caps if r['case_id']==c['id']]) for c in b.capability_catalog()}}
    out['all_full_policy_contracts_passed']=all(r['contract_passed'] for r in ingress if r['comparator']=='full_policy') and all(r['contract_passed'] for r in caps)
    out['limitations']={c['id']:out['comparators']['full_policy']['by_case'][c['id']] for c in b.ingress_catalog() if c['kind']=='limitation'}
    return out

def lines(rows): return ''.join(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for row in rows)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze: freeze()
    ingress,caps=execute();result=summary(ingress,caps)
    write(ROOT/'case_catalog.json',dict(profiles=b.profiles(),comparators=b.COMPARATORS,ingress=b.ingress_catalog(),capabilities=b.capability_catalog()))
    (ROOT/'ingress_trials.jsonl').write_text(lines(ingress),encoding='utf-8')
    (ROOT/'capability_trials.jsonl').write_text(lines(caps),encoding='utf-8')
    write(ROOT/'results.json',result)
    if not result['all_full_policy_contracts_passed']: raise ValueError('software contract mismatch; retain all results for diagnosis')
    print(f"SECURITY BENCHMARK: {len(ingress)} ingress + {len(caps)} capability trials. Known attack grants retained. NO PROMOTION.")
if __name__=='__main__': main()
