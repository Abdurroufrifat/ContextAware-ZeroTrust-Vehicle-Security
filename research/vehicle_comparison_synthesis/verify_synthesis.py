"""Independent scalar metrics and full route-mixture reconstruction."""
import argparse,csv,hashlib,itertools,json,math,unittest
from pathlib import Path
import numpy as np
import synthesis_core as c
ROOT=c.ROOT;PROJECT=c.PROJECT
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantile(values):
    values=sorted(float(v) for v in values);at=(len(values)-1)*.95;low=math.floor(at);high=math.ceil(at)
    return values[low]+(at-low)*(values[high]-values[low])
def scalar_metrics(y,p):
    errors=[float(a)-float(b) for a,b in zip(p,y)];absolute=[abs(x) for x in errors];n=len(y)
    return dict(count=n,mae_mps=math.fsum(absolute)/n,rmse_mps=math.sqrt(math.fsum(v*v for v in errors)/n),p95_mps=quantile(absolute),maximum_mps=max(absolute),bias_mps=math.fsum(errors)/n,above_half_mps=sum(x>.5 for x in absolute),above_2_mps=sum(x>2 for x in absolute))
def equal(a,b):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b): raise ValueError('metric schema differs')
        for k in a: equal(a[k],b[k])
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b): raise ValueError('list differs')
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,float):
        if type(b) not in (int,float) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10): raise ValueError('numeric metric differs')
    elif a!=b: raise ValueError('metric differs')

def verify_metrics(v):
    with np.load(c.RESEARCH/'vehicle_matched_baselines/oof_predictions.npz',allow_pickle=False) as z:base={k:z[k] for k in z.files}
    with np.load(c.RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz',allow_pickle=False) as z:axis={k:z[k] for k in z.files}
    members=read(c.RESEARCH/'vehicle_route_validation/membership.json');y=base['target_mps']
    np.testing.assert_array_equal(y,axis['target_mps']);np.testing.assert_array_equal(base['fold_id'],axis['fold_id'])
    p={n:base[n+'__prediction_mps'] for n in c.METHODS[:3]};p.update({n:axis[n+'__prediction_mps'] for n in c.METHODS[3:]})
    if len(members)!=27178 or v['cohort_rows']!=len(members) or v['methods']!=list(c.METHODS): raise ValueError('fixed cohort changed')
    def cell(ids):return {n:scalar_metrics([y[i] for i in ids],[p[n][i] for i in ids]) for n in c.METHODS}
    equal(v['pooled'],cell(range(len(y))));cells=6
    for field,section in [('group','by_route'),('route','by_recording'),('window_id','by_window')]:
        labels=sorted({m[field] for m in members})
        if set(v[section])!=set(labels): raise ValueError('missing stratum')
        for label in labels:
            ids=[i for i,m in enumerate(members) if m[field]==label];equal(v[section][label],cell(ids));cells+=6
    route_names=sorted({m['group'] for m in members});expected=list(itertools.product(route_names,repeat=3))
    if len(v['route_mixtures'])!=27 or v['distinct_route_count_mixtures']!=10: raise ValueError('route mixture missing')
    checked=[]
    for entry,draw in zip(v['route_mixtures'],expected):
        ids=[i for group in draw for i,m in enumerate(members) if m['group']==group]
        if entry['routes']!=list(draw) or entry['rows']!=len(ids): raise ValueError('route mixture identity differs')
        scores=cell(ids);equal(entry['metrics'],scores);cells+=6
        delta={n:{k:scores[n][k]-scores[c.REFERENCE][k] for k in c.DELTA_METRICS} for n in c.METHODS}
        equal(entry['delta_vs_original'],delta);checked.append(delta)
    ranges={n:{k:dict(minimum=min(v[n][k] for v in checked),maximum=max(v[n][k] for v in checked),negative_count=sum(v[n][k]<0 for v in checked),draws=27) for k in c.DELTA_METRICS} for n in c.METHODS}
    equal(v['route_mixture_delta_range'],ranges)
    equal(v['failed_windows'],{n:[label for label,s in v['by_window'].items() if s[n]['p95_mps']>.5] for n in c.METHODS})
    equal(v['failed_recordings'],{n:[label for label,s in v['by_recording'].items() if s[n]['p95_mps']>.5] for n in c.METHODS})
    return cells

def verify_csv(v):
    expected={}
    for population,groups in [('pooled',{'all':v['pooled']}),('route',v['by_route']),('recording',v['by_recording']),('window',v['by_window'])]:
        for label,models in groups.items():
            for model,values in models.items():expected[population,label,model]=values
    seen=set()
    with (ROOT/'comparison.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=row.pop('population'),row.pop('label'),row.pop('model')
            if key not in expected or key in seen: raise ValueError('duplicate/unknown CSV row')
            seen.add(key);values={k:int(value) if k in ('count','above_half_mps','above_2_mps') else float(value) for k,value in row.items()};equal(values,expected[key])
    if seen!=set(expected):raise ValueError('missing CSV row')

def verify_prior(strict=False):
    verified=missing=protected=0
    for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
        if name=='FINAL_PROJECT_README.md':continue
        low=name.lower()
        if low.startswith('research/vehicle_final_evaluation/') or 'urban02' in low or 'rav4' in low:protected+=1;continue
        p=PROJECT/name
        if not p.is_file():missing+=1;continue
        if sha(p)!=h:raise ValueError('prior file changed: '+name)
        verified+=1
    if strict and missing:raise ValueError(f'incomplete project: {missing} prior files unavailable')
    return verified,missing,protected

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete-project',action='store_true');args=parser.parse_args()
    c.check_lock()
    for name,h in read(ROOT/'sha256.json').items():
        if sha(ROOT/name)!=h:raise ValueError('artifact changed: '+name)
    result=read(ROOT/'results.json');cells=verify_metrics(result);verify_csv(result);equal(result,c.compute())
    import test_synthesis
    tests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_synthesis))
    if not tests.wasSuccessful():raise ValueError('regression failure')
    verified,missing,protected=verify_prior(args.require_complete_project)
    print(f'COMPARISON SYNTHESIS VERIFIED: 27178 paired rows, 6 models, 27 route draws, {cells} independent metric cells; {tests.testsRun} tests.')
    print(f'Prior files: {verified} verified, {missing} unavailable; {protected} protected paths not opened.')
    print('Existing accuracy failures retained; route ranges are not confidence intervals; named published systems not reproduced.')
    print('No new fitting; witness disabled; reserved final closed. COMPARISON_SYNTHESIS_COMPLETE_NO_PROMOTION')
if __name__=='__main__':main()
