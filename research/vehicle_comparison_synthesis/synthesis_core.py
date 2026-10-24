"""Deterministic synthesis of immutable predictions; no model import or fitting."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent;RESEARCH=ROOT.parent
METHODS=('affine_median_matched','ridge_context','huber_context','original_context','uncorrected_radar_imu','signed_axis_corrected')
DELTA_METRICS=('mae_mps','rmse_mps','p95_mps')
PRIMARY='signed_axis_corrected';REFERENCE='original_context'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def align(y,a,fold,other_fold):
    if not np.array_equal(y,a) or not np.array_equal(fold,other_fold): raise ValueError('frozen targets/folds differ')

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y<0) or np.any(p<0): raise ValueError('unaligned/nonfinite/negative speeds')
    d=p-y;a=abs(d)
    return dict(count=len(y),mae_mps=float(a.mean()),rmse_mps=float(np.sqrt(np.mean(d*d))),p95_mps=float(np.quantile(a,.95,method='linear')),maximum_mps=float(a.max()),bias_mps=float(d.mean()),above_half_mps=int((a>.5).sum()),above_2_mps=int((a>2).sum()))

def paired_delta(y,p,reference):
    a=metrics(y,p);b=metrics(y,reference)
    return {k:a[k]-b[k] for k in DELTA_METRICS}

def route_draws(groups):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not len(groups): raise ValueError('invalid route labels')
    labels=sorted(set(groups.tolist()))
    if len(labels)!=3: raise ValueError('exactly three named routes required')
    lookup={g:np.flatnonzero(groups==g) for g in labels}
    return [(list(draw),np.concatenate([lookup[g] for g in draw])) for draw in itertools.product(labels,repeat=3)]

def load():
    members=read(RESEARCH/'vehicle_route_validation/membership.json')
    with np.load(RESEARCH/'vehicle_matched_baselines/oof_predictions.npz',allow_pickle=False) as z: base={k:z[k].copy() for k in z.files}
    with np.load(RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz',allow_pickle=False) as z: axis={k:z[k].copy() for k in z.files}
    with np.load(RESEARCH/'vehicle_route_validation/oof_predictions.npz',allow_pickle=False) as z: route={k:z[k].copy() for k in z.files}
    y=base['target_mps'];fold=base['fold_id'];align(y,axis['target_mps'],fold,axis['fold_id']);align(y,route['target_mps'],fold,route['fold_id'])
    if len(members)!=27178 or y.shape!=(27178,) or fold.shape!=y.shape or fold.dtype.kind not in 'iu': raise ValueError('cohort differs')
    if [m['id'] for m in members]!=list(range(len(members))): raise ValueError('row identity differs')
    seen=set()
    for m in members:
        key=(m['window_id'],m['time_ns'])
        if type(m['time_ns']) is not int or m['time_ns']<0 or key in seen: raise ValueError('duplicate or invalid epoch identity')
        seen.add(key)
    groups=np.array([m['group'] for m in members]);recordings=np.array([m['route'] for m in members]);windows=np.array([m['window_id'] for m in members])
    if sorted(set(groups.tolist()))!=['Urban01','Urban03','Urban04'] or len(set(recordings))!=5 or len(set(windows))!=15: raise ValueError('strata differ')
    for group in sorted(set(groups)):
        if len(set(fold[groups==group].tolist()))!=1: raise ValueError('route is split across folds')
    if len(set(fold.tolist()))!=3: raise ValueError('fold identity differs')
    pred={name:base[name+'__prediction_mps'] for name in METHODS[:3]}
    pred.update({name:axis[name+'__prediction_mps'] for name in METHODS[3:]})
    # Original predictions must be exactly the same, not merely similar fits.
    if not np.array_equal(pred[REFERENCE],base['context_boosting__prediction_mps']) or not np.array_equal(pred[REFERENCE],route['prediction_mps']): raise ValueError('original model predictions disagree')
    for values in pred.values(): metrics(y,values)
    return y,pred,groups,recordings,windows

def compute():
    y,pred,groups,recordings,windows=load()
    def cell(ids): return {name:metrics(y[ids],pred[name][ids]) for name in METHODS}
    result=dict(status='COMPARISON_SYNTHESIS_COMPLETE_NO_PROMOTION',cohort_rows=len(y),methods=list(METHODS),pooled=cell(np.arange(len(y))),
                by_route={g:cell(np.flatnonzero(groups==g)) for g in sorted(set(groups))},
                by_recording={g:cell(np.flatnonzero(recordings==g)) for g in sorted(set(recordings))},
                by_window={g:cell(np.flatnonzero(windows==g)) for g in sorted(set(windows))})
    draws=[]
    for labels,ids in route_draws(groups):
        scores=cell(ids)
        draws.append(dict(routes=labels,rows=len(ids),metrics=scores,delta_vs_original={name:{k:scores[name][k]-scores[REFERENCE][k] for k in DELTA_METRICS} for name in METHODS}))
    result['route_mixtures']=draws
    result['distinct_route_count_mixtures']=len({tuple(labels.count(g) for g in sorted(set(groups))) for labels,_ in route_draws(groups)})
    result['route_mixture_delta_range']={name:{k:dict(minimum=min(d['delta_vs_original'][name][k] for d in draws),maximum=max(d['delta_vs_original'][name][k] for d in draws),negative_count=sum(d['delta_vs_original'][name][k]<0 for d in draws),draws=len(draws)) for k in DELTA_METRICS} for name in METHODS}
    result['failed_windows']={name:[window for window,scores in result['by_window'].items() if scores[name]['p95_mps']>.5] for name in METHODS}
    result['failed_recordings']={name:[recording for recording,scores in result['by_recording'].items() if scores[name]['p95_mps']>.5] for name in METHODS}
    result['claims']=dict(new_fit=False,same_rows=True,equal_information_budget=False,published_systems_reproduced=False,confidence_interval=False,statistical_significance=False,independent_physical_error_bound=False,reserved_final_opened=False,promotion_allowed=False)
    return result

def check_lock():
    protocol=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json')
    if any(protocol[k] for k in ('new_fitting','threshold_changes','reserved_final_opened','witness_enabled','promotion_allowed')): raise ValueError('scope changed')
    if sha(ROOT/'protocol.json')!=lock['protocol_sha256']: raise ValueError('protocol changed')
    for name,h in {**protocol['input_sha256'],**lock['code_sha256']}.items():
        if sha(PROJECT/name)!=h: raise ValueError('frozen source changed: '+name)

def freeze():
    if (ROOT/'analysis_lock.json').exists(): raise ValueError('do not replace the frozen synthesis')
    names=['research/vehicle_comparison_synthesis/synthesis_core.py','research/vehicle_comparison_synthesis/generate_synthesis.py']
    write(ROOT/'analysis_lock.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),code_sha256={n:sha(PROJECT/n) for n in names},retrospective=True,external_preregistration=False))
    check_lock()
