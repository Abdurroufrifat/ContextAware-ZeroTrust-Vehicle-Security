"""Reproduce provisional calibration metrics from development-only matched records."""
from pathlib import Path
import csv,json,math,statistics
import probe_core as c
HERE=Path(__file__).resolve().parent

def load_pairs():
    rows=list(csv.DictReader((HERE/'causal_pairs.csv').open(newline='')))
    for r in rows:
        for k in ('time_ns','obd_header_age_ns','radar_delivery_ns','radar_history_samples'):r[k]=int(r[k])
        for k in ('feature_raw','label_mps'):r[k]=float(r[k])
    return rows

def quantile(values,p):
    a=sorted(values)
    if not a:return None
    i=(len(a)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return a[lo]+(a[hi]-a[lo])*(i-lo)

def describe(rows,model):
    ys=[r['label_mps'] for r in rows]
    result={'pairs':len(rows),'label_min_mps':min(ys) if ys else None,'label_max_mps':max(ys) if ys else None,
            'moving_label_pairs_ge_1mps':sum(y>=1 for y in ys)}
    if model and rows:
        errors=[c.predict(model,r['feature_raw'])-r['label_mps'] for r in rows]
        result.update({'mae_mps':statistics.mean(abs(e) for e in errors),'rmse_mps':math.sqrt(statistics.mean(e*e for e in errors)),
                       'p95_abs_disagreement_mps':quantile([abs(e) for e in errors],.95),
                       'maximum_abs_disagreement_mps':max(abs(e) for e in errors),
                       'fraction_abs_disagreement_gt_1mps':sum(abs(e)>1 for e in errors)/len(errors)})
    return result

def compute():
    rows=load_pairs();windows=json.loads((HERE/'window_summary.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
    fit=[r for r in rows if r['role']=='fit'];model=c.fit_affine(fit)
    roles={}
    for role in protocol['window_fractions']:
        group=[r for r in rows if r['role']==role];stats=describe(group,model)
        den=sum(w['eligible_radar'] for w in windows if w['role']==role)
        stats.update({'eligible_radar':den,'pair_coverage':len(group)/den if den else 0.0})
        stats['provisional_quality_target_met']=bool(model and stats['pair_coverage']>=.95 and stats.get('p95_abs_disagreement_mps',math.inf)<=1)
        roles[role]=stats
    bywindow=[]
    for w in windows:bywindow.append({**w,**describe([r for r in rows if r['window_id']==w['id']],model)})
    statuses=['units_not_verified','single_development_drive','OBD_is_benign_comparison_label_not_ground_truth','live_availability_and_authentication_unverified']
    if model is None:statuses.append('insufficient_or_nonphysical_fit')
    if any(not roles[k]['provisional_quality_target_met'] for k in ('calibration','diagnostic')):statuses.append('provisional_quality_target_failed')
    return {'scope':protocol['scope'],'model':model,'model_units':'output m/s per recorded raw radar unit; NOT a verified mounting-angle calibration',
            'roles':roles,'windows':bywindow,'overall_provisional_quality_met':all(roles[k]['provisional_quality_target_met'] for k in roles),
            'witness_enabled':False,'attack_evaluation_performed':False,'production_policy_changed':False,'unresolved':statuses}

if __name__=='__main__':
    result=compute();(HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('model','roles','overall_provisional_quality_met','witness_enabled','unresolved')},indent=2))
