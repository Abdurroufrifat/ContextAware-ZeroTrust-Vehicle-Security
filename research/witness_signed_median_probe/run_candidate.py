"""Reproduce one exploratory comparison; never enables or changes a policy."""
from pathlib import Path
import csv,json
from base_loader import BASE,baseline,baseline_analysis
import candidate_core
HERE=Path(__file__).resolve().parent

def read_stream(name):
    with (BASE/name).open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for key in ('bag_time_ns','header_time_ns','sequence'):r[key]=int(r[key])
        r['speed_raw']=float(r['speed_raw'])
    return rows


def compute():
    radar=read_stream('radar_raw.csv');obd=read_stream('obd_raw.csv')
    windows=json.loads((BASE/'window_summary.json').read_text());previous=json.loads((BASE/'result.json').read_text())
    pairs=[];windowstats={}
    for w in windows:
        r=[v for v in radar if v['window_id']==w['id']];o=[v for v in obd if v['window_id']==w['id']]
        rows,stats=candidate_core.paired_records(r,o,w['start_ns']);windowstats[w['id']]=stats
        for row in rows:pairs.append({'window_id':w['id'],'role':w['role'],**row})
    oldpairs=baseline_analysis.load_pairs()
    if [{k:v for k,v in r.items() if k!='feature_raw'} for r in pairs]!=[{k:v for k,v in r.items() if k!='feature_raw'} for r in oldpairs]:
        raise ValueError('matched endpoint set changed')
    model=baseline.fit_affine([r for r in pairs if r['role']=='fit']);roles={};bywindow=[]
    for role,oldmetrics in previous['roles'].items():
        selected=[r for r in pairs if r['role']==role];stats=baseline_analysis.describe(selected,model)
        den=sum(w['eligible_radar'] for w in windows if w['role']==role);coverage=len(selected)/den if den else 0.0
        stats.update({'eligible_radar':den,'pair_coverage':coverage,'provisional_quality_target_met':bool(model and coverage>=.95 and stats.get('p95_abs_disagreement_mps',float('inf'))<=1)})
        roles[role]={'baseline':oldmetrics,'candidate':stats}
    for w in windows:
        stats=baseline_analysis.describe([r for r in pairs if r['window_id']==w['id']],model)
        bywindow.append({'id':w['id'],'role':w['role'],'baseline_p95_mps':next(v['p95_abs_disagreement_mps'] for v in previous['windows'] if v['id']==w['id']),**stats,**windowstats[w['id']]})
    output=[]
    old_by_key={(r['window_id'],r['time_ns']):r for r in oldpairs}
    for row in pairs:
        old=old_by_key[row['window_id'],row['time_ns']]
        output.append({**row,'baseline_feature_raw':old['feature_raw'],
            'baseline_prediction_mps':baseline.predict(previous['model'],old['feature_raw']),
            'candidate_prediction_mps':baseline.predict(model,row['feature_raw']) if model else None})
    result={'scope':'Exploratory comparison on already inspected development windows; not an independent validation',
      'candidate':'magnitude after signed median','baseline_model':previous['model'],'candidate_model':model,
      'roles':roles,'windows':bywindow,'matched_records':len(pairs),'same_endpoints_and_labels':True,
      'candidate_provisional_target_met':all(r['candidate']['provisional_quality_target_met'] for r in roles.values()),
      'independent_validation':False,'witness_enabled':False,'production_policy_changed':False,
      'quality_evidence':{'raw_radar_records':1629,'constant_object_count':1,'constant_range':0,'all_four_signal_arrays_empty':True},
      'remaining_limits':['No independent test: all nine windows were previously examined','Raw radar unit setting remains unknown','Benign OBD comparison labels are not independent ground truth','Low-speed/standstill-heavy samples; later windows do not cover higher speeds','Bag arrival timestamps do not verify live causal timing or clock security','No cryptographic witness channel or startup/recovery attack evaluation in this step']}
    return result,output

if __name__=='__main__':
    result,rows=compute();(HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (HERE/'comparison.csv').open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps({k:result[k] for k in ('candidate_model','roles','matched_records','candidate_provisional_target_met','witness_enabled')},indent=2))
