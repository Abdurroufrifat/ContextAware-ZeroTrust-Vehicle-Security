"""Separate exact joins, scalar admission/metrics and historical parity."""
import io,math,sys,unittest,zipfile
import numpy as np
from head_reference_core import ROOT,RESEARCH,PROJECT,METHODS,read,load,digest,freeze
import verify_reference as old
close=old.close

def scalar_cell(y,pred,obd,ids):
    value=old.scalar_cell(y,pred,obd,ids)
    for model in value['models'].values():
        a=model['primary_reference']['p95_mps'];b=model['obd']['p95_mps'];model['both_reference_p95_le_0_5']=None if a is None or b is None else a<=.5 and b<=.5
    return value
def prior_cell(value,methods):
    return {'rows':value['rows'],'models':{method:{key:v for key,v in value['models'][method].items() if key!='both_reference_p95_le_0_5'} for method in methods},'obd_minus_primary':value['obd_minus_primary']}

def verify():
    freeze();result=read(ROOT/'results.json');previous=read(RESEARCH/'vehicle_nonlinear_reference/results.json');records=read(RESEARCH/'vehicle_reference_check/matched_endpoints.json');diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz');pairs=load(RESEARCH/'vehicle_frame_reference_audit/reference_pairs.npz');identity=load(ROOT/'row_identity.npz');prior_identity=load(RESEARCH/'vehicle_nonlinear_reference/row_identity.npz');members=read(RESEARCH/'vehicle_route_validation/membership.json');old_oof=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz');held=load(RESEARCH/'vehicle_matched_head_comparison/held_predictions.npz');nominal=load(RESEARCH/'vehicle_matched_head_comparison/replays/nominal.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/development_replay.npz');signed=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');parent=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');base=load(RESEARCH/'vehicle_sensor_faults/replays/nominal.npz');guard=load(RESEARCH/'vehicle_deadband_guard/replays/nominal__deadband.npz');frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');night=load(RESEARCH/'vehicle_night_training/predictions.npz');source=load(RESEARCH/'vehicle_night_training/decisions.npz')['day_plus_night__quality'];peer=night['day_plus_night__masked_cnn_4ch'];lookup={}
    for j,row in enumerate(records):
        if type(row['time_ns']) is not int or row['time_ns']<0:raise AssertionError('noninteger source clock')
        key=(row['window_id'],row['time_ns'])
        if key in lookup:raise AssertionError('duplicate exact source key')
        lookup[key]=j
    if set(result['populations'])!={'fit','full'}:raise AssertionError('scope omitted')
    metric_cells=historical_cells=matches=0
    for scope in ('fit','full'):
        q={key.split('__',1)[1]:v for key,v in diag.items() if key.startswith(scope+'__')};n=len(q['row_id']);ids=[];present=[];obd=[]
        if q['time_ns'].dtype.kind not in 'iu':raise AssertionError('endpoint time type')
        for i,(window,t) in enumerate(zip(q['window_id'],q['time_ns'])):
            j=lookup.get((str(window),int(t)),-1);ids.append(j);present.append(j>=0)
            if j<0:obd.append(math.nan);continue
            row=records[j]
            if row['reference_horizontal_mps']!=float(q['target_mps'][i]) or int(t)-row['reference_header_time_ns']!=int(q['reference_age_ns'][i]):raise AssertionError('primary target or age changed')
            value=float(row['label_mps'])
            if not math.isfinite(value) or value<0:raise AssertionError('invalid OBD')
            obd.append(value)
        matches+=sum(present);np.testing.assert_array_equal(ids,pairs[scope+'__reference_match_id']);np.testing.assert_array_equal(present,pairs[scope+'__comparison_present']);np.testing.assert_array_equal(q['row_id'],pairs[scope+'__row_id']);np.testing.assert_allclose(obd,pairs[scope+'__obd_mps'],rtol=0,atol=0,equal_nan=True)
        if scope=='fit':
            if n!=27178 or len(members)!=n:raise AssertionError('held cohort')
            np.testing.assert_array_equal(q['row_id'],np.arange(n));np.testing.assert_array_equal(q['target_mps'],held['target_mps']);np.testing.assert_array_equal(held['fold_id'],old_oof['fold_id']);np.testing.assert_array_equal(q['time_ns'],[v['time_ns'] for v in members]);np.testing.assert_array_equal(q['window_id'],[v['window_id'] for v in members])
            pred={method:old_oof[method+'__prediction_mps'] for method in old.METHODS}
            pred.update({method:held[method+'__prediction_mps'] for method in ('huber7','extra_trees7')});accepted=[True]*n;own={method:[True]*n for method in METHODS}
            for method,alias in [('parent_radar_imu','radar_imu_parent'),('linear7','linear7'),('nonlinear7','nonlinear7')]:np.testing.assert_array_equal(pred[method],held[alias+'__prediction_mps'])
        else:
            if n!=40678:raise AssertionError('full cohort')
            np.testing.assert_array_equal(q['row_id'],np.flatnonzero([v['ready'] for v in frames]));np.testing.assert_array_equal(q['time_ns'],[frames[i]['time_ns'] for i in q['row_id']])
            pred={'original_context':signed['original_context__ready_prediction_mps'],'parent_radar_imu':parent['radar_imu__ready_prediction_mps'],'signed6_previous':signed['signed_axis_corrected__ready_prediction_mps'],**{method:new[method+'__ready_prediction_mps'] for method in ('linear7','nonlinear7')},**{method:nominal[method+'__ready_prediction_mps'] for method in ('huber7','extra_trees7')}}
            np.testing.assert_allclose(pred['linear7'],nominal['linear7__ready_prediction_mps'],rtol=0,atol=1e-9)
            accepted=[bool(new['nonlinear7__quality'][raw]) for raw in q['row_id']];own={method:[] for method in METHODS}
            frozen={'original_context':signed['original_context__quality'],'parent_radar_imu':parent['radar_imu__quality'],'signed6_previous':signed['signed_axis_corrected__quality'],'linear7':new['linear7__quality'],'nonlinear7':new['nonlinear7__quality']}
            for i,raw in enumerate(q['row_id']):
                fresh=bool(guard['history_clean'][raw]) and bool(guard['endpoint_content_fresh'][raw])
                for method in old.METHODS:own[method].append(bool(frozen[method][raw]) and (method=='original_context' or fresh))
                for method in ('linear7','nonlinear7','huber7','extra_trees7'):
                    p=float(pred[method][i]);v=float(peer[i]);ok=bool(source[raw]) and int(base['current__status'][raw])==0 and fresh and math.isfinite(p) and math.isfinite(v) and p>=0 and v>=0 and abs(p-v)<=1.
                    if method in old.METHODS:
                        if ok!=own[method][-1]:raise AssertionError('historical eligibility parity')
                    else:own[method].append(ok)
        if any(p.shape!=(n,) or not np.isfinite(p).all() or np.any(p<0) for p in pred.values()):raise AssertionError('prediction shape/values')
        for key,v in [('row_id',q['row_id']),('time_ns',q['time_ns']),('match_id',ids),('present',present),('accepted',accepted)]:
            np.testing.assert_array_equal(identity[scope+'__'+key],v);np.testing.assert_array_equal(prior_identity[scope+'__'+key],v)
        for method in METHODS:np.testing.assert_array_equal(identity[scope+'__own_accepted__'+method],own[method])
        categories={'pooled':{'all':list(range(n))},'by_window':{name:[i for i,v in enumerate(q['window_id']) if v==name] for name in sorted(set(q['window_id']))}}
        if scope=='fit':
            for category,key in [('by_group','group'),('by_recording','route')]:categories[category]={value:[i for i,v in enumerate(members) if v[key]==value] for value in sorted({v[key] for v in members})}
        if set(categories)!=set(result['populations'][scope]):raise AssertionError('category omitted')
        for category,pops in categories.items():
            if set(pops)!=set(result['populations'][scope][category]):raise AssertionError('population omitted')
            for name,rows in pops.items():
                selected=[i for i in rows if present[i]];all_cell=scalar_cell(q['target_mps'],pred,obd,selected);v={'parent_rows':len(rows),'matched_rows':len(selected),'missing_rows':len(rows)-len(selected),'all_matched':all_cell};metric_cells+=len(METHODS)*2+1
                oldpop=previous['populations'][scope][category][name]
                for key in ('parent_rows','matched_rows','missing_rows'):close(v[key],oldpop[key])
                close(prior_cell(all_cell,old.METHODS),oldpop['all_matched']);historical_cells+=len(old.METHODS)*2+1
                if scope=='full':
                    common=[i for i in selected if accepted[i]];v['fixed_hgb_accepted_matched']=scalar_cell(q['target_mps'],pred,obd,common);metric_cells+=len(METHODS)*2+1;close(prior_cell(v['fixed_hgb_accepted_matched'],old.METHODS),oldpop['nonlinear_accepted_matched']);historical_cells+=len(old.METHODS)*2+1;own_result={}
                    for method in METHODS:
                        admitted=[i for i in rows if own[method][i]];match=[i for i in admitted if present[i]]
                        own_result[method]={'admitted_parent_rows':len(admitted),'admitted_matched_rows':len(match),'admitted_missing_rows':len(admitted)-len(match),'comparison':scalar_cell(q['target_mps'],{method:pred[method]},obd,match)};metric_cells+=3
                    v['own_admitted_matched']=own_result
                close(v,result['populations'][scope][category][name])
    if result['decision']!='FROZEN_HEAD_REFERENCE_COMPARISON_COMPLETE_NO_PROMOTION':raise AssertionError('decision changed')
    for flag in ('models_refitted','labels_changed','thresholds_changed','reserved_final_opened','production_policy_changed','promotion_allowed','reference_threat_independence_established'):
        if result[flag] is not False:raise AssertionError('unsupported claim')
    for name,h in read(ROOT/'sha256.json').items():
        if digest(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
    prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
    for name,h in prior.items():
        if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest changed '+name)
    for name,h in manifest.items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):protected+=1;continue
        p=PROJECT/name
        if not p.is_file():missing+=1;continue
        if digest(p)!=h:raise AssertionError('project hash '+name)
        verified+=1
    import test_head_reference
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_head_reference))
    if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
    print('HEAD REFERENCE VERIFICATION PASS:',matches,'exact matched rows;',metric_cells,'scalar metric cells;',historical_cells,'historical parity cells;',tests.testsRun,'tests')
    print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
    print('FROZEN_HEAD_REFERENCE_COMPARISON_COMPLETE_NO_PROMOTION')
if __name__=='__main__':verify()
