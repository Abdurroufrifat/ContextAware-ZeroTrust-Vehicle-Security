from head_core import *
def score():
    check_models()
    if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('scoring already attempted; no search')
    fit=fitting_data();held={method:np.full(len(fit['y']),np.nan) for method in NEW};seen=np.zeros(len(fit['y']),int)
    for scope in SCOPES[:3]:
        _,_,_,ids,_=training_batch(scope,fit);x=head_features(fit['parent'][ids],fit['axes'][ids]);seen[ids]+=1
        for method in NEW:held[method][ids]=np.maximum(0.,fit['parent'][ids]+predict(read(ROOT/'models'/scope/(method+'.json')),x))
    if not np.all(seen==1):raise ValueError('held rows duplicated or omitted')
    old=fit['prior'];held.update(radar_imu_parent=fit['parent'],nonlinear7=old['nonlinear7__prediction_mps'],linear7=old['linear7__prediction_mps'])
    held_result={method:held_summary(fit,p) for method,p in held.items()};f.write_replay(ROOT/'held_predictions.npz',{'target_mps':fit['y'],'fold_id':fit['fold'],**{method+'__prediction_mps':p for method,p in held.items()}})
    data=f.inputs();scale=np.array(read(RESEARCH/'vehicle_deadband_guard/parameters.json')['scale']);times=data['times'];_,y,_,masks=fb.cohort();(ROOT/'replays').mkdir();cases={}
    for case in d.CASES:
        base=load(fb.paths(case)['none']);guard=load(fb.paths(case)['deadband']);source=d.attack_source(data,case,scale);axes=f.axes(source,data['events'])[data['ready']];parent=base['radar_imu_parent__ready_prediction_mps'];x=head_features(parent,axes)
        candidate=np.maximum(0.,parent+f.infer(data['head'],x));np.testing.assert_allclose(candidate,base['nonlinear7__ready_prediction_mps'],rtol=0,atol=1e-9)
        radar=fb.branch(base,'radar_only');outputs={'radar_only':fb.select(radar,radar,False)};saved={}
        predictions={'linear7':np.maximum(0.,parent+n.ridge_predict(read(RESEARCH/'vehicle_nonlinear_correction/models/all_fit/linear7.json'),x)),**{method:np.maximum(0.,parent+predict(read(ROOT/'models/all_fit'/(method+'.json')),x)) for method in NEW}}
        for method in MODELS:
            if method in ('radar_imu_parent','nonlinear7'):primary=fb.branch(guard,method)
            else:
                p=np.full(len(data['events']),np.nan);p[data['ready']]=predictions[method];keep=data['source']&(base['current__status']==0)&np.isfinite(p)&np.isfinite(data['peer'])&(p>=0)&(data['peer']>=0)&(abs(p-data['peer'])<=1.)
                raw=n.serve(times,data['events'],data['ready'],p,keep,base['binding']);primary=d.g.apply_guard(raw,guard['request_content_fresh']);primary={key:primary[key] for key in fb.FIELDS};saved[method+'__ready_prediction_mps']=predictions[method]
            for strategy in ('withhold','fallback'):
                name=method+'__deadband__'+strategy;out=fb.select(primary,radar,strategy=='fallback');outputs[name]=out
                if method in (*NEW,'linear7'):
                    for key,v in out.items():saved[name+'__'+key]=v
        cases[case]={name:fb.summarize(out,times,y,masks) for name,out in outputs.items()};f.write_replay(ROOT/'replays'/(case+'.npz'),saved)
        print('SCORED',case,{method:cases[case][method+'__deadband__fallback']['during']['accepted_errors']['p95_mps'] for method in ('nonlinear7',*NEW)},flush=True)
    write(ROOT/'results.json',{'decision':'MATCHED_HEAD_COMPARISON_COMPLETE_NO_PROMOTION','held':held_result,'cases':cases,'vehicle_original_models_changed':False,'labels_changed':False,'threshold_search':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False})
    print('MATCHED_HEAD_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)
if __name__=='__main__':score()
