import platform,time,warnings
from head_core import *
def train():
    protocol=check_lock()
    if (ROOT/'models').exists() or (ROOT/'model_lock.json').exists():raise ValueError('fit already attempted; do not tune/retry')
    import sklearn
    from sklearn.linear_model import HuberRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    if sklearn.__version__!='1.8.0':raise ValueError('fitting requires pinned sklearn 1.8.0')
    data=fitting_data();(ROOT/'models').mkdir();runs=[];start=time.monotonic()
    for scope in SCOPES:
        x,y,w,held,hashes=training_batch(scope,data)
        for method in NEW:
            fit_start=time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if method=='huber7':
                    mean,scale=scaler(x,w);z=(x-mean)/scale;model=HuberRegressor(**protocol['huber']);model.fit(z,y,sample_weight=w);native=model.predict(z)
                    artifact={'method':method,'mean':mean.tolist(),'scale':scale.tolist(),'coefficient':model.coef_.tolist(),'intercept':float(model.intercept_),'robust_residual_scale':float(model.scale_),'n_iter':int(model.n_iter_),'converged_before_iteration_cap':int(model.n_iter_)<protocol['huber']['max_iter']}
                else:
                    model=ExtraTreesRegressor(**protocol['extra_trees']);model.fit(x,y,sample_weight=w);native=model.predict(x);artifact=export_extra(model)
            portable=predict(artifact,x);delta=float(np.max(abs(native-portable)))
            if delta>1e-8:raise ValueError('native/portable training mismatch')
            artifact.update(scope=scope,feature_names=n.FEATURES,output='clip(parent speed + residual, minimum=0)')
            write(ROOT/'models'/scope/(method+'.json'),artifact);f.write_replay(ROOT/'models'/scope/(method+'_training.npz'),{'native':native,'portable':portable})
            runs.append({'scope':scope,'method':method,'augmented_rows':len(x),'held_rows':len(held),**hashes,'native_export_max_difference_mps':delta,'warnings':[str(v.message) for v in caught],'seconds':time.monotonic()-fit_start,'n_iter':int(model.n_iter_) if method=='huber7' else None})
            print('FROZEN',scope,method,'rows',len(x),'export difference',delta,'warnings',len(caught),flush=True)
    write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'seconds':time.monotonic()-start,'scope':'Same in-sample-parent residual training as frozen HGB; not independent calibration'})
    files={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_eight_locked_before_scoring':True})
    print('ALL EIGHT COMPARATOR MODELS LOCKED BEFORE SCORING',flush=True)
if __name__=='__main__':train()
