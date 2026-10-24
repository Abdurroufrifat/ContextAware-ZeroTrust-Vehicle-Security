"""Independent saved arithmetic and GRU forward replay; no fitting or validation access.

Does not import the Phase2 runner, its model, feature, calibration or metric functions.
The existing base-policy implementation is replayed, not independently reimplemented.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected,tolerance=1e-8):
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=tolerance,equal_nan=True)


def independent_features(x,t,s):
    values=np.full((len(x),6),np.nan);values[:,:3]=x
    for i in range(5,len(x)):
        if any(s[k]!=t[k-1] for k in range(i-4,i+1)):continue
        total=sum(float(x[k,2])*(int(t[k])-int(s[k]))/1000 for k in range(i-4,i+1))
        duration=(int(t[i])-int(t[i-5]))/1000
        values[i,3:]=[x[i,0]-x[i,1],(x[i,0]-x[i-5,0]-total)/duration,(x[i,1]-x[i-5,1]-total)/duration]
    p=np.array([i for i in range(31,len(x)) if np.isfinite(values[i-31:i+1]).all()
                and all(s[k]==t[k-1] for k in range(i-30,i+1))],dtype=np.int64)
    return values,p


def independent_metrics(c,b):
    t,s=c['time_ms'],c['start_ms'];p=c['positions'];y=c['labels'];base=c['base_grant']
    score=b['score'];threshold=float(b['threshold']);n=len(t)
    assert t.shape==s.shape==(n,) and np.all(t>s) and np.all(np.diff(t)>0) and np.all(s[1:]>=t[:-1])
    assert score.shape==p.shape and np.array_equal(b['positions'],p) and np.all(np.diff(p)>0)
    known=np.isfinite(score);alarm=known&(score>threshold)
    expected=np.zeros(n,dtype=bool);expected[p]=base[p]&known&~alarm
    effective=base.copy();effective[p]&=known
    for name,value in [('known',known),('alarm',alarm),('grant',expected),('effective_base',effective)]:
        assert np.array_equal(b[name],value), 'changed decision: '+name
    assert np.isfinite(threshold)
    full_alarm=np.zeros(n,dtype=bool);full_alarm[p]=alarm
    scored=np.zeros(n,dtype=bool);scored[p]=True
    duration=t-s
    def groups(mask):
        idx=np.flatnonzero(mask)
        if not len(idx):return []
        breaks=np.flatnonzero((np.diff(idx)!=1)|(s[idx[1:]]!=t[idx[:-1]]))+1
        return list(np.split(idx,breaks))
    normal=~y;grants=y&expected
    normal_bursts=groups(normal&full_alarm);normal_denials=groups(normal&~expected);grant_runs=groups(grants)
    events=[]
    for idx in groups(y):
        begin,end=int(idx[0]),int(idx[-1]);hits=idx[full_alarm[idx]];denials=idx[~expected[idx]]
        events.append(dict(first_epoch=begin,last_epoch=end,start_ms=int(s[begin]),end_ms=int(t[end]),
            epochs=len(idx),scored_epochs=int(scored[idx].sum()),
            first_neural_alarm_delay_ms=int(t[hits[0]]-s[begin]) if len(hits) else None,
            first_denial_delay_ms=int(t[denials[0]]-s[begin]) if len(denials) else None,
            granted_epochs=int(expected[idx].sum()),
            granted_epochs_after_first_denial=int(expected[idx[idx>denials[0]]].sum()) if len(denials) else 0,
            fully_denied=not expected[idx].any()))
    active=y[p]
    return dict(epochs=n,scored_epochs=len(p),unscored_epochs=n-len(p),normal_epochs=int(normal.sum()),
        normal_scored_epochs=int((normal&scored).sum()),normal_granted_epochs=int((normal&expected).sum()),
        normal_denied_scored_epochs=int((normal&scored&~expected).sum()),normal_neural_alarm_bursts=len(normal_bursts),
        normal_neural_longest_alarm_run_epochs=max(map(len,normal_bursts),default=0),
        normal_longest_denial_run_epochs=max(map(len,normal_denials),default=0),
        normal_granted_observed_span_ms=int(duration[normal&expected].sum()),attack_epochs=int(y.sum()),attack_events=len(events),
        neural_missed_events=sum(e['first_neural_alarm_delay_ms'] is None for e in events),
        fully_denied_events=sum(e['fully_denied'] for e in events),
        events_with_grants_after_first_denial=sum(e['granted_epochs_after_first_denial']>0 for e in events),
        attack_granted_epochs=int(grants.sum()),attack_granted_observed_span_ms=int(duration[grants].sum()),
        attack_longest_grant_run_epochs=max(map(len,grant_runs),default=0),
        attack_longest_grant_run_observed_span_ms=max((int(duration[idx].sum()) for idx in grant_runs),default=0),events=events,
        scored_attack_epochs=int(active.sum()),attack_alarm_epochs=int(alarm[active].sum()),
        active_endpoint_tpr=float(alarm[active].mean()) if active.any() else None,
        normal_alarm_epochs=int(alarm[~active].sum()),normal_eligible_epochs=int((normal&scored).sum()),
        normal_eligible_denied=int((normal&scored&~expected).sum()),normal_total_denied=int((normal&~expected).sum()),
        unknown_scored_epochs=int((~known).sum()))


def verify_score_contract(block,method,seed,lock,reference_profiles):
    if method=='original_nn_frozen':
        spec=reference_profiles[str(seed)]
        assert float(block['threshold'])==spec['threshold'],'original threshold changed'
        profile=spec['profile']
        expected=((block['residual']-np.asarray(profile['median']))/np.asarray(profile['scale'])).max(1)
        np.testing.assert_allclose(block['score'],expected,rtol=0,atol=1e-10)
    else:
        assert float(block['threshold'])==lock['thresholds'][f'{method}__{seed}'],'frozen evaluation threshold changed'


def verify_physics_reference(context,report):
    for scope in ('native','matched'):
        c=dict(context)
        if scope=='native':c['positions']=np.arange(len(c['time_ms']),dtype=np.int64)
        p=c['positions'];score=(~c['base_grant'][p]).astype(float)
        alarm=score>.5;grant=np.zeros(len(c['time_ms']),dtype=bool);grant[p]=c['base_grant'][p]
        b=dict(score=score,threshold=np.array(.5),positions=p,known=np.ones(len(p),dtype=bool),alarm=alarm,grant=grant,effective_base=c['base_grant'])
        metric=independent_metrics(c,b)
        def rename(v):
            if isinstance(v,list):return [rename(x) for x in v]
            if not isinstance(v,dict):return v
            return {('active_policy_denial_fraction' if k=='active_endpoint_tpr' else k.replace('neural','policy').replace('alarm','denial')):rename(x) for k,x in v.items()}
        assert rename(metric)==report[scope]


def load_gru(path,d):
    state=torch.load(path,map_location='cpu',weights_only=True)
    gru=nn.GRU(d,32,batch_first=True);head=nn.Linear(32,d)
    gru.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('gru.')})
    head.load_state_dict({k[5:]:v for k,v in state.items() if k.startswith('head.')})
    gru.eval();head.eval()
    return gru,head


def neural_values(model,joint,p,cols,scaler,profile):
    selected=joint[:,cols];mean=np.asarray(scaler['mean'])[cols];scale=np.asarray(scaler['scale'])[cols]
    history=np.stack([selected[i-31:i] for i in p])
    x=torch.tensor(((history-mean)/scale).astype(np.float32));target=((selected[p]-mean)/scale).astype(np.float32)
    gru,head=model
    with torch.no_grad():
        predictions=[]
        for chunk in x.split(128):
            _,h=gru(chunk);predictions.append(head(h[-1]).numpy())
    prediction=np.concatenate(predictions)
    residual=np.abs(target-prediction)
    contribution=np.maximum(0,(residual-np.asarray(profile['median'],dtype=np.float32))/np.asarray(profile['scale'],dtype=np.float32))
    return prediction,residual,contribution,contribution.max(axis=1)


def audit(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();torch.set_num_threads(2)
    manifest=json.loads((output/'file_hashes.json').read_text());verified=0
    for relative,digest in manifest.items():
        path=(output/relative).resolve();assert path.is_relative_to(output) and sha(path)==digest;verified+=1
    summary=json.loads((output/'summary.json').read_text());lock=json.loads((output/'model_lock.json').read_text())
    protocol=json.loads((root/'research/phase1_method_decision/PHASE2_PROTOCOL.json').read_text())
    assert sha(root/'research/phase1_method_decision/PHASE2_PROTOCOL.json')==lock['phase1_protocol_sha256']
    assert sha(output/'model_lock.json')==summary['model_lock_sha256']
    for name,digest in lock['fit_calibration_sha256'].items():assert sha(output/name)==digest
    for name,digest in lock['code_sha256'].items():assert sha(root/name)==digest,'changed training source: '+name
    index32=root/'artifacts/vehicle_transfer_stage32/file_hashes.json'
    assert sha(index32)==lock['stage32_output_index_sha256']
    for name,digest in json.loads(index32.read_text()).items():assert sha(index32.parent/name)==digest
    assert sha(root/'data/reference/vehicle_transfer_stage25/model_contract.json')==protocol['model_contract_sha256']
    reference_profiles=json.loads((index32.parent/'profiles.json').read_text())['original']
    assert lock['role_order']==['fit','calibrate','lock_written','development_evaluate']
    expected_eval={(r['trip'],r['segment']) for r in protocol['development_segments'] if r['role']=='development_evaluate'}
    assert len(expected_eval)==4 and summary['complete_coverage'] and not summary['errors']
    for forbidden in protocol['locked_rav4_metadata_only']:
        stage=24 if forbidden['trip'].startswith('rav24') else 31
        assert not (root/f'data/raw/vehicle_data_stage{stage}'/forbidden['trip']).exists()
    columns={'joint_history_gru6':[0,1,2,3,4,5],'raw_history_gru3':[0,1,2],'innovation_history_gru3':[3,4,5]}
    networks={(m,s):load_gru(output/'fit'/f'{m}__{s}.pt',len(cols)) for m,cols in columns.items() for s in (7,19,43)}
    # Independently recover the unique-endpoint, equal-route feature scaler.
    fit_values=[];fit_routes=[];fit_context={}
    for row in protocol['development_segments']:
        if row['role']!='fit':continue
        c=read_npz(root/'artifacts/vehicle_transfer_stage32/contexts'/f"{row['trip']}__{row['segment']}.npz")
        joint,p=independent_features(c['features'],c['time_ms'],c['start_ms'])
        endpoints=sorted({k for i in p for k in range(int(i)-31,int(i)+1)})
        fit_values.extend(joint[endpoints]);fit_routes.extend([row['trip']]*len(endpoints));fit_context[row['trip'],row['segment']]=(joint,p)
    fv=np.asarray(fit_values);fr=np.asarray(fit_routes)
    mean=np.mean([fv[fr==r].mean(0) for r in set(fr)],0)
    variance=np.mean([((fv[fr==r]-mean)**2).mean(0) for r in set(fr)],0)
    close(lock['scaler']['mean'],mean);close(lock['scaler']['scale'],np.maximum(np.sqrt(variance),.01))
    max_prediction=0.;max_feature=0.
    fit_residuals=defaultdict(list);residual_routes=[]
    for (trip,segment),(joint,p) in fit_context.items():
        residual_routes.extend([trip]*len(p))
        for (method,seed),network in networks.items():
            identifier=f'{method}__{seed}'
            _,residual,_,_=neural_values(network,joint,p,columns[method],lock['scaler'],lock['profiles'][identifier])
            fit_residuals[identifier].append(residual)
    rr=np.asarray(residual_routes)
    weights=np.array([1/((rr==r).sum()*len(set(rr))) for r in rr])
    def quantile(x,q):
        order=np.argsort(x,kind='stable');cdf=np.cumsum(weights[order]);return x[order[min(np.searchsorted(cdf,q*cdf[-1],side='left'),len(x)-1)]]
    for identifier,values in fit_residuals.items():
        residual=np.concatenate(values)
        med=np.array([quantile(residual[:,j],.5) for j in range(residual.shape[1])])
        spread=np.array([max(.01,quantile(residual[:,j],.75)-quantile(residual[:,j],.25)) for j in range(residual.shape[1])])
        close(lock['profiles'][identifier]['median'],med,1e-6);close(lock['profiles'][identifier]['scale'],spread,1e-6)
    # Calibration forward replay and worst-route higher-quantile threshold.
    cal_scores=defaultdict(lambda:defaultdict(list))
    for row in protocol['development_segments']:
        if row['role']!='calibrate':continue
        key=f"{row['trip']}__{row['segment']}"
        c=read_npz(root/'artifacts/vehicle_transfer_stage32/contexts'/f'{key}.npz');joint,p=independent_features(c['features'],c['time_ms'],c['start_ms'])
        for seed in (7,19,43):
            branches={}
            for method,cols in columns.items():
                identifier=f'{method}__{seed}';pred,residual,contrib,score=neural_values(networks[method,seed],joint,p,cols,lock['scaler'],lock['profiles'][identifier]);branches[method]=contrib
                b=read_npz(output/'calibration'/f'{key}__{identifier}.npz')
                close(b['prediction'],pred,1e-6);close(b['residual'],residual,1e-6);close(b['score'],score,1e-4)
                cal_scores[identifier][row['trip']].extend(score.tolist())
            identifier=f'separate_gru_or__{seed}';score=np.column_stack([branches[m].max(1) for m in ('raw_history_gru3','innovation_history_gru3')]).max(1)
            b=read_npz(output/'calibration'/f'{key}__{identifier}.npz');close(b['score'],score,1e-4)
            cal_scores[identifier][row['trip']].extend(score.tolist())
    for identifier,values in cal_scores.items():
        threshold=max(float(np.quantile(v,.95,method='higher')) for v in values.values());close(lock['thresholds'][identifier],threshold,1e-4)
    # Replay base policy once per stream; independently replay each small GRU.
    from run_vehicle_ai_stage13 import base_grants
    from vehicle_native_data import NativeFrame
    contract=json.loads((root/'data/reference/vehicle_transfer_stage25/model_contract.json').read_text());physical=contract['model']
    grouped=defaultdict(list)
    for row in summary['rows']:grouped[row['context']].append(row)
    assert len(grouped)==60
    records=[]
    for context_name,rows in grouped.items():
        c=read_npz(output/'contexts'/context_name);joint,p=independent_features(c['features'],c['time_ms'],c['start_ms'])
        assert np.array_equal(c['positions'],p)
        first=rows[0];assert (first['trip'],first['segment']) in expected_eval
        normal=read_npz(root/'artifacts/vehicle_transfer_stage32/contexts'/f"{first['trip']}__{first['segment']}.npz")
        labels=np.max(abs(c['features']-normal['features']),1)>1e-9
        assert np.array_equal(labels,c['labels'])
        frames=[NativeFrame(first['trip'],i,int(t),int(c['start_ms'][i]),*(int(v) for v in c['source_time_ms'][i]),
            float(c['features'][i,0]+physical['gnss_offset']),float(c['features'][i,1]),float(c['features'][i,2]+physical['imu_offset']),int(c['maximum_gap_ms'][i])) for i,t in enumerate(c['time_ms'])]
        base,issued,quorum=base_grants(frames,contract)
        assert np.array_equal(base,c['base_grant']) and np.array_equal(issued,c['issued']) and np.array_equal(quorum,c['quorum'])
        calculated={}
        for seed in (7,19,43):
            for method,cols in columns.items():
                identifier=f'{method}__{seed}';calculated[method,seed]=neural_values(networks[method,seed],joint,p,cols,lock['scaler'],lock['profiles'][identifier])
        for row in rows:
            b=read_npz(output/'blocks'/row['block']);method,seed=row['method'],row['seed']
            verify_score_contract(b,method,seed,lock,reference_profiles)
            if method in columns:
                pred,residual,contribution,score=calculated[method,seed]
                max_prediction=max(max_prediction,float(np.max(abs(pred-b['prediction']))))
                close(b['prediction'],pred,1e-6);close(b['residual'],residual,1e-6);close(b['contributions'],contribution,1e-4);close(b['score'],score,1e-4)
                assert np.array_equal(score>float(b['threshold']),b['alarm'])
            elif method=='separate_gru_or':
                score=np.column_stack([calculated[m,seed][3] for m in ('raw_history_gru3','innovation_history_gru3')]).max(1)
                close(b['score'],score,1e-4);assert np.array_equal(score>float(b['threshold']),b['alarm'])
            metric=independent_metrics(c,b)
            # All scalar metrics and event identities are reconstructed, not copied.
            for field,value in metric.items():
                if field=='events':assert value==row['metrics'][field]
                elif value is None:assert row['metrics'][field] is None
                elif isinstance(value,(float,np.floating)):close(row['metrics'][field],value)
                else:assert row['metrics'][field]==value,(row['block'],field)
            records.append(dict(row,metrics=metric))
    assert len(records)==summary['rows_observed']==summary['rows_expected']==900
    physics=json.loads((output/'physics_reference.json').read_text())['rows']
    assert len(physics)==summary['physics_reference_rows']==60
    for row in physics:
        c=read_npz(output/'contexts'/row['context'])
        verify_physics_reference(c,row)
    # Re-evaluate all gate predicates directly from reconstructed route counts.
    all_normal=True
    for seed in (7,19,43):
        data={}
        for method in ('joint_history_gru6','raw_history_gru3'):
            routes={}
            for trip in protocol['development_roles']['development_evaluate']:
                selected=[r['metrics'] for r in records if r['seed']==seed and r['method']==method and r['case']=='normal' and r['trip']==trip]
                n=sum(v['normal_eligible_epochs'] for v in selected);total=sum(v['normal_epochs'] for v in selected)
                routes[trip]=dict(alarm=sum(v['normal_alarm_epochs'] for v in selected)/n,unavailable=sum(v['normal_eligible_denied'] for v in selected)/n,
                    total_unavailable=sum(v['normal_total_denied'] for v in selected)/total,u0=1-n/total,unknown=sum(v['unknown_scored_epochs'] for v in selected))
            data[method]=routes
        candidate=data['joint_history_gru6'];raw=data['raw_history_gru3'];stored=summary['normal']['per_seed'][str(seed)]
        limits=all(v['alarm']<=.05 and v['unavailable']<=.05 and v['total_unavailable']<=v['u0']+.05*(1-v['u0']) and not v['unknown'] for v in candidate.values())
        benefit=np.mean([v['unavailable'] for v in raw.values()])-np.mean([v['unavailable'] for v in candidate.values()])
        material=benefit>=.01 and max(v['unavailable'] for v in candidate.values())<=max(v['unavailable'] for v in raw.values())
        assert limits==stored['normal_limits_pass'] and bool(material)==stored['material_benefit_pass']
        for trip,value in candidate.items():
            for source,target in [('alarm','alarm_fraction'),('unavailable','eligible_unavailability'),('total_unavailable','total_unavailability'),('u0','structural_unscored_fraction')]:close(stored['routes'][trip][target],value[source])
        all_normal=all_normal and limits and material
    assert bool(all_normal)==summary['normal']['all_normal_gates_pass']
    failed=[];cells=0
    for trip in protocol['development_roles']['development_evaluate']:
        for case in protocol['fixed_attack_cases']+['pgd_joint_history_gru6','pgd_raw_history_gru3']:
            for seed in (7,19,43):
                select=lambda m:[r for r in records if r['trip']==trip and r['case']==case and r['seed']==seed and r['method']==m]
                candidate=select('joint_history_gru6')
                for reference in ('original_nn_frozen','raw_history_gru3'):
                    refs=select(reference);cells+=1;reasons=[]
                    count=sum(r['metrics']['scored_attack_epochs'] for r in candidate)
                    loss=100*(sum(r['metrics']['attack_alarm_epochs'] for r in refs)-sum(r['metrics']['attack_alarm_epochs'] for r in candidate))/count
                    if loss>5:reasons.append('recall_loss_over_5pp')
                    if max(r['metrics']['attack_longest_grant_run_epochs'] for r in candidate)>max(r['metrics']['attack_longest_grant_run_epochs'] for r in refs):reasons.append('longer_grant_run')
                    if sum(r['metrics']['attack_granted_observed_span_ms'] for r in candidate)>sum(r['metrics']['attack_granted_observed_span_ms'] for r in refs):reasons.append('added_grant_span')
                    for row in candidate:
                        ref=next(r for r in refs if r['segment']==row['segment'])
                        for event,old in zip(row['metrics']['events'],ref['metrics']['events'],strict=True):
                            if old['first_neural_alarm_delay_ms'] is not None and event['first_neural_alarm_delay_ms'] is None:reasons.append('new_missed_event')
                    if reasons:failed.append((trip,case,seed,reference,tuple(sorted(set(reasons)))))
    stored={(r['trip'],r['case'],r['seed'],r['reference'],tuple(r['reasons'])) for r in summary['attack']['failed_cells']}
    assert set(failed)==stored and cells==summary['attack']['checked_cells']==168
    assert summary['attack']['all_attack_gates_pass']==(not failed)
    assert summary['primary_gates_pass']==bool(all_normal and not failed)
    for key in ('active_policy_changed','reserved_validation_accessed','stage16_final_routes_accessed','eligible_for_validation'):assert summary[key] is False
    # Verify all declared white-box budgets against unmodified contexts.
    for name in sorted((output/'attacks').glob('*.json')):
        trace=json.loads(name.read_text());c=read_npz(output/'contexts'/(name.stem+'.npz'))
        trip,segment,*_=name.stem.split('__');original=read_npz(root/'artifacts/vehicle_transfer_stage32/contexts'/f'{trip}__{segment}.npz')
        delta=c['features']-original['features'];t=c['time_ms'];lo=t[0]+(t[-1]-t[0])/3;hi=t[0]+2*(t[-1]-t[0])/3;active=(t>=lo)&(t<=hi)
        close(delta[~active],0,1e-12);assert np.all(delta[active,:2]>=3-1e-10) and np.all(delta[active,:2]<=8+1e-10) and np.all(abs(delta[active,2])<=.4+1e-10)
        assert len(trace['objectives'])==21 and trace['best_objective']==min(trace['objectives'])
        assert trace['best_iteration']==trace['objectives'].index(min(trace['objectives']))
        states=read_npz(output/'attacks'/trace['iterates_file'])['iterates']
        assert states.shape==(21,len(t),3)
        close(states[trace['best_iteration']],c['features'],1e-12)
        target=name.stem.split('__pgd_',1)[1]
        objective_values=[]
        for state in states:
            changes=state-original['features']
            close(changes[~active],0,1e-12)
            assert np.all(changes[active,:2]>=3-1e-10) and np.all(changes[active,:2]<=8+1e-10) and np.all(abs(changes[active,2])<=.4+1e-10)
            assert np.all(state[active,0]+physical['gnss_offset']<=100+1e-10) and np.all(state[active,1]<=100+1e-10)
            j,p=independent_features(state,t,c['start_ms'])
            losses=[]
            for seed in (7,19,43):
                identifier=f'{target}__{seed}';_,_,_,score=neural_values(networks[target,seed],j,p,columns[target],lock['scaler'],lock['profiles'][identifier])
                excess=score[active[p]]-np.float32(lock['thresholds'][identifier])
                losses.append(np.logaddexp(np.float32(0),excess).mean())
            objective_values.append(float(np.mean(losses)))
        close(trace['objectives'],objective_values,1e-4)
    return dict(status='PASS',indexed_files_verified=verified,decision_rows_independently_recomputed=len(records),
        base_policy_streams_replayed=len(grouped),gru_checkpoints_independently_loaded=9,gate_cells_recomputed=cells,
        standalone_physics_reference_rows_recomputed=len(physics),pgd_iterates_independently_replayed=168,
        maximum_independent_prediction_error=max_prediction,calibration_profiles_and_thresholds_recomputed=True,
        original_graph_encoder_independently_reimplemented=False,base_policy_independently_reimplemented=False,
        new_fit_performed=False,reserved_validation_accessed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.root.resolve();result=audit(root,a.output or root/'artifacts/vehicle_joint_phase2')
    print(json.dumps(result,indent=2))
