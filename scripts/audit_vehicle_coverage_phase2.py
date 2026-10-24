"""Independent replay for the coverage intervention; no fitting or locked payload access.

Numerical replay uses the original independent auditor, not producer arithmetic.
The producer is imported only for its explicit protocol-family verification.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_vehicle_joint_phase2 import (read_npz,sha,close,independent_features,
    independent_metrics,verify_score_contract,verify_physics_reference,load_gru,neural_values)
import run_vehicle_coverage_phase2 as recipe


def verify_lock_protocol(root,output,lock):
    protocol=recipe.verify_protocol_family(root)
    if (lock['phase1_protocol_sha256']!=recipe.PARENT_PROTOCOL_SHA
        or lock['coverage_protocol_sha256']!=recipe.PROTOCOL_SHA
        or sha(output/'coverage_protocol.json')!=recipe.PROTOCOL_SHA):
        raise ValueError('coverage protocol binding mismatch')
    return protocol


def audit(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();torch.set_num_threads(2)
    manifest=json.loads((output/'file_hashes.json').read_text());verified=0
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='file_hashes.json'}
    assert actual==set(manifest),'output file set differs from frozen index'
    for relative,digest in manifest.items():
        path=(output/relative).resolve();assert path.is_relative_to(output) and sha(path)==digest;verified+=1
    summary=json.loads((output/'summary.json').read_text());lock=json.loads((output/'model_lock.json').read_text())
    protocol=verify_lock_protocol(root,output,lock)
    assert summary['coverage_protocol_sha256']==recipe.PROTOCOL_SHA
    assert lock['intervention']==protocol['intervention']
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
    return dict(status='PASS',experiment='single_normal_coverage_exchange',indexed_files_verified=verified,decision_rows_independently_recomputed=len(records),
        base_policy_streams_replayed=len(grouped),gru_checkpoints_independently_loaded=9,gate_cells_recomputed=cells,
        standalone_physics_reference_rows_recomputed=len(physics),pgd_iterates_independently_replayed=168,
        maximum_independent_prediction_error=max_prediction,calibration_profiles_and_thresholds_recomputed=True,
        original_graph_encoder_independently_reimplemented=False,base_policy_independently_reimplemented=False,
        new_fit_performed=False,reserved_validation_accessed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.root.resolve();result=audit(root,a.output or root/'artifacts/vehicle_coverage_phase2')
    print(json.dumps(result,indent=2))
