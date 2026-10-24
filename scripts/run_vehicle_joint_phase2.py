"""One frozen development experiment; no reserved-route access or promotion."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from coact_cps.vehicle import joint_phase2 as core
from coact_cps.vehicle.ai_models13 import create_adapter, score_nodes
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.transfer25 import scaled_windows
from vehicle_ai_data13 import features
from vehicle_native_data import NativeFrame
from vehicle_recovery_scenarios10b import perturb, period
from vehicle_native_scenarios import bounds
from run_vehicle_ai_stage13 import base_grants
import run_vehicle_development_stage17 as source13
import run_vehicle_transfer_stage32 as saved32

PROTOCOL_SHA='a77d26a7ac3ff09b24d133ba0e15852062711da56876552282893e4b0d9157e9'
INDEX32_SHA='ca9d40d98e4eb34a4f65e097809727c3c689f8fe084af46a556397e0401619d7'
SOURCE_SHA=source13.SOURCE_SHA256
METHODS=core.ARMS+('original_nn_frozen',)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_protocol(path):
    if sha(path)!=PROTOCOL_SHA:raise ValueError('Phase1 protocol pin mismatch')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_role(root, protocol, role):
    result={}
    for row in protocol['development_segments']:
        if row['role']!=role:continue
        trip,segment=row['trip'],row['segment']
        core.require_role(protocol['development_segments'],trip,segment,role)
        key=f'{trip}__{segment}'
        with np.load(Path(root)/'artifacts/vehicle_transfer_stage32/contexts'/f'{key}.npz',allow_pickle=False) as z:
            c={k:z[k] for k in z.files}
        c['joint']=core.joint_features(torch.tensor(c['features'],dtype=torch.float64),c['time_ms'],c['start_ms'])
        c['positions']=core.valid_positions(c['joint'],c['time_ms'],c['start_ms'])
        if not len(c['positions']):raise ValueError('no common causal forecast endpoints')
        c['trip']=trip;c['segment']=segment;c['route_id']=row.get('route_id',trip)
        result[key]=c
    if not result:raise ValueError('empty declared data role')
    return result


def frames_from_context(c, route_id, model):
    x=c['features'];times=c['source_time_ms']
    return [NativeFrame(route_id,i,int(t),int(c['start_ms'][i]),int(times[i,0]),
            int(times[i,1]),int(times[i,2]),float(x[i,0]+model.gnss_offset),float(x[i,1]),
            float(x[i,2]+model.imu_offset),int(c['maximum_gap_ms'][i])) for i,t in enumerate(c['time_ms'])]


def train_history(x, y, routes, seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    model=core.HistoryGRU(x.shape[-1]);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    weights=torch.tensor(core.route_weights(routes),dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed);history=[]
    for _ in range(20):
        model.train();total=0.
        for indices in torch.randperm(len(x),generator=generator).split(64):
            optimizer.zero_grad();error=(model(x[indices])-y[indices]).square().mean(1)
            loss=(error*weights[indices]).mean();loss.backward();optimizer.step()
            total+=float(loss.detach())*len(indices)
        history.append(total/len(x))
    model.eval()
    return model,history


def check_alignment(root, data, plan31, quality31, model, scaler):
    plan24=json.loads((root/'data/reference/vehicle_data_stage24/protocol.json').read_text())
    eligible31={(r['trip'],r['segment']) for r in quality31['eligible_segments']}
    errors=[]
    for key,c in data.items():
        legacy=c['trip'].startswith('rav24')
        plan=plan24 if legacy else plan31
        trip=next(r for r in plan['trips'] if r['id']==c['trip'])
        eligible={(r['id'],seg) for r in plan24['trips'] if r['role']=='development' for seg in r['segments']} if legacy else eligible31
        fresh,_,_=saved32.segment_context(root,trip,c['segment'],eligible,'legacy' if legacy else 'expanded',model,scaler)
        error=float(np.max(np.abs(fresh['features']-c['features'])))
        if error>1e-12:raise ValueError('saved normal alignment differs from pinned raw replay')
        for name in ('time_ms','start_ms','source_time_ms','maximum_gap_ms'):
            if not np.array_equal(fresh[name],c[name]):raise ValueError('saved alignment clock/metadata mismatch')
        errors.append(dict(segment=key,maximum_feature_error=error))
    return errors


def fit_all(data, output):
    unique_values=[];unique_routes=[]
    for c in data.values():
        endpoints=np.unique((c['positions'][:,None]-np.arange(31,-1,-1)).ravel())
        unique_values.append(c['joint'][endpoints].numpy());unique_routes.extend([c['trip']]*len(endpoints))
    scaler=core.fit_scaler(np.concatenate(unique_values),np.asarray(unique_routes))
    write_json(output/'fit/scaler.json',scaler)
    models={};profiles={};training=[]
    for method,channels in core.CHANNELS.items():
        xs=[];ys=[];routes=[]
        for c in data.values():
            x,y=core.standardized_windows(c['joint'],c['positions'],channels,scaler)
            xs.append(x);ys.append(y)
            routes.extend([c['trip']]*len(y))
        x,y=torch.cat(xs),torch.cat(ys);route_array=np.asarray(routes)
        for seed in (7,19,43):
            model,loss=train_history(x,y,route_array,seed)
            with torch.no_grad():
                prediction=torch.cat([model(b) for b in x.split(128)])
                residual=(y-prediction).abs().numpy()
            profile=core.fit_profile(residual,route_array)
            checkpoint=output/'fit'/f'{method}__{seed}.pt';torch.save(model.state_dict(),checkpoint)
            np.savez_compressed(output/'fit'/f'{method}__{seed}.npz',prediction=prediction.numpy(),residual=residual,routes=route_array)
            models[method,seed]=model;profiles[f'{method}__{seed}']=profile
            training.append(dict(method=method,seed=seed,epochs=20,loss=loss,windows=len(y),
                parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(checkpoint)))
            print(f'fit {method} seed{seed}: {len(y)} windows',flush=True)
    write_json(output/'fit/profiles.json',profiles);write_json(output/'fit/training.json',training)
    return models,scaler,profiles,training


def scores_for(models,scaler,profiles,joint,positions,method,seed):
    if method=='separate_gru_or':
        branches=[scores_for(models,scaler,profiles,joint,positions,m,seed) for m in ('raw_history_gru3','innovation_history_gru3')]
        contributions=torch.cat([b[3] for b in branches],dim=1)
        return contributions.max(1).values,torch.cat([b[1] for b in branches],1),torch.cat([b[2] for b in branches],1),contributions
    return core.model_score(models[method,seed],joint,positions,core.CHANNELS[method],scaler,profiles[f'{method}__{seed}'])


def calibrate_all(models,scaler,profiles,data,output):
    thresholds={};route_values=defaultdict(lambda:defaultdict(list))
    for key,c in data.items():
        for method in core.ARMS:
            for seed in (7,19,43):
                with torch.no_grad():score,residual,prediction,contributions=scores_for(models,scaler,profiles,c['joint'],c['positions'],method,seed)
                route_values[f'{method}__{seed}'][c['trip']].append(score.numpy())
                np.savez_compressed(output/'calibration'/f'{key}__{method}__{seed}.npz',positions=c['positions'],score=score.numpy(),residual=residual.numpy(),prediction=prediction.numpy(),contributions=contributions.numpy())
    for model_id,by_route in route_values.items():
        thresholds[model_id]=core.calibrate({r:np.concatenate(v) for r,v in by_route.items()})
    write_json(output/'calibration/thresholds.json',thresholds)
    return thresholds


def _route_counts(rows):
    counts=defaultdict(lambda:defaultdict(int))
    fields=('normal_eligible_epochs','normal_alarm_epochs','normal_eligible_denied','normal_epochs','normal_total_denied','epochs','scored_epochs','unknown_scored_epochs')
    for r in rows:
        for field in fields:counts[r['trip']][field]+=r['metrics'][field]
    return counts


def normal_gates(rows):
    normal=[r for r in rows if r['case']=='normal'];per_seed={}
    for seed in (7,19,43):
        pairs={m:_route_counts([r for r in normal if r['method']==m and r['seed']==seed]) for m in core.ARMS[:2]}
        candidate=pairs['joint_history_gru6'];raw=pairs['raw_history_gru3'];routes={};baseline={}
        for trip,c in candidate.items():
            n=c['normal_eligible_epochs'];all_n=c['normal_epochs']
            if n<=0 or all_n<=0:raise ValueError('empty required normal denominator')
            u0=1-n/all_n
            routes[trip]=dict(alarm_fraction=c['normal_alarm_epochs']/n,eligible_unavailability=c['normal_eligible_denied']/n,
                total_unavailability=c['normal_total_denied']/all_n,structural_unscored_fraction=u0,total_budget=u0+.05*(1-u0),unknown_scored_epochs=c['unknown_scored_epochs'])
        for trip,c in raw.items():baseline[trip]=c['normal_eligible_denied']/c['normal_eligible_epochs']
        if not routes or set(routes)!=set(baseline):raise ValueError('normal route coverage mismatch')
        macro=float(np.mean([v['alarm_fraction'] for v in routes.values()]));worst=max(v['alarm_fraction'] for v in routes.values())
        macro_u=float(np.mean([v['eligible_unavailability'] for v in routes.values()]));worst_u=max(v['eligible_unavailability'] for v in routes.values())
        limits=all(v['alarm_fraction']<=.05 and v['eligible_unavailability']<=.05 and v['total_unavailability']<=v['total_budget'] and v['unknown_scored_epochs']==0 for v in routes.values())
        benefit=float(np.mean(list(baseline.values())))-macro_u
        material=benefit>=.01 and worst_u<=max(baseline.values())
        per_seed[str(seed)]=dict(routes=routes,macro_alarm_fraction=macro,worst_route_alarm_fraction=worst,
            macro_eligible_unavailability=macro_u,worst_route_eligible_unavailability=worst_u,
            normal_limits_pass=bool(limits),availability_gain_over_raw_percentage_points=100*benefit,
            material_benefit_pass=bool(material))
    return dict(per_seed=per_seed,all_normal_gates_pass=all(v['normal_limits_pass'] and v['material_benefit_pass'] for v in per_seed.values()))


def attack_gates(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['case']!='normal':grouped[row['trip'],row['case'],row['seed'],row['method']].append(row)
    checks=[];failed=[]
    for (trip,case,seed,method),candidate in grouped.items():
        if method!='joint_history_gru6':continue
        c_by_seg={r['segment']:r['metrics'] for r in candidate}
        for reference in ('original_nn_frozen','raw_history_gru3'):
            refs=grouped.get((trip,case,seed,reference),[]);r_by_seg={r['segment']:r['metrics'] for r in refs}
            reasons=[]
            if set(c_by_seg)!=set(r_by_seg):reasons.append('missing_reference_segment')
            if not reasons:
                c_n=sum(v['scored_attack_epochs'] for v in c_by_seg.values());r_n=sum(v['scored_attack_epochs'] for v in r_by_seg.values())
                if c_n<=0 or c_n!=r_n:reasons.append('attack_denominator_mismatch')
                else:
                    loss=100*(sum(v['attack_alarm_epochs'] for v in r_by_seg.values())/r_n-sum(v['attack_alarm_epochs'] for v in c_by_seg.values())/c_n)
                    if loss>5.:reasons.append('recall_loss_over_5pp')
                for segment,c in c_by_seg.items():
                    r=r_by_seg[segment]
                    if len(c['events'])!=len(r['events']):reasons.append('event_coverage_mismatch');continue
                    for ce,re in zip(c['events'],r['events'],strict=True):
                        if ce['first_epoch']!=re['first_epoch'] or ce['last_epoch']!=re['last_epoch']:reasons.append('event_identity_mismatch')
                        if re['first_neural_alarm_delay_ms'] is not None and ce['first_neural_alarm_delay_ms'] is None:reasons.append('new_missed_event')
                if max(v['attack_longest_grant_run_epochs'] for v in c_by_seg.values())>max(v['attack_longest_grant_run_epochs'] for v in r_by_seg.values()):reasons.append('longer_grant_run')
                if sum(v['attack_granted_observed_span_ms'] for v in c_by_seg.values())>sum(v['attack_granted_observed_span_ms'] for v in r_by_seg.values()):reasons.append('added_grant_span')
            check=dict(trip=trip,case=case,seed=seed,reference=reference,reasons=sorted(set(reasons)),pass_check=not reasons)
            if not reasons:check['recall_loss_percentage_points']=loss
            checks.append(check)
            if reasons:failed.append(check)
    return dict(checked_cells=len(checks),failed_cells=failed,all_attack_gates_pass=bool(checks) and not failed)


def physics_reference(context):
    """Existing physical/base policy, with native and common-endpoint accounting.

    Its denial indicator is a policy decision, not a standalone anomaly score.
    """
    result={}
    for scope in ('native','matched'):
        c=dict(context)
        if scope=='native':c['positions']=np.arange(len(c['time_ms']),dtype=np.int64)
        score=(~c['base_grant'][c['positions']]).astype(float)
        _,metric=core.evaluate_decisions(c,score,.5)
        def rename(v):
            if isinstance(v,list):return [rename(x) for x in v]
            if not isinstance(v,dict):return v
            return {('active_policy_denial_fraction' if k=='active_endpoint_tpr' else k.replace('neural','policy').replace('alarm','denial')):rename(x) for k,x in v.items()}
        result[scope]=rename(metric)
    return result


def verify_inputs(root,protocol,source):
    folder=root/'artifacts/vehicle_transfer_stage32'
    if sha(folder/'file_hashes.json')!=INDEX32_SHA:raise ValueError('Stage32 output index pin mismatch')
    for relative,digest in json.loads((folder/'file_hashes.json').read_text()).items():
        path=(folder/relative).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path)!=digest:raise ValueError('Stage32 payload mismatch')
    prior=json.loads((folder/'frozen_protocol.json').read_text())
    for name,digest in prior['code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('pinned preprocessing/model source changed: '+name)
    for name,digest in protocol['attack_generator_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('pinned attack generator changed: '+name)
    plan31,quality31=saved32.verify(root)
    if sha(source)!=SOURCE_SHA:raise ValueError('Stage13 archive pin mismatch')
    z=source13.read_source(source);source_protocol,lock,source_rows,count=source13.audit_source(z)
    contract_path=root/saved32.CONTRACT
    if sha(contract_path)!=saved32.CONTRACT_SHA:raise ValueError('physical model contract pin mismatch')
    return z,source_protocol,lock,source_rows,count,json.loads(contract_path.read_text()),plan31,quality31


def run(root,source,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('output already exists; never overwrite frozen results')
    protocol=load_protocol(root/'research/phase1_method_decision/PHASE2_PROTOCOL.json')
    z,sp,lock,source_rows,checked,contract,plan31,quality31=verify_inputs(root,protocol,source)
    model=ModelConfig(**contract['model']);cfg=sp['config']
    output.mkdir(parents=True)
    for folder in ('fit','calibration','contexts','blocks','attacks'):(output/folder).mkdir()
    started=time.monotonic(); torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    fit=load_role(root,protocol,'fit');alignment=check_alignment(root,fit,plan31,quality31,model,lock['scaler'])
    models,scaler,profiles,training=fit_all(fit,output)
    calibration=load_role(root,protocol,'calibrate');alignment+=check_alignment(root,calibration,plan31,quality31,model,lock['scaler'])
    thresholds=calibrate_all(models,scaler,profiles,calibration,output)
    immutable={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    code={p.relative_to(root).as_posix():sha(p) for parent in ('src','scripts') for p in sorted((root/parent).rglob('*.py'))}
    frozen=dict(schema='coact-phase2-model-lock-v1',phase1_protocol_sha256=PROTOCOL_SHA,
        stage32_output_index_sha256=INDEX32_SHA,original_source_sha256=SOURCE_SHA,original_source_files_verified=checked,
        training=training,scaler=scaler,profiles=profiles,thresholds=thresholds,fit_calibration_sha256=immutable,
        code_sha256=code,environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,torch_threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        role_order=['fit','calibrate','lock_written','development_evaluate'],validation_payload_accessed=False)
    write_json(output/'model_lock.json',frozen);lock_hash=sha(output/'model_lock.json')
    del fit,calibration
    evaluation=load_role(root,protocol,'development_evaluate');alignment+=check_alignment(root,evaluation,plan31,quality31,model,lock['scaler'])
    original={};specs={v['seed']:v for v in lock['models'] if v['method']=='coact_physics'};original_replay=[]
    for seed in (7,19,43):
        a=create_adapter('coact_physics',cfg,root,seed)
        raw=z.read(source13.ROOT+f'fit/seed_{seed}/coact_physics.pt')
        if hashlib.sha256(raw).hexdigest()!=specs[seed]['checkpoint_sha256']:raise ValueError('original checkpoint mismatch')
        a.model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True));a.model.eval();original[seed]=a
    rows=[];errors=[];streams=[];physics=[]
    for key,c in evaluation.items():
        original_frames=frames_from_context(c,c['route_id'],model)
        cases=['normal']+protocol['fixed_attack_cases']+['pgd_joint_history_gru6','pgd_raw_history_gru3']
        for case in cases:
            try:
                if case.startswith('pgd_'):
                    target=case.removeprefix('pgd_');begin,end=bounds(original_frames)
                    active=np.array([begin<=f.time_ms<=end for f in original_frames],dtype=bool)
                    active_positions=torch.as_tensor(active[c['positions']])
                    def objective(attacked):
                        joint=core.joint_features(attacked,c['time_ms'],c['start_ms'])
                        losses=[]
                        for seed in (7,19,43):
                            score,*_=scores_for(models,scaler,profiles,joint,c['positions'],target,seed)
                            losses.append(torch.nn.functional.softplus(score[active_positions]-thresholds[f'{target}__{seed}']).mean())
                        return torch.stack(losses).mean()
                    attacked,trace=core.projected_attack(torch.tensor(c['features'],dtype=torch.float64),active,objective,model.gnss_offset)
                    iterate_file=f'{key}__{case}.npz'
                    np.savez_compressed(output/'attacks'/iterate_file,iterates=trace.pop('iterates'))
                    trace['iterates_file']=iterate_file
                    write_json(output/'attacks'/f'{key}__{case}.json',trace)
                    changed=[replace(f,gnss_speed=float(attacked[i,0])+model.gnss_offset,wheel_speed=float(attacked[i,1]),imu_accel=float(attacked[i,2])+model.imu_offset) for i,f in enumerate(original_frames)]
                    phases=np.array(['active' if active[i] else 'before' if f.time_ms<begin else 'after' for i,f in enumerate(original_frames)])
                else:
                    changed=perturb(original_frames,case,model,contract['session_config'])
                    phases=np.array([period(f,original_frames,case) for f in original_frames])
                f=features(changed,model)
                labels=np.max(np.abs(f-c['features']),axis=1)>1e-9 if case!='normal' else np.zeros(len(f),dtype=bool)
                base,issued,quorum=base_grants(changed,contract)
                context={n:c[n] for n in ('time_ms','start_ms','source_time_ms','maximum_gap_ms','positions')}
                context.update(features=f,labels=labels,base_grant=base,issued=issued,quorum=quorum,period=phases)
                physics.append(dict(trip=c['trip'],segment=c['segment'],case=case,
                    method='original_physics_policy',seed=None,context=f'{key}__{case}.npz',
                    **physics_reference(context),periods={p:dict(epochs=int((phases==p).sum()),native_grants=int(base[phases==p].sum())) for p in ('before','active','after')}))
                context_name=f'{key}__{case}.npz';np.savez_compressed(output/'contexts'/context_name,**context)
                joint=core.joint_features(torch.tensor(f,dtype=torch.float64),c['time_ms'],c['start_ms'])
                if not np.array_equal(core.valid_positions(joint,c['time_ms'],c['start_ms']),c['positions']):raise ValueError('attack changed shared endpoint coverage')
                original_windows,old_p=scaled_windows(f,c['time_ms'],c['start_ms'],32,lock['scaler'])
                select=np.searchsorted(old_p,c['positions'])
                if not np.array_equal(old_p[select],c['positions']):raise ValueError('original endpoint alignment mismatch')
                for seed in (7,19,43):
                    nodes=score_nodes(original[seed],original_windows[select],cfg,seed)
                    original_score=((nodes-specs[seed]['node_calibration']['median'])/specs[seed]['node_calibration']['scale']).max(1)
                    if case=='normal':
                        with np.load(root/'artifacts/vehicle_transfer_stage32/neural'/f'{key}__{seed}.npz') as old:
                            old_select=np.searchsorted(old['positions'],c['positions'])
                            error=float(np.max(np.abs(nodes-old['nodes'][old_select])))
                            if error>.001:raise ValueError('original normal replay tolerance exceeded')
                            original_replay.append(dict(segment=key,seed=seed,maximum_node_error=error))
                    for method in METHODS:
                        if method=='original_nn_frozen':
                            score=original_score;threshold=specs[seed]['threshold'];extra=dict(residual=nodes)
                        else:
                            with torch.no_grad():score_t,residual,prediction,contributions=scores_for(models,scaler,profiles,joint,c['positions'],method,seed)
                            score=score_t.numpy();threshold=thresholds[f'{method}__{seed}']
                            extra=dict(residual=residual.numpy(),prediction=prediction.numpy(),contributions=contributions.numpy())
                        block,metric=core.evaluate_decisions(context,score,threshold)
                        phase_metrics={phase:dict(epochs=int((phases==phase).sum()),grants=int(block['grant'][phases==phase].sum())) for phase in ('before','active','after')}
                        block_name=f'{key}__{case}__{method}__{seed}.npz'
                        np.savez_compressed(output/'blocks'/block_name,positions=c['positions'],threshold=np.array(threshold),**block,**extra)
                        rows.append(dict(trip=c['trip'],segment=c['segment'],case=case,method=method,seed=seed,context=context_name,block=block_name,metrics=metric,periods=phase_metrics))
                streams.append(dict(segment=key,case=case,attack_epochs=int(labels.sum()),base_grants=int(base.sum())))
                print(f'evaluated {key} {case}',flush=True)
            except (ValueError,RuntimeError) as exc:
                errors.append(dict(segment=key,case=case,error=str(exc)));print(f'incomplete {key} {case}: {exc}',flush=True)
    for relative,digest in immutable.items():
        if sha(output/relative)!=digest:raise ValueError('fit/calibration file modified during evaluation')
    if sha(output/'model_lock.json')!=lock_hash:raise ValueError('frozen model lock changed')
    normal=normal_gates(rows);attack=attack_gates(rows)
    expected=len(evaluation)*(1+len(protocol['fixed_attack_cases'])+2)*len(METHODS)*3
    coverage=not errors and len(rows)==expected and len({(r['trip'],r['segment'],r['case'],r['method'],r['seed']) for r in rows})==expected
    passed=coverage and normal['all_normal_gates_pass'] and attack['all_attack_gates_pass']
    result=dict(status='PRIMARY_PASS_SECONDARY_COMPARISONS_REQUIRED' if passed else 'DEVELOPMENT_GATES_FAILED_NOT_PROMOTED',
        phase=2,candidate='joint_history_gru6',active_policy_changed=False,reserved_validation_accessed=False,
        stage16_final_routes_accessed=False,eligible_for_validation=False,secondary_baselines_run=False,
        original_source_files_verified=checked,model_lock_sha256=lock_hash,rows_expected=expected,rows_observed=len(rows),
        complete_coverage=coverage,errors=errors,normal=normal,attack=attack,primary_gates_pass=bool(passed),
        original_normal_replay=original_replay,raw_alignment_replay=alignment,streams=streams,
        wall_seconds=time.monotonic()-started,rows=rows)
    write_json(output/'physics_reference.json',dict(description='Deterministic existing policy; denial metrics include its integrity, quorum and physical checks. Native uses all aligned epochs; matched restricts grants to the shared complete-history endpoints. Denial recall is not a neural detector recall.',rows=physics))
    result['physics_reference_rows']=len(physics)
    write_json(output/'summary.json',result)
    write_json(output/'file_hashes.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','streams','raw_alignment_replay','original_normal_replay','attack')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--stage13-zip',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root.resolve()
    run(root,args.stage13_zip or root/'data/reference/vehicle_phase2/original_stage13.zip',args.output or root/'artifacts/vehicle_joint_phase2')
