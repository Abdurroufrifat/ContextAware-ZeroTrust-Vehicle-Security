"""One normal-cohort coverage exchange using the unchanged Phase2 recipe.

Imports the frozen fitting, calibration, scoring, attack and gate primitives.
It does not change or bypass the original Phase1 protocol guard.
"""
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

from copy import deepcopy
from run_vehicle_joint_phase2 import (
    sha, write_json, load_role, frames_from_context, check_alignment,
    fit_all, calibrate_all, scores_for, normal_gates, attack_gates,
    physics_reference, verify_inputs, INDEX32_SHA, SOURCE_SHA, METHODS,
)

PARENT_PROTOCOL_PATH='research/phase1_method_decision/PHASE2_PROTOCOL.json'
PARENT_PROTOCOL_SHA='a77d26a7ac3ff09b24d133ba0e15852062711da56876552282893e4b0d9157e9'
PROTOCOL_PATH='research/phase2_coverage/COVERAGE_PROTOCOL.json'
PROTOCOL_SHA='ec01d13054fc4d26927db6f7ba8ee951dfd33a8d8981e810c3831f1acabc397b'
EXPECTED_ROLES={'fit': ['rav24_01', 'rav24_02', 'rav31_02', 'rav31_03'], 'calibrate': ['rav31_04', 'rav31_05'], 'development_evaluate': ['rav24_03', 'rav31_06']}
EXPECTED_INTERVENTION={'name': 'single_normal_coverage_exchange', 'changed_factor': 'exchange rav24_02 and rav31_04 between fit and calibration only', 'selection_basis': 'previously inspected normal speed-support inventory; no best attack outcome selection', 'all_roles_previously_inspected': True, 'whole_split_is_chronological': False, 'validation_may_open': False, 'parent_protocol_sha256': 'a77d26a7ac3ff09b24d133ba0e15852062711da56876552282893e4b0d9157e9', 'proposal_sha256': '5e8ba7162aa4c3d50f24c3611916cbfbc69b13573bbea8a46a049d556a5ce7ba', 'parent_summary_sha256': '5865d282934a108ba0f87bee3ca7149c9303fb40b738d59ac503601f163e91c1', 'parent_model_lock_sha256': 'bb793597ce8636d430c6fd05600a5f506d2514a06255d4719c3e5279aa8db721', 'stop_rule': 'One run of all original gates. Failure remains negative, no automatic sweep; pass requires secondary comparisons before any validation review.'}


def validate_protocol(protocol, parent):
    expected=deepcopy(parent)
    expected.update(schema='coact-cps-phase2-coverage-v1',status='FROZEN_BEFORE_SINGLE_RUN',phase=2,
                    development_roles=EXPECTED_ROLES,intervention=EXPECTED_INTERVENTION)
    for row in expected['development_segments']:
        row['role']=next(k for k,v in EXPECTED_ROLES.items() if row['trip'] in v)
    if protocol!=expected:
        raise ValueError('Only the frozen normal-role exchange is authorized')
    return protocol


def load_protocol(path):
    if sha(path)!=PROTOCOL_SHA:
        raise ValueError('coverage protocol pin mismatch')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_protocol_family(root):
    parent_path=root/PARENT_PROTOCOL_PATH
    if sha(parent_path)!=PARENT_PROTOCOL_SHA:
        raise ValueError('parent protocol pin mismatch')
    protocol=load_protocol(root/PROTOCOL_PATH)
    validate_protocol(protocol,json.loads(parent_path.read_text()))
    if sha(root/'research/phase2_coverage/NEXT_EXPERIMENT_PROPOSAL.json')!=EXPECTED_INTERVENTION['proposal_sha256']:
        raise ValueError('diagnosis proposal pin mismatch')
    for name,field in [('summary.json','parent_summary_sha256'),('model_lock.json','parent_model_lock_sha256')]:
        if sha(root/'artifacts/vehicle_joint_phase2'/name)!=EXPECTED_INTERVENTION[field]:
            raise ValueError('parent evidence pin mismatch')
    return protocol


def run(root,source,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('output already exists; never overwrite frozen results')
    protocol=verify_protocol_family(root)
    z,sp,lock,source_rows,checked,contract,plan31,quality31=verify_inputs(root,protocol,source)
    model=ModelConfig(**contract['model']);cfg=sp['config']
    output.mkdir(parents=True)
    for folder in ('fit','calibration','contexts','blocks','attacks'):(output/folder).mkdir()
    write_json(output/'coverage_protocol.json',protocol)
    started=time.monotonic(); torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    fit=load_role(root,protocol,'fit');alignment=check_alignment(root,fit,plan31,quality31,model,lock['scaler'])
    models,scaler,profiles,training=fit_all(fit,output)
    calibration=load_role(root,protocol,'calibrate');alignment+=check_alignment(root,calibration,plan31,quality31,model,lock['scaler'])
    thresholds=calibrate_all(models,scaler,profiles,calibration,output)
    immutable={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    code={p.relative_to(root).as_posix():sha(p) for parent in ('src','scripts') for p in sorted((root/parent).rglob('*.py'))}
    frozen=dict(schema='coact-phase2-coverage-lock-v1',phase1_protocol_sha256=PARENT_PROTOCOL_SHA,
        coverage_protocol_sha256=PROTOCOL_SHA,intervention=EXPECTED_INTERVENTION,
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
        phase=2,experiment='single_normal_coverage_exchange',coverage_protocol_sha256=PROTOCOL_SHA,
        parent_summary_sha256=EXPECTED_INTERVENTION['parent_summary_sha256'],
        whole_split_is_chronological=False,all_roles_previously_inspected=True,candidate='joint_history_gru6',active_policy_changed=False,reserved_validation_accessed=False,
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
    run(root,args.stage13_zip or root/'data/reference/vehicle_phase2/original_stage13.zip',args.output or root/'artifacts/vehicle_coverage_phase2')
