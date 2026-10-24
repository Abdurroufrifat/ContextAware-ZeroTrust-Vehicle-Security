"""Frozen-weight vehicle enrollment compared on controlled RAV4 development attacks."""
import argparse
import io
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_vehicle_development_stage17 as prior  # noqa: E402
from collect_vehicle_data_stage24 import run as verify_data  # noqa: E402
from run_vehicle_ai_stage13 import base_grants  # noqa: E402
from run_vehicle_transfer_stage25 import native_frames,CONTRACT,CONTRACT_SHA  # noqa: E402
from vehicle_ai_data13 import features  # noqa: E402
from vehicle_recovery_scenarios10b import perturb  # noqa: E402
from coact_cps.vehicle.ai_models13 import create_adapter,score_nodes  # noqa: E402
from coact_cps.vehicle.native_detectors import ModelConfig  # noqa: E402
from coact_cps.vehicle.transfer25 import scaled_windows  # noqa: E402
from coact_cps.vehicle.temporal23 import summarize  # noqa: E402
from coact_cps.vehicle.enrollment26 import enroll_profile,calibrate_threshold,score,screen  # noqa: E402

MANIFEST25='73115b3a9d3f2aeeaa1e4bc05350761eec3ef8f1201df5ff8aef9c4be8ab35a6'
FIT,CALIBRATE,EVALUATE='rav24_01','rav24_02','rav24_03'


def verify_saved(root):
    folder=root/'artifacts/vehicle_transfer_stage25'
    raw=(folder/'file_hashes.json').read_bytes()
    if prior.sha(raw)!=MANIFEST25:raise ValueError('Stage25 saved index mismatch')
    index=json.loads(raw)
    for name,digest in index.items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or prior.sha(path.read_bytes())!=digest:
            raise ValueError('Stage25 indexed payload mismatch')
    return folder


def run(root,source,output):
    root,output=Path(root).resolve(),Path(output)
    if output.exists():raise ValueError('output must not already exist')
    z=prior.read_source(source);protocol,lock,_,checked=prior.audit_source(z)
    saved=verify_saved(root);verify_data(root)
    data=json.loads((root/'data/reference/vehicle_data_stage24/protocol.json').read_text())
    raw=(root/CONTRACT).read_bytes()
    if prior.sha(raw)!=CONTRACT_SHA:raise ValueError('original contract mismatch')
    contract=json.loads(raw);model=ModelConfig(**contract['model']);cfg=protocol['config']
    models={m['seed']:m for m in lock['models'] if m['method']=='coact_physics'}
    profiles={}
    for seed in prior.SEEDS:
        with np.load(saved/'blocks'/f'{FIT}__{seed}.npz',allow_pickle=False) as fit:
            p=enroll_profile(fit['nodes'],np.zeros(len(fit['nodes']),dtype=bool))
        with np.load(saved/'blocks'/f'{CALIBRATE}__{seed}.npz',allow_pickle=False) as cal:
            q=calibrate_threshold(cal['nodes'],np.zeros(len(cal['nodes']),dtype=bool),p)
        profiles[seed]=dict(profile=p,threshold=q)
    trip=next(t for t in data['trips'] if t['id']==EVALUATE)
    original,_=native_frames(root,trip)
    adapters={}
    for seed in prior.SEEDS:
        a=create_adapter('coact_physics',cfg,root,seed)
        raw=z.read(prior.ROOT+f'fit/seed_{seed}/coact_physics.pt')
        if prior.sha(raw)!=models[seed]['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        a.model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
        a.model.eval();adapters[seed]=a
    code=['scripts/run_vehicle_enrollment_stage26.py','src/coact_cps/vehicle/enrollment26.py',
          'scripts/run_vehicle_transfer_stage25.py','src/coact_cps/vehicle/transfer25.py',
          'scripts/run_vehicle_ai_stage13.py','scripts/vehicle_recovery_scenarios10b.py',
          'scripts/vehicle_session_scenarios.py','scripts/vehicle_native_scenarios.py',
          'src/coact_cps/vehicle/temporal23.py','src/coact_cps/vehicle/ai_models13.py',
          'src/coact_cps/models/temporal_graph.py']
    frozen=dict(schema='vehicle-enrollment-development-stage26-v1',original_source_sha256=prior.SOURCE_SHA256,
                stage25_manifest_sha256=MANIFEST25,model_contract_sha256=CONTRACT_SHA,
                roles=dict(fit_normal=FIT,threshold_normal=CALIBRATE,exploratory_attack_check=EVALUATE),
                design_sha256=prior.sha((root/'STAGE26_FROZEN_DESIGN.txt').read_bytes()),
                code_sha256={n:prior.sha((root/n).read_bytes()) for n in code},
                profiles={str(s):p for s,p in profiles.items()},models=[models[s] for s in prior.SEEDS],
                feature_scaler=lock['scaler'],cases=cfg['cases'],model_training=False,
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                reserved_validation_accessed=False,stage16_final_routes_accessed=False,eligible_for_test=False)
    output.mkdir(parents=True);(output/'blocks').mkdir();(output/'contexts').mkdir()
    (output/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[];replay=[]
    for case in cfg['cases']:
        changed=perturb(original,case,model,contract['session_config'])
        labels=np.array([case!='normal' and max(abs(a.gnss_speed-b.gnss_speed),
            abs(a.wheel_speed-b.wheel_speed),abs(a.imu_accel-b.imu_accel))>1e-9
            for a,b in zip(changed,original,strict=True)],dtype=bool)
        base,issued,quorum=base_grants(changed,contract)
        c=dict(features=features(changed,model),time_ms=np.array([f.time_ms for f in changed]),
               start_ms=np.array([f.interval_start_ms for f in changed]),labels=labels,
               base_grant=base,issued=issued,quorum=quorum)
        x,positions=scaled_windows(c['features'],c['time_ms'],c['start_ms'],cfg['window'],lock['scaler'])
        c['positions']=positions
        np.savez_compressed(output/'contexts'/f'{case}.npz',**c)
        for seed,a in adapters.items():
            spec=models[seed];nodes=score_nodes(a,x,cfg,seed)
            if case=='normal':
                with np.load(saved/'blocks'/f'{EVALUATE}__{seed}.npz',allow_pickle=False) as old:
                    error=float(np.max(np.abs(nodes-old['nodes'])))
                    if not np.array_equal(positions,old['positions']) or error>0.001:
                        raise ValueError('Stage25 normal replay mismatch')
                replay.append(dict(seed=seed,maximum_absolute_node_error=error))
            arms={
                'original':(score(nodes,spec['node_calibration']),spec['threshold']),
                'enrolled':(score(nodes,profiles[seed]['profile']),profiles[seed]['threshold'])}
            block=dict(positions=positions,nodes=nodes);row=dict(case=case,seed=seed)
            for arm,(scores,threshold) in arms.items():
                alarm=scores>threshold;grant=np.zeros(len(labels),dtype=bool)
                grant[positions]=base[positions]&~alarm
                block.update({arm+'_score':scores,arm+'_alarm':alarm,arm+'_grant':grant})
                temporal=summarize(c['time_ms'],c['start_ms'],labels,positions,alarm,base,grant)
                row[arm]=dict(metrics=prior.metrics(c,alarm),temporal=temporal)
            np.savez_compressed(output/'blocks'/f'{case}__{seed}.npz',**block)
            rows.append(row)
    normal=[r for r in rows if r['case']=='normal']
    normal_summary={a:dict(alarm_fraction=float(np.mean([r[a]['metrics']['normal_endpoint_fpr'] for r in normal])),
        grant_fraction=float(np.mean([r[a]['metrics']['grant_fraction_all_epochs'] for r in normal])))
        for a in ('original','enrolled')}
    losses=[];span_deltas=[];new_misses=[];longer=[]
    for case in cfg['cases']:
        if case=='normal':continue
        group=[r for r in rows if r['case']==case]
        losses.append(dict(case=case,mean_tpr_loss=float(np.mean([
            r['original']['metrics']['active_endpoint_tpr']-r['enrolled']['metrics']['active_endpoint_tpr']
            for r in group]))))
        span_deltas.append(dict(case=case,mean_added_observed_span_ms=float(np.mean([
            r['enrolled']['temporal']['attack_granted_observed_span_ms']-
            r['original']['temporal']['attack_granted_observed_span_ms'] for r in group]))))
        for r in group:
            old,new=r['original']['temporal'],r['enrolled']['temporal']
            for i,(before,after) in enumerate(zip(old['events'],new['events'],strict=True)):
                if before['first_neural_alarm_delay_ms'] is not None and after['first_neural_alarm_delay_ms'] is None:
                    new_misses.append(dict(case=case,seed=r['seed'],event_index=i))
            if new['attack_longest_grant_run_epochs']>old['attack_longest_grant_run_epochs']:
                longer.append(dict(case=case,seed=r['seed'],original=old['attack_longest_grant_run_epochs'],
                                   enrolled=new['attack_longest_grant_run_epochs']))
    worst=max(losses,key=lambda r:r['mean_tpr_loss']);old,new=normal_summary['original'],normal_summary['enrolled']
    gate=screen(new['alarm_fraction'],old['alarm_fraction'],new['grant_fraction'],old['grant_fraction'],
                worst['mean_tpr_loss'],len(new_misses),bool(longer),any(r['mean_added_observed_span_ms']>0 for r in span_deltas))
    result=dict(status='DEVELOPMENT_ONLY_NOT_PROMOTED',eligible_for_test=False,policy_changed=False,
                reserved_validation_accessed=False,stage16_final_routes_accessed=False,model_training=False,
                source_files_verified=checked,normal_replay=replay,normal=normal_summary,
                worst_case_loss=worst,additional_missed_events=new_misses,longer_attack_grant_runs=longer,
                case_losses=losses,case_observed_span_deltas=span_deltas,development_screen=gate,rows=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    index={p.relative_to(output).as_posix():prior.sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'file_hashes.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','case_losses','case_observed_span_deltas')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage13-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.stage13_zip,args.output)
