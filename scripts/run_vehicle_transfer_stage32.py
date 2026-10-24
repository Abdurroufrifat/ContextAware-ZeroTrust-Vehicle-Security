"""Frozen normal-transfer comparison with quarantine enforcement and segment resets."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_vehicle_development_stage17 as prior
from collect_vehicle_data_stage24 import run as verify24
from collect_vehicle_data_stage31 import protocol as protocol31,inspect as inspect31
from vehicle_native_data import align_channels
from vehicle_integrity_data import wheel_receipt_view
from vehicle_ai_data13 import features
from coact_cps.vehicle.ai_models13 import create_adapter,score_nodes
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.transfer25 import scaled_windows
from coact_cps.vehicle.enrollment26 import score as neural_score
from coact_cps.vehicle.innovation29 import innovations,score as innovation_score
from coact_cps.vehicle.transfer32 import require_eligible,alarm_summary,aggregate

DATA31='9721cc59014931f10d917c797c0b267e4625603111360ab66f9312bf2ad45c6c'
SUMMARY31='fdec84a87c731845f9d178a1aa8118004dd93d322ce9f271a60e7562df92bb92'
SOURCES={'vehicle_enrollment_stage26':'53b291a59b80aaacb0f246982ca225c6f91994cb939f86599a0ba6142c3488c5',
         'vehicle_innovation_stage29':'d64719692e62b76779665626c7ec921d925c9f0e739631b06510b15f894ab501'}
CONTRACT='data/reference/vehicle_transfer_stage25/model_contract.json'
CONTRACT_SHA='dcf3d4c49f65e8b51e5b76d6d0ca2083969c0b4cdd533a767019b46fc1bdd811'
METHODS=('original_nn','enrolled_nn','innovation_max','innovation_covariance')


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def verify(root):
    if prior.sha((root/'data/reference/vehicle_data_stage31/protocol.json').read_bytes())!=DATA31:
        raise ValueError('Stage31 protocol pin mismatch')
    raw=(root/'artifacts/vehicle_data_stage31/summary.json').read_bytes()
    if prior.sha(raw)!=SUMMARY31:raise ValueError('Stage31 eligibility summary pin mismatch')
    p=protocol31(root);s=json.loads(raw)
    if inspect31(root,p)!=s:raise ValueError('Stage31 integrity/quarantine verification mismatch')
    verify24(root)
    for folder,pin in SOURCES.items():
        directory=root/'artifacts'/folder;raw=(directory/'file_hashes.json').read_bytes()
        if prior.sha(raw)!=pin:raise ValueError('fixed profile-source index mismatch')
        for name,digest in json.loads(raw).items():
            file=(directory/name).resolve()
            if not file.is_relative_to(directory.resolve()) or prior.sha(file.read_bytes())!=digest:
                raise ValueError('fixed profile-source payload mismatch')
    return p,s


def segment_context(root,trip,segment,eligible,pool,model,scaler):
    require_eligible(trip,segment,eligible)
    folder=root/f'data/raw/vehicle_data_stage{24 if pool=="legacy" else 31}'/trip['id']/f'segment_{segment}'
    channels=[];receipt=None
    for channel in ('GNSS/live_gnss_ublox','IMU/accelerometer','CAN/wheel_speed'):
        t=np.load(folder/channel/'t.npy',allow_pickle=False);v=np.load(folder/channel/'value.npy',allow_pickle=False)
        if channel=='CAN/wheel_speed':t,v,receipt=wheel_receipt_view(t,v)
        channels.append((t,v))
    frames=align_channels(trip['route_id'],*channels,stride=2,edge_s=2.0)
    f=features(frames,model);t=np.array([a.time_ms for a in frames],dtype=np.int64)
    s=np.array([a.interval_start_ms for a in frames],dtype=np.int64)
    x,p=scaled_windows(f,t,s,32,scaler)
    if not len(p):raise ValueError('no complete segment-local neural windows')
    innovation=innovations(f,t,s)[p]
    if not np.isfinite(innovation).all():raise ValueError('unavailable innovation history at a scored position')
    c=dict(features=f,time_ms=t,start_ms=s,positions=p,innovations=innovation,
        source_time_ms=np.array([[a.gnss_time_ms,a.imu_time_ms,a.wheel_time_ms] for a in frames],dtype=np.int64),
        maximum_gap_ms=np.array([a.maximum_gap_ms for a in frames],dtype=np.int64))
    return c,x,receipt


def run(root,source,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('output must not already exist')
    torch.set_num_threads(1)
    data31,quality31=verify(root)
    z=prior.read_source(source);source_protocol,lock,source_rows,checked=prior.audit_source(z)
    cfg=source_protocol['config']
    if cfg['window']!=32:raise ValueError('original neural window contract changed')
    raw=(root/CONTRACT).read_bytes()
    if prior.sha(raw)!=CONTRACT_SHA:raise ValueError('original model contract mismatch')
    model=ModelConfig(**json.loads(raw)['model']);specs={s['seed']:s for s in lock['models'] if s['method']=='coact_physics'}
    enrolled=json.loads((root/'artifacts/vehicle_enrollment_stage26/frozen_protocol.json').read_text())['profiles']
    innovation_profiles=json.loads((root/'artifacts/vehicle_innovation_stage29/profiles.json').read_text())
    profiles=dict(original={str(s):dict(profile=specs[s]['node_calibration'],threshold=specs[s]['threshold']) for s in prior.SEEDS},
                  enrolled=enrolled,innovation=innovation_profiles,feature_scaler=lock['scaler'])
    adapters={};replay=[]
    for seed in prior.SEEDS:
        adapter=create_adapter('coact_physics',cfg,root,seed)
        raw=z.read(prior.ROOT+f'fit/seed_{seed}/coact_physics.pt')
        if prior.sha(raw)!=specs[seed]['checkpoint_sha256']:raise ValueError('original checkpoint pin mismatch')
        adapter.model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True));adapter.model.eval()
        r=next(r for r in source_rows if r['trip']=='civic13_09' and r['case']=='normal' and r['seed']==seed)
        c,b=prior.checked_block(z,r,specs[seed]);x,p=scaled_windows(c['features'],c['time_ms'],c['start_ms'],32,lock['scaler'])
        if not np.array_equal(p,c['positions']):raise ValueError('original replay position mismatch')
        error=float(np.max(abs(score_nodes(adapter,x,cfg,seed)-b['nodes'])))
        if error>.001:raise ValueError('original residual replay tolerance exceeded')
        replay.append(dict(seed=seed,maximum_absolute_node_error=error));adapters[seed]=adapter
    data24=json.loads((root/'data/reference/vehicle_data_stage24/protocol.json').read_text())
    eligible31={(r['trip'],r['segment']) for r in quality31['eligible_segments']}
    planned=[]
    for pool,data in [('legacy',data24),('expanded',data31)]:
        for trip in data['trips']:
            if trip['role'] not in ('development','development_pool'):continue
            eligible={(t['id'],s) for t in data24['trips'] if t['role']=='development' for s in t['segments']} if pool=='legacy' else eligible31
            for segment in trip['segments']:
                if (trip['id'],segment) in eligible:planned.append((pool,trip,segment,eligible))
    code=('scripts/run_vehicle_transfer_stage32.py','src/coact_cps/vehicle/transfer32.py','scripts/vehicle_native_data.py',
          'scripts/vehicle_integrity_data.py','scripts/vehicle_ai_data13.py','src/coact_cps/vehicle/transfer25.py',
          'src/coact_cps/vehicle/innovation29.py','src/coact_cps/vehicle/ai_models13.py','src/coact_cps/models/temporal_graph.py')
    frozen=dict(schema='vehicle-expanded-normal-transfer-stage32-v1',source_zip_sha256=prior.SOURCE_SHA256,
        stage31_protocol_sha256=DATA31,stage31_summary_sha256=SUMMARY31,profile_source_manifests=SOURCES,
        model_contract_sha256=CONTRACT_SHA,models=[specs[s] for s in prior.SEEDS],methods=list(METHODS),
        planned_segments=[dict(pool=pool,trip=t['id'],route_id=t['route_id'],segment=s) for pool,t,s,_ in planned],
        quarantined_segments_not_preprocessed=quality31['quarantined_segments'],
        code_sha256={n:prior.sha((root/n).read_bytes()) for n in code},
        design_sha256=prior.sha((root/'STAGE32_FROZEN_DESIGN.txt').read_bytes()),
        per_segment_history_reset=True,torch_threads=1,training=False,parameters_changed=False)
    output.mkdir(parents=True)
    for name in ('contexts','neural','innovation'):(output/name).mkdir()
    (output/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    rows=[];prepared=[];excluded=[]
    for pool,trip,segment,eligible in planned:
        key=f'{trip["id"]}__{segment}'
        try:c,x,receipt=segment_context(root,trip,segment,eligible,pool,model,lock['scaler'])
        except ValueError as exc:
            excluded.append(dict(pool=pool,trip=trip['id'],segment=segment,reason=str(exc)));continue
        np.savez_compressed(output/'contexts'/f'{key}.npz',**c)
        prepared.append(dict(pool=pool,trip=trip['id'],segment=segment,wheel_receipt_audit=receipt))
        for seed,adapter in adapters.items():
            nodes=score_nodes(adapter,x,cfg,seed);block=dict(nodes=nodes,positions=c['positions'])
            for method,name in [('original_nn','original'),('enrolled_nn','enrolled')]:
                p=profiles[name][str(seed)];values=neural_score(nodes,p['profile']);alarm=values>p['threshold']
                block.update({method+'_score':values,method+'_alarm':alarm})
                rows.append(dict(pool=pool,trip=trip['id'],segment=segment,method=method,seed=seed,
                    **alarm_summary(alarm,c['positions'],c['time_ms'],c['start_ms'])))
            np.savez_compressed(output/'neural'/f'{key}__{seed}.npz',**block)
        block=dict(positions=c['positions'])
        for method in METHODS[2:]:
            p=innovation_profiles[method];values=innovation_score(c['innovations'],p['geometry'],method);alarm=values>p['threshold']
            block.update({method+'_score':values,method+'_alarm':alarm})
            rows.append(dict(pool=pool,trip=trip['id'],segment=segment,method=method,seed=None,
                **alarm_summary(alarm,c['positions'],c['time_ms'],c['start_ms'])))
        np.savez_compressed(output/'innovation'/f'{key}.npz',**block)
    summaries={}
    for pool in ('legacy','expanded'):
        summaries[pool]={}
        complete=not any(r['pool']==pool for r in excluded)
        for method in METHODS:
            group=[r for r in rows if r['pool']==pool and r['method']==method]
            if not group:summaries[pool][method]=dict(status='NO_PREPARED_DATA',descriptive_5_percent_check=False);continue
            s=aggregate(group);s['complete_planned_cohort']=complete
            s['descriptive_5_percent_check']=complete and s['macro_alarm_fraction']<=.05 and s['worst_trip_alarm_fraction']<=.05
            summaries[pool][method]=s
    result=dict(status='NORMAL_DEVELOPMENT_TRANSFER_ONLY',active_policy_changed=False,eligible_for_test=False,
        reserved_validation_accessed=False,stage16_final_routes_accessed=False,quarantined_segments_preprocessed=False,
        training=False,parameters_changed=False,source_files_verified=checked,original_replay=replay,
        prepared_segments=prepared,preprocessing_exclusions=excluded,groups=summaries,rows=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['pool','trip','segment','method','seed','scored_endpoints','alarm_endpoints','presumed_normal_alarm_fraction','longest_alarm_run_epochs']
    with (output/'segment_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    index={p.relative_to(output).as_posix():prior.sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'file_hashes.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(prepared_segments=len(prepared),exclusions=excluded,groups={pool:{method:dict(
        macro=v.get('macro_alarm_fraction'),worst=v.get('worst_trip_alarm_fraction'),check=v['descriptive_5_percent_check'])
        for method,v in group.items()} for pool,group in summaries.items()}),indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage13-zip',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.stage13_zip,args.output)
