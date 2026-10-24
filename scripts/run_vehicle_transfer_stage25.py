"""Frozen Civic checkpoints on RAV4 development data; no fitting or reservation release."""

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
from vehicle_ai_data13 import features  # noqa: E402
from vehicle_integrity_data import wheel_receipt_view  # noqa: E402
from vehicle_native_data import align_channels,merge_channel  # noqa: E402

from coact_cps.vehicle.ai_models13 import create_adapter,score_nodes  # noqa: E402
from coact_cps.vehicle.native_detectors import ModelConfig  # noqa: E402
from coact_cps.vehicle.transfer25 import normal_metrics,scaled_windows  # noqa: E402

DATA_PROTOCOL='704acd41fd0168215b1e3ab4339b754ea16eb870566aa098234a7be8d2fe9fee'
CONTRACT_SHA='dcf3d4c49f65e8b51e5b76d6d0ca2083969c0b4cdd533a767019b46fc1bdd811'
REPLAY_TOLERANCE=0.001
CONTRACT='data/reference/vehicle_transfer_stage25/model_contract.json'


def native_frames(root,trip):
    if trip['role']!='development':raise ValueError('reserved route access forbidden')
    collected={c:[] for c in ('GNSS/live_gnss_ublox','IMU/accelerometer','CAN/wheel_speed')}
    previous=None;receipt=[]
    for segment in trip['segments']:
        folder=root/'data/raw/vehicle_data_stage24'/trip['id']/f'segment_{segment}'
        for channel in collected:
            t=np.load(folder/channel/'t.npy',allow_pickle=False)
            v=np.load(folder/channel/'value.npy',allow_pickle=False)
            if channel=='CAN/wheel_speed':
                t,v,audit=wheel_receipt_view(t,v);receipt.append(audit)
            collected[channel].append((t,v))
        gt,gv=collected['GNSS/live_gnss_ublox'][-1]
        bounds=(float(gt[0]),float(gt[-1]),float(gv[0,3]),float(gv[-1,3]))
        if previous is not None:
            boot=bounds[0]-previous[1];utc=(bounds[2]-previous[3])/1000
            if not 0<=boot<=0.5 or not 0<=utc<=0.5 or abs(boot-utc)>0.1:
                raise ValueError('native part clock continuity failed')
        previous=bounds
    frames=align_channels(trip['route_id'],*(merge_channel(collected[c]) for c in collected),
                          stride=2,edge_s=2.0)
    return frames,receipt


def run(root,source,output):
    root,output=Path(root).resolve(),Path(output)
    if output.exists():raise ValueError('output must not already exist')
    z=prior.read_source(source);protocol,lock,source_rows,checked=prior.audit_source(z)
    raw=(root/'data/reference/vehicle_data_stage24/protocol.json').read_bytes()
    if prior.sha(raw)!=DATA_PROTOCOL:raise ValueError('Stage24 frozen data protocol mismatch')
    data=json.loads(raw);verify_data(root)
    contract_raw=(root/CONTRACT).read_bytes()
    if (prior.sha(contract_raw)!=CONTRACT_SHA or protocol['source_sha256'][
            'data/reference/vehicle_native_stage10b/stage10b_linux_frozen_transfer.json']!=CONTRACT_SHA):
        raise ValueError('original preprocessing contract mismatch')
    model_config=ModelConfig(**json.loads(contract_raw)['model'])
    cfg=protocol['config'];models={m['seed']:m for m in lock['models'] if m['method']=='coact_physics'}
    prepared={}
    for trip in data['trips']:
        if trip['role']=='development':
            frames,receipt=native_frames(root,trip)
            f=features(frames,model_config);t=np.array([a.time_ms for a in frames],dtype=np.int64)
            s=np.array([a.interval_start_ms for a in frames],dtype=np.int64)
            x,p=scaled_windows(f,t,s,cfg['window'],lock['scaler'])
            prepared[trip['id']]=dict(features=f,time_ms=t,start_ms=s,positions=p,x=x,receipt=receipt)
    if set(prepared)!={'rav24_01','rav24_02','rav24_03'}:raise ValueError('exact development trips required')
    adapters={};replay=[]
    for seed in prior.SEEDS:
        adapter=create_adapter('coact_physics',cfg,root,seed)
        spec=models[seed];raw=z.read(prior.ROOT+f'fit/seed_{seed}/coact_physics.pt')
        if prior.sha(raw)!=spec['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
        adapter.model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
        adapter.model.eval();adapters[seed]=adapter
        row=next(r for r in source_rows if r['trip']=='civic13_09' and r['case']=='normal' and r['seed']==seed)
        c,b=prior.checked_block(z,row,spec)
        x,p=scaled_windows(c['features'],c['time_ms'],c['start_ms'],cfg['window'],lock['scaler'])
        if not np.array_equal(p,c['positions']):raise ValueError('original replay positions mismatch')
        nodes=score_nodes(adapter,x,cfg,seed);error=float(np.max(np.abs(nodes-b['nodes'])))
        if error>REPLAY_TOLERANCE:raise ValueError('baseline replay tolerance exceeded')
        replay.append(dict(seed=seed,maximum_absolute_node_error=error))
    code=['scripts/run_vehicle_transfer_stage25.py','src/coact_cps/vehicle/transfer25.py',
          'scripts/vehicle_native_data.py','scripts/vehicle_integrity_data.py',
          'scripts/vehicle_ai_data13.py','src/coact_cps/vehicle/ai_models13.py',
          'src/coact_cps/models/temporal_graph.py']
    frozen=dict(schema='vehicle-frozen-transfer-stage25-v1',stage24_protocol_sha256=DATA_PROTOCOL,
                original_source_sha256=prior.SOURCE_SHA256,model_contract_sha256=CONTRACT_SHA,
                design_sha256=prior.sha((root/'STAGE25_FROZEN_DESIGN.txt').read_bytes()),
                code_sha256={n:prior.sha((root/n).read_bytes()) for n in code},
                models=[models[s] for s in prior.SEEDS],scaler=lock['scaler'],
                replay_tolerance=REPLAY_TOLERANCE,alarm_screen=0.05,
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                training=False,threshold_changed=False,reserved_validation_accessed=False,
                stage16_final_routes_accessed=False,eligible_for_test=False)
    output.mkdir(parents=True);(output/'blocks').mkdir();(output/'contexts').mkdir()
    (output/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[]
    for trip,context in prepared.items():
        np.savez_compressed(output/'contexts'/f'{trip}.npz',**{
            k:v for k,v in context.items() if k not in ('x','receipt')})
        for seed,adapter in adapters.items():
            spec=models[seed];nodes=score_nodes(adapter,context['x'],cfg,seed)
            metrics,score,alarm=normal_metrics(nodes,spec['node_calibration'],spec['threshold'],
                context['positions'],context['time_ms'],context['start_ms'])
            np.savez_compressed(output/'blocks'/f'{trip}__{seed}.npz',positions=context['positions'],
                                nodes=nodes,score=score,alarm=alarm)
            rows.append(dict(trip=trip,seed=seed,**metrics))
    trip_means={t:float(np.mean([r['presumed_normal_alarm_fraction'] for r in rows if r['trip']==t]))
                for t in prepared}
    mean=float(np.mean(list(trip_means.values())));worst=max(trip_means.values())
    result=dict(status='DEVELOPMENT_TRANSFER_ONLY_NOT_PROMOTED',policy_changed=False,
                eligible_for_test=False,reserved_validation_accessed=False,stage16_final_routes_accessed=False,
                model_training=False,threshold_changed=False,source_files_verified=checked,
                baseline_replay=replay,mean_alarm_fraction=mean,worst_trip_mean_alarm_fraction=worst,
                descriptive_5_percent_screen_passed=mean<=0.05 and worst<=0.05,
                trip_means=trip_means,rows=rows,receipt_audits={t:c['receipt'] for t,c in prepared.items()})
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    index={p.relative_to(output).as_posix():prior.sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'file_hashes.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','receipt_audits')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage13-zip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.stage13_zip,args.output)
