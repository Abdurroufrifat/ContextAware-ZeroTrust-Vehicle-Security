"""Normal-only cross-sensor detector comparison on inspected development data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coact_cps.vehicle.innovation29 import innovations,fit_geometry,score,ARMS
from coact_cps.vehicle.temporal23 import summarize
from coact_cps.vehicle.decoder27 import evaluate_gate

SOURCES={'vehicle_transfer_stage25':'73115b3a9d3f2aeeaa1e4bc05350761eec3ef8f1201df5ff8aef9c4be8ab35a6',
         'vehicle_decoder_stage27':'71186700194f4799d1daea9ed1185e018f6c675ad378cbfb2179450eec7ca821'}
REFERENCES=('original','matched_enrollment')
SEEDS=(7,19,43)


def sha(data):return hashlib.sha256(data).hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as obj:return {k:obj[k] for k in obj.files}


def verify_sources(root):
    checked=0
    for folder,pin in SOURCES.items():
        path=root/'artifacts'/folder;raw=(path/'file_hashes.json').read_bytes()
        if sha(raw)!=pin:raise ValueError('fixed source index mismatch: '+folder)
        for name,digest in json.loads(raw).items():
            item=(path/name).resolve()
            if not item.is_relative_to(path.resolve()) or sha(item.read_bytes())!=digest:
                raise ValueError('fixed source evidence mismatch: '+name)
            checked+=1
    for trip in ('rav24_04','rav24_05'):
        if (root/'data/raw/vehicle_data_stage24'/trip).exists():raise ValueError('reserved validation payload present')
    return checked


def values(context):
    result=innovations(context['features'],context['time_ms'],context['start_ms'])[context['positions']]
    if not np.isfinite(result).all():raise ValueError('unavailable history at a fixed evaluation position')
    return result


def metrics(c,alarm):
    p=c['positions'];active=c['labels'][p]
    grant=np.zeros(len(c['labels']),dtype=bool);grant[p]=c['base_grant'][p]&~alarm
    return dict(normal_endpoint_fpr=float(alarm[~active].mean()) if (~active).any() else None,
        active_endpoint_tpr=float(alarm[active].mean()) if active.any() else None,
        normal_endpoint_count=int((~active).sum()),active_endpoint_count=int(active.sum()),
        grant_fraction_all_epochs=float(grant.mean()),
        active_grant_fraction=float(grant[c['labels']].mean()) if c['labels'].any() else None),grant


def comparisons(rows,reference_rows,arm):
    normal=next(r[arm] for r in rows if r['case']=='normal')
    normal_refs={ref:dict(alarm_fraction=float(np.mean([r[ref]['metrics']['normal_endpoint_fpr']
        for r in reference_rows if r['case']=='normal'])),grant_fraction=float(np.mean([
        r[ref]['metrics']['grant_fraction_all_epochs'] for r in reference_rows if r['case']=='normal'])))
        for ref in REFERENCES}
    losses=[];spans=[];misses=[];longer=[]
    for r in rows:
        case=r['case']
        if case=='normal':continue
        after=r[arm]
        group=[b for b in reference_rows if b['case']==case]
        for ref in REFERENCES:
            before=[b[ref] for b in group]
            losses.append(dict(case=case,comparator=ref,mean_tpr_loss=float(np.mean([
                b['metrics']['active_endpoint_tpr'] for b in before]))-after['metrics']['active_endpoint_tpr']))
            spans.append(dict(case=case,comparator=ref,mean_added_observed_span_ms=
                after['temporal']['attack_granted_observed_span_ms']-float(np.mean([
                b['temporal']['attack_granted_observed_span_ms'] for b in before]))))
            for b in group:
                old,new=b[ref]['temporal'],after['temporal']
                for i,(a,z) in enumerate(zip(old['events'],new['events'],strict=True)):
                    if a['first_neural_alarm_delay_ms'] is not None and z['first_neural_alarm_delay_ms'] is None:
                        misses.append(dict(case=case,comparator=ref,seed=b['seed'],event_index=i))
                if new['attack_longest_grant_run_epochs']>old['attack_longest_grant_run_epochs']:
                    longer.append(dict(case=case,comparator=ref,seed=b['seed'],
                        original=old['attack_longest_grant_run_epochs'],candidate=new['attack_longest_grant_run_epochs']))
    worst=max(losses,key=lambda r:r['mean_tpr_loss'])
    fpr=normal['metrics']['normal_endpoint_fpr'];grants=normal['metrics']['grant_fraction_all_epochs']
    gate=evaluate_gate(fpr,normal_refs['matched_enrollment']['alarm_fraction'],
        all(grants>=r['grant_fraction'] for r in normal_refs.values()),worst['mean_tpr_loss'],
        bool(misses),bool(longer),any(r['mean_added_observed_span_ms']>0 for r in spans))
    return dict(normal_alarm_fraction=fpr,normal_grant_fraction=grants,reference_normal=normal_refs,
        worst_case_loss=worst,case_losses=losses,case_span_deltas=spans,additional_missed_events=misses,
        longer_attack_grant_runs=longer,development_screen=gate)


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('output must not already exist')
    checked=verify_sources(root);source25=root/'artifacts/vehicle_transfer_stage25'
    source27=root/'artifacts/vehicle_decoder_stage27'
    fit=values(load(source25/'contexts/rav24_01.npz'))
    cal=values(load(source25/'contexts/rav24_02.npz'))
    geometry=fit_geometry(fit,np.zeros(len(fit),dtype=bool))
    profiles={arm:dict(geometry=geometry,threshold=float(np.quantile(score(cal,geometry,arm),.95))) for arm in ARMS}
    reference=json.loads((source27/'summary.json').read_text())['rows']
    cases=json.loads((source27/'frozen_protocol.json').read_text())['cases']
    code=('src/coact_cps/vehicle/innovation29.py','scripts/run_vehicle_innovation_stage29.py',
          'src/coact_cps/vehicle/temporal23.py','src/coact_cps/vehicle/decoder27.py')
    frozen=dict(schema='vehicle-innovations-development-stage29-v1',source_manifests=SOURCES,
        history=5,max_interval_ms=500,shrinkage=.1,scale_floor=.01,threshold_quantile=.95,
        roles=dict(fit='rav24_01',calibrate='rav24_02',evaluate='rav24_03'),
        fit_positions=len(fit),calibration_positions=len(cal),deterministic_arms=list(ARMS),
        references=list(REFERENCES),reference_seeds=list(SEEDS),cases=cases,
        code_sha256={n:sha((root/n).read_bytes()) for n in code},
        design_sha256=sha((root/'STAGE29_FROZEN_DESIGN.txt').read_bytes()),
        policy_masks_reused=True,neural_execution=False,normal_routes_presumed_benign=True,
        temporal_schema_alarm_field='first_neural_alarm_delay_ms names a scorer alarm in this inherited schema')
    output.mkdir(parents=True);(output/'blocks').mkdir()
    (output/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    np.savez_compressed(output/'fit.npz',innovations=fit,labels=np.zeros(len(fit),dtype=bool))
    np.savez_compressed(output/'calibration.npz',innovations=cal,labels=np.zeros(len(cal),dtype=bool))
    rows=[]
    for case in cases:
        c=load(source27/'contexts'/f'{case}.npz');x=values(c);row=dict(case=case)
        block=dict(positions=c['positions'],innovations=x)
        for arm in ARMS:
            scores=score(x,geometry,arm);alarm=scores>profiles[arm]['threshold']
            m,grant=metrics(c,alarm)
            row[arm]=dict(metrics=m,temporal=summarize(c['time_ms'],c['start_ms'],c['labels'],
                c['positions'],alarm,c['base_grant'],grant))
            block.update({arm+'_score':scores,arm+'_alarm':alarm,arm+'_grant':grant})
        rows.append(row);np.savez_compressed(output/'blocks'/f'{case}.npz',**block)
    result=dict(status='DEVELOPMENT_ONLY_NOT_PROMOTED',eligible_for_test=False,active_policy_changed=False,
        reserved_validation_accessed=False,stage16_final_routes_accessed=False,source_files_verified=checked,
        deterministic_candidate_replicates=1,arms={arm:comparisons(rows,reference,arm) for arm in ARMS},rows=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'case_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['case','arm','normal_endpoint_fpr','active_endpoint_tpr',
            'grant_fraction_all_epochs','attack_longest_grant_run_epochs','attack_granted_observed_span_ms'])
        for r in rows:
            for arm in ARMS:
                m,t=r[arm]['metrics'],r[arm]['temporal']
                writer.writerow([r['case'],arm,m['normal_endpoint_fpr'],m['active_endpoint_tpr'],
                    m['grant_fraction_all_epochs'],t['attack_longest_grant_run_epochs'],t['attack_granted_observed_span_ms']])
    index={p.relative_to(output).as_posix():sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'file_hashes.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({arm:{k:result['arms'][arm][k] for k in ('normal_alarm_fraction','normal_grant_fraction',
        'worst_case_loss','development_screen')} for arm in ARMS},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
