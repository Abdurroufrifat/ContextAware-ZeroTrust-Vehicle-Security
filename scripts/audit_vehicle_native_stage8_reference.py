import csv,hashlib,io,json,statistics,subprocess,zipfile
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1];out=root/'data/reference/vehicle_native_stage8'
archive=out/'linux_study_review.zip';sha=lambda b:hashlib.sha256(b).hexdigest()
counts=defaultdict(Counter);normal=defaultdict(Counter);normal_flags={};paired_exposure=defaultdict(Counter);groups={};issued=mutated=releases=lost_requests=unready_grants=0
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    hashes=json.loads(z.read('file_hashes.json'));assert set(hashes)==set(z.namelist())-{'file_hashes.json'}
    for n,h in hashes.items():assert sha(z.read(n))==h,n
    s=json.loads(z.read('summary.json'));cfg=json.loads(z.read('benchmark_config.json'));f=json.loads(z.read('frozen_calibration.json'))
    receipt=json.loads(z.read('freeze_receipt.json'))
    assert receipt['calibration_sha256']==s['calibration_sha256']==sha(z.read('frozen_calibration.json'))
    assert z.read('frozen_calibration.json')==(out/'linux_frozen_calibration.json').read_bytes()
    assert s['source_sha256']==f['source_sha256']
    for n,h in s['source_sha256'].items():
        assert sha((root/n).read_bytes().replace(b'\r\n',b'\n'))==h,n
        assert sha(z.read('frozen_sources/'+n).replace(b'\r\n',b'\n'))==h,n
    assert sha(z.read('benchmark_config.json'))==f['config_sha256']
    assert sha(z.read('native_input_manifest.json'))==f['manifest_sha256']==s['input_manifest_sha256']
    native=json.loads(z.read('native_input_manifest.json'));arrays=0
    for n,m in json.loads(z.read('native_chunk_manifests.json')).items():
        p=root/n;assert json.loads(p.read_bytes())==m and sha(p.read_bytes())==cfg['chunk_manifest_sha256'][n]
        for item in m['files']:
            path=p.parent/item['path'];assert sha(path.read_bytes())==item['sha256']
            assert list(np.load(path,allow_pickle=False).shape)==item['shape'];arrays+=1
    with z.open('requests.csv') as stream:
        for total,r in enumerate(csv.DictReader(io.TextIOWrapper(stream)),1):
            key=(r['trip'],r['case'],r['frame']);label=(r['time_ms'],r['period'],r['perturbed'])
            if key not in groups:groups[key]=(label,set())
            assert groups[key][0]==label and r['method'] not in groups[key][1];groups[key][1].add(r['method'])
            assert r['nonce_rejected']=='True' and r['expiry_bounded']=='True'
            assert (r['display_mutated']=='True')!=(r['display_accepted']=='True')
            assert (r['final_grant']=='True')==((r['capability_authorized']=='True') and (r['display_accepted']=='True'))
            issued+=r['issued_outcome']=='ALLOW';mutated+=r['display_mutated']=='True'
            if r['recovery_release']=='True':
                releases+=1;assert r['issued_outcome']=='CHALLENGE' and r['final_grant']=='False'
            if r['anchor_lost']=='True':
                lost_requests+=1;assert r['issued_outcome']=='DENY' and r['final_grant']=='False'
            if r['method'].startswith('anchor') and r['anchor_ready']=='False' and r['final_grant']=='True':unready_grants+=1
            c=counts[r['trip'],r['case'],r['method'],r['period']];c['requests']+=1
            for field,k in [('grants','final_grant'),('motion_alarms','motion_alarm'),('quarantine_requests','quarantined'),('recovery_releases','recovery_release'),('current_detector_alarms','current_detector_alarm')]:c[field]+=r[k]=='True'
            c['blocked']+=r['final_grant']=='False';c['challenges']+=r['issued_outcome']=='CHALLENGE'
            if r['case']=='normal':normal_flags[r['trip'],r['frame'],r['method']]=r['final_grant']=='True'
            elif r['perturbed']=='True':
                clean=normal_flags[r['trip'],r['frame'],r['method']];grant=r['final_grant']=='True'
                paired_exposure[r['case'],r['method']]['active_requests']+=1
                paired_exposure[r['case'],r['method']]['attack_only_denials']+=clean and not grant
                paired_exposure[r['case'],r['method']]['also_denied_normally']+=not clean and not grant
                paired_exposure[r['case'],r['method']]['attack_enabled_grants']+=not clean and grant
            if r['case']=='normal':normal[s['route_by_episode'][r['trip']],r['method']]['requests']+=1;normal[s['route_by_episode'][r['trip']],r['method']]['grants']+=r['final_grant']=='True'
    assert total==s['requests']==218820 and sum(s['epochs'].values())==1042
    assert len(groups)==21882 and all(g[1]==set(s['methods']) for g in groups.values())
    seen=set()
    with z.open('contexts.jsonl') as stream:
        for line in stream:
            c=json.loads(line);key=(c['trip'],c['case'],str(c['native']['frame']))
            assert key not in seen;seen.add(key);label=groups[key][0]
            assert str(c['observed']['time_ms'])==label[0] and c['evaluation_only']['period']==label[1]
            assert c['evaluation_only']['active']==(label[2]=='True')
            for k in ('time_ms','interval_start_ms','gnss_time_ms','imu_time_ms','wheel_time_ms','maximum_gap_ms'):assert c['native'][k]==c['observed'][k]
    assert seen==set(groups)
    metrics=list(csv.DictReader(io.StringIO(z.read('per_trip_metrics.csv').decode())));assert len(metrics)==len(counts)
    routecounts=defaultdict(Counter);episode_sets=defaultdict(set)
    for m in metrics:
        key=(m['trip'],m['case'],m['method'],m['period']);c=counts[key]
        for k,v in c.items():assert int(m[k])==v
        assert abs(float(m['grant_fraction'])-c['grants']/c['requests'])<1e-14
        key2=(s['route_by_episode'][m['trip']],m['case'],m['method'],m['period']);episode_sets[key2].add(m['trip']);routecounts[key2].update(c)
    routes=list(csv.DictReader(io.StringIO(z.read('per_route_metrics.csv').decode())));assert len(routes)==len(routecounts)
    for r in routes:
        key=(r['route_id'],r['case'],r['method'],r['period']);c=routecounts[key]
        for k,v in c.items():assert int(r[k])==v
        assert int(r['episodes'])==len(episode_sets[key])==3
        assert abs(float(r['grant_fraction'])-c['grants']/c['requests'])<1e-14
    route_ids=sorted(set(s['route_by_episode'].values()));assert len(route_ids)==s['route_clusters']==2
    intervals=list(csv.DictReader(io.StringIO(z.read('paired_trip_intervals.csv').decode())));rng=np.random.default_rng(cfg['bootstrap_seed'])
    for i in intervals:
        v=np.array([routecounts[t,i['case'],'anchor_recovery',i['period']]['grants']/routecounts[t,i['case'],'anchor_recovery',i['period']]['requests']-routecounts[t,i['case'],i['reference'],i['period']]['grants']/routecounts[t,i['case'],i['reference'],i['period']]['requests'] for t in route_ids])
        assert int(i['route_clusters'])==2
        low,high=np.quantile(v[rng.integers(0,2,(10000,2))].mean(axis=1),[.025,.975])
        for k,vv in [('candidate_minus_reference',v.mean()),('bootstrap_lower',low),('bootstrap_upper',high)]:assert abs(float(i[k])-vv)<1e-14
old=json.loads((root/'data/reference/vehicle_native_stage7/linux_frozen_calibration.json').read_bytes())
assert f['model']==old['model'] and f['thresholds']==old['thresholds'] and f['session_config']['stage7_horizon']==old['horizon']
assert all(s['source_sha256'][n]==h for n,h in old['source_sha256'].items())
(root/'artifacts/vehicle_native_stage8').mkdir(parents=True,exist_ok=True)
audit=dict(status='PASS',candidate_status='EXPERIMENTAL_NOT_PROMOTED',requests=total,native_epochs=1042,episodes=6,route_clusters=2,cases=21,methods=10,paired_contexts=len(groups),episode_metric_rows=len(metrics),route_metric_rows=len(routes),paired_route_interval_rows=len(intervals),raw_arrays_verified=arrays,source_snapshots_verified=len(s['source_sha256']),archive_content_hashes_verified=len(hashes),issued_capabilities=issued,altered_cards=mutated,recovery_releases=releases,all_releases_challenge=True,reference_lost_requests=lost_requests,all_lost_reference_requests_denied=True,grants_while_anchor_unready=unready_grants,all_denominators_verified=True,all_route_intervals_recomputed=True,stage6_stage7_code_and_parameters_unchanged=True,study_sha256=sha(archive.read_bytes()),calibration_sha256=s['calibration_sha256'])
audit['active_denial_comparison_with_same_normal_epochs']={case:{method:dict(paired_exposure[case,method]) for method in ['horizon_recovery','anchor_dual','anchor_recovery']} for case in s['cases'] if case!='normal'}
(root/'artifacts/vehicle_native_stage8/reference_independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
methods=['gated_dual','gated_dual_matched','horizon_recovery','anchor_dual','anchor_recovery']
lines=['CO-ACT-CPS Stage8 results (project evidence; no manuscript)','', '234 full-project tests and22 new tests passed; one existing legacy warning.', 'Independent review completed; one false denial reason fixed with RED/GREEN.',
 'Candidate status: EXPERIMENTAL, NOT PROMOTED over the Stage7 combined method.',
 'Stage8combined normal909/1042 versus Stage7combined967/1042; observer-aware',
 'grants231/345 versus Stage7combined216/345. The intended joint improvement',
 'in attack exposure and normal availability did not hold on the unused routes.',
 'Gap-reenrollment grants0/345 versus Stage7combined230/345, at the explicit',
 'cost of0/670honest post-dropout grants versus Stage7combined601/670.', '1042 native epochs,6separate recordings on2test routes,21cases,10methods.', '218820 requests;21882 shared contexts. Validation140epochs/29400requests.', 'Source/config/data/calibration frozen before heldout outcomes; no retuning.', 'Audit PASS: all hashes, receipt, paired streams, capabilities/cards, per-episode', 'and per-route denominators, and every paired whole-route interval verified.', '', 'Inference:2route clusters, not6independent trips or218820independent samples.', 'Pool all episode counts within each route before giving routes equal weight.', 'Intervals are exploratory and cannot establish population generalization.', '', 'Active-window grants, equal-route percentages (lower means less exposure):', 'case                       old-gate matched-gate Stage7-comb anchor-gate anchor-comb']
for case in s['cases']:
    if case=='normal':continue
    vals=[100*statistics.mean(routecounts[t,case,m,'active']['grants']/routecounts[t,case,m,'active']['requests'] for t in route_ids) for m in methods]
    lines.append(f'{case:26s} '+' '.join(f'{v:11.2f}' for v in vals))
lines+=['','Pooled active-window grants / requests:']
for case in ('joint_jump','joint_ramp','coherent_hold','observer_aware','horizon_budget','gap_reenrollment','anchor_budget','enrollment_ramp'):
    for m in methods:
        cs=[routecounts[t,case,m,'active'] for t in route_ids];lines.append(f'{case:20s} {m:22s} {sum(c["grants"] for c in cs)}/{sum(c["requests"] for c in cs)}')
lines+=['','Unperturbed normal requests, whole-route pooling of all periods:']
for m in s['methods']:
    cs=[normal[t,m] for t in route_ids];lines.append(f'{m:22s} {sum(c["grants"] for c in cs)}/{sum(c["requests"] for c in cs)} grants; equal-route%={100*statistics.mean(c["grants"]/c["requests"] for c in cs):.2f}')
lines+=['','Honest after-withdrawal grants / requests (availability cost):']
for case in ('joint_jump','joint_ramp','coherent_withdrawal','observer_aware','imu_dropout','horizon_budget','gap_reenrollment','anchor_budget'):
    for m in methods:
        cs=[routecounts[t,case,m,'after'] for t in route_ids];lines.append(f'{case:20s} {m:22s} {sum(c["grants"] for c in cs)}/{sum(c["requests"] for c in cs)}')
lines+=['','Attack-only denials vs the same unperturbed epochs (candidate):']
for case in ('joint_ramp','coherent_hold','startup_common_bias','observer_aware','horizon_budget','gap_reenrollment','anchor_budget','enrollment_ramp'):
    c=paired_exposure[case,'anchor_recovery'];lines.append(f'{case:20s} {dict(c)}')
lines+=['','Persistent startup/enrollment controls must not credit ordinary normal blocks',
'as attack detection. The paired comparison above separates this contribution.',
'', 'Enrollment and continuity limitations:',f'{unready_grants} anchor-method grants occurred before the extra reference was ready.', 'These use existing3epoch native checks and cannot be called anchored assurance.', f'{lost_requests} requests had a lost session reference; all denied.', 'Reference loss intentionally blocks honest post-IMU-gap traffic for that session.', 'Recovery never changes enrolled slope/intercept; initial enrollment can be', 'poisoned. Coherent all-source motion and biased startup remain counterexamples.', 'Scores/biases are empirical consistency measures, not physical attestation.', 'Identity/signers are synthetic inputs. Same RAV4/highway/logger. No human study,', 'actuation safety or learned-AI novelty. Read all controls and costs; no uniform', 'winner or journal acceptance is claimed. Windows reproduction is pending.', '', 'Study SHA256:'+audit['study_sha256'],'Calibration SHA256:'+audit['calibration_sha256'],'','Independent audit record:',json.dumps(audit,indent=2)]
(root/'VEHICLE_STAGE8_RESULTS.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(audit,indent=2))
print('\n'.join(lines[13:36]))
