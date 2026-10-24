import csv, hashlib, io, json, statistics, zipfile, subprocess
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
out=root/'data/reference/vehicle_native_stage7'
sha=lambda b: hashlib.sha256(b).hexdigest()
archive=out/'linux_study_review.zip'
counts=defaultdict(Counter); groups={}; normal=defaultdict(Counter)
issued=mutated=releases=rows=0
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,h in json.loads(z.read('file_hashes.json')).items(): assert sha(z.read(name))==h,name
    s=json.loads(z.read('summary.json')); f=json.loads(z.read('frozen_calibration.json'))
    cfg=json.loads(z.read('benchmark_config.json')); receipt=json.loads(z.read('freeze_receipt.json'))
    assert receipt['calibration_sha256']==s['calibration_sha256']==sha(z.read('frozen_calibration.json'))
    assert z.read('frozen_calibration.json')==(out/'linux_frozen_calibration.json').read_bytes()
    assert s['source_sha256']==f['source_sha256']
    for name,h in s['source_sha256'].items():
        assert sha((root/name).read_bytes().replace(b'\r\n',b'\n'))==h,name
        assert sha(z.read('frozen_sources/'+name).replace(b'\r\n',b'\n'))==h,name
    assert sha(z.read('benchmark_config.json'))==f['config_sha256']==sha((root/'configs/vehicle_native_stage7.json').read_bytes())
    assert sha(z.read('native_input_manifest.json'))==f['manifest_sha256']==s['input_manifest_sha256']
    chunks=json.loads(z.read('native_chunk_manifests.json')); arrays=0
    for name,m in chunks.items():
        assert m==json.loads((root/name).read_bytes())
        assert sha((root/name).read_bytes())==cfg['chunk_manifest_sha256'][name]
        for item in m['files']:
            p=(root/name).parent/item['path']; assert sha(p.read_bytes())==item['sha256'],str(p)
            assert list(np.load(p,allow_pickle=False).shape)==item['shape']; arrays+=1
    with z.open('requests.csv') as stream:
        for r in csv.DictReader(io.TextIOWrapper(stream)):
            rows+=1; key=(r['trip'],r['case'],r['frame'])
            labels=(r['time_ms'],r['period'],r['perturbed'])
            if key not in groups: groups[key]=(labels,set())
            assert groups[key][0]==labels and r['method'] not in groups[key][1]
            groups[key][1].add(r['method'])
            assert r['nonce_rejected']=='True' and r['expiry_bounded']=='True'
            assert (r['display_mutated']=='True')!=(r['display_accepted']=='True')
            assert (r['final_grant']=='True')==((r['capability_authorized']=='True') and (r['display_accepted']=='True'))
            issued+=r['issued_outcome']=='ALLOW'; mutated+=r['display_mutated']=='True'
            if r['recovery_release']=='True':
                releases+=1; assert r['issued_outcome']=='CHALLENGE' and r['final_grant']=='False'
            c=counts[r['trip'],r['case'],r['method'],r['period']]; c['requests']+=1
            for m,k in [('grants','final_grant'),('motion_alarms','motion_alarm'),('quarantine_requests','quarantined'),('recovery_releases','recovery_release'),('current_detector_alarms','current_detector_alarm')]: c[m]+=r[k]=='True'
            c['blocked']+=r['final_grant']=='False'; c['challenges']+=r['issued_outcome']=='CHALLENGE'
            if r['case']=='normal':
                n=normal[r['trip'],r['method']]; n['requests']+=1;n['grants']+=r['final_grant']=='True'
    assert rows==s['requests']==203566 and sum(s['epochs'].values())==974
    assert len(groups)==18506 and all(g[1]==set(s['methods']) for g in groups.values())
    seen=set()
    with z.open('contexts.jsonl') as stream:
        for line in stream:
            c=json.loads(line); key=(c['trip'],c['case'],str(c['native']['frame']))
            assert key not in seen; seen.add(key)
            labels=groups[key][0]
            assert str(c['observed']['time_ms'])==labels[0] and c['evaluation_only']['period']==labels[1]
            assert c['evaluation_only']['active']==(labels[2]=='True')
            for k in ('time_ms','gnss_time_ms','imu_time_ms','wheel_time_ms','interval_start_ms','maximum_gap_ms'): assert c['native'][k]==c['observed'][k]
    assert seen==set(groups)
    metrics=list(csv.DictReader(io.StringIO(z.read('per_trip_metrics.csv').decode())))
    assert len(metrics)==len(counts)
    for m in metrics:
        c=counts[m['trip'],m['case'],m['method'],m['period']]
        for k,v in c.items(): assert int(m[k])==v,(m,k,v)
        assert abs(float(m['grant_fraction'])-c['grants']/c['requests'])<1e-14
    intervals=list(csv.DictReader(io.StringIO(z.read('paired_trip_intervals.csv').decode())))
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    for i in intervals:
        v=np.array([counts[t,i['case'],'horizon_recovery',i['period']]['grants']/counts[t,i['case'],'horizon_recovery',i['period']]['requests']-counts[t,i['case'],i['reference'],i['period']]['grants']/counts[t,i['case'],i['reference'],i['period']]['requests'] for t in sorted(s['epochs'])])
        assert int(i['trip_clusters'])==len(v)==5
        low,high=np.quantile(v[rng.integers(0,5,(10000,5))].mean(axis=1),[.025,.975])
        for k,a in [('candidate_minus_reference',v.mean()),('bootstrap_lower',low),('bootstrap_upper',high)]: assert abs(float(i[k])-a)<1e-14,(i,k,a)

(root/'artifacts/vehicle_native_stage7').mkdir(parents=True,exist_ok=True)
audit=dict(status='PASS',request_rows=rows,native_epochs=974,paired_groups=len(groups),methods=11,cases=19,trip_clusters=5,metric_rows=len(metrics),paired_interval_rows=len(intervals),native_arrays_verified=arrays,issued_capabilities=issued,altered_cards=mutated,recovery_releases=releases,all_releases_challenge=True,all_denominators_verified=True,all_bootstrap_intervals_recomputed=True,source_data_receipt_hashes_verified=True,development_ancestry_check="See bundled Linux independent audit; portable overlay has no Git history",study_sha256=sha(archive.read_bytes()))
(root/'artifacts/vehicle_native_stage7/independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
methods=['gated_dual','gated_dual_matched','recovery_dual','horizon_dual','horizon_recovery']
lines=['CO-ACT-CPS Stage7: reference evidence (no manuscript)', 'Full-project reference:212 tests; Stage7:23 tests. One expected legacy warning.', '974 native epochs, five held-out routes,19 cases,11 methods;203566 requests.', 'New validation:379 epochs on two routes;79211 requests. No test-driven retuning.', 'Independent audit PASS: all hashes, shared streams, denominators, capability', 'nonce/expiry, exact cards, recovery releases and paired bootstrap intervals.', 'Four clean development routes selectfactor2 on declared grid; current detector', 'alarm fraction0.3764% equal-trip; blocked calibration fraction6.10% includes', 'warmup and quarantine. Original six controls retain Stage6 thresholds.', '', 'Active attack grants, equal-trip percentages (lower is less exposed):', 'case                       old-gate matched-gate recovery  horizon combined']
for case in s['cases']:
    if case=='normal':continue
    vals=[100*statistics.mean(counts[t,case,m,'active']['grants']/counts[t,case,m,'active']['requests'] for t in s['epochs']) for m in methods]
    lines.append(f'{case:26s} '+' '.join(f'{v:9.2f}' for v in vals))
lines+=['','Normal whole-trip grants (all periods aggregated before weighting trips):']
for m in s['methods']:
    ns=[normal[t,m] for t in s['epochs']];g=sum(n['grants'] for n in ns);n=sum(c['requests'] for c in ns)
    lines.append(f'{m:24s} {g}/{n}; equal-trip%={100*statistics.mean(c["grants"]/c["requests"] for c in ns):.2f}')
lines+=['','After withdrawal: honest grants / requests (availability, not detection):']
for case in ('gnss_bias','joint_jump','joint_ramp','coherent_withdrawal','observer_aware','imu_dropout','horizon_budget','gap_reenrollment'):
    for m in methods:
        cs=[counts[t,case,m,'after'] for t in s['epochs']]
        lines.append(f'{case:22s} {m:24s} {sum(c["grants"] for c in cs)}/{sum(c["requests"] for c in cs)}')
lines+=['','Key comparisons:', 'Combined0/323 active coordinated jumps versus matched71/323 and old26/323.', 'Combined observer-aware218/323 accepted; coherent and slow-ramp attacks remain.', 'Combined normal930/974 grants versus matched962/974 and old958/974.', 'After joint-jump withdrawal combined163/321 vs matched57/321 and old238/321.', 'Recovery improves one matched comparison but does not uniformly beat old gate.', '', 'Limits:', 'Same RAV4/highway/logger; five routes do not establish other-vehicle robustness.', 'Native presumed benign data plus controlled perturbations; signers/identity', 'and attestation inputs are synthetic controls, not hardware-attested truth.', 'Coherent forgery, startup bias, horizon-budget adaptation and forged enrollment', 'after IMU reference loss remain counterexamples. Recovery is conditional source', 'consistency. Releases stay CHALLENGE with three later complete epochs required.', 'Bounded wheel receipt ties use last native log update; raw arrays unchanged;', 'receipt order is not proof of physical sample order. GNSS/IMU stay strict.', 'Bootstrap resamples five whole trips, so intervals are exploratory.', 'No human study, physical actuation, learned-AI novelty, deployment certification', 'or journal acceptance claim. Windows Stage7 reproduction remains to be checked.', '', 'Study SHA256:'+audit['study_sha256'], 'Calibration SHA256:'+receipt['calibration_sha256']]
(root/'VEHICLE_STAGE7_RESULTS.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(audit,indent=2))
