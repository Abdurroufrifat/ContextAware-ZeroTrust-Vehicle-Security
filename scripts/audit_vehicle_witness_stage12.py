"""Reconstruct saved synthetic metrics and check witness evidence without policies."""
import argparse
import hashlib
import hmac
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import canonical_json, digest
from vehicle_witness_simulation12 import changed_stream, period, reference, trajectory

DOMAIN = b'COACT-CPS-WITNESS-12\x00'


def sha(b):return hashlib.sha256(b).hexdigest()


def check_frames(wires,r,c,use=False):
    assert type(wires) is dict and set(wires)=={'identity','gnss','wheel','imu'}
    f=c['native'];now=r['time_ms']
    revoked=r['period']=='active' and (r['case']=='identity_revoked' or (use and r['case']=='postissue_revoked'))
    signals=dict(identity=dict(subject_id='native-reader',credential_valid=True,revoked=revoked,attestation_ok=True),
        gnss=dict(speed_mps=f['gnss_speed']),wheel=dict(speed_mps=f['wheel_speed']),
        imu=dict(acceleration_mps2=f['imu_accel'],interval_start_ms=f['interval_start_ms'],
                 interval_end_ms=f['time_ms'],maximum_gap_ms=f['maximum_gap_ms']))
    for source,wire in wires.items():
        p=json.loads(wire);b=p['body'];assert set(p)=={'body','mac'}
        key=hashlib.sha256(('Stage12 simulation collector '+source).encode()).digest()
        mac=hmac.new(key,b'COACT-VEHICLE-SOURCE-V1\x00'+canonical_json(b).encode(),hashlib.sha256).hexdigest()
        assert hmac.compare_digest(mac,p['mac'])
        timestamp=now if source=='identity' else f[source+'_time_ms']
        assert b==dict(source=source,vehicle_id='native-ego',frame_id=f"epoch-{r['frame']}",
            timestamp_ms=timestamp,sequence=3*r['frame']+(2 if use else 1),signals=signals[source])


def check_capability(r):
    packet=json.loads(r['capability']);b=packet['body']
    assert set(packet)=={'body','mac'}
    mac=hmac.new(b'CAPABILITY SIMULATION KEY STAGE12! ',b'COACT-VEHICLE-CAPABILITY-V1\x00'+canonical_json(b).encode(),hashlib.sha256).hexdigest()
    assert hmac.compare_digest(mac,packet['mac'])
    assert set(b)=={'request','mode','context_digest','policy_digest','issued_ms','expires_ms'}
    assert b['request']==r['request'] and b['policy_digest']==r['policy_digest'] and b['mode']=='stream'
    assert b['issued_ms']==r['time_ms']<=r['use_time_ms']<b['expires_ms']==r['time_ms']+250


def valid_certificate(r,context,public_key,cfg):
    """Separate signature/binding/error calculation; does not call WitnessVerifier."""
    try:
        wire=r['certificate']
        if type(wire) is not str or len(wire)>8192:return False,None
        packet=json.loads(wire);b=packet['body']
        assert set(packet)=={'body','signature'}
        required={'schema','issuer','challenge','vehicle_id','request_digest','policy_digest',
            'epoch','issued_ms','expires_ms','sample_ms','speed_mps','uncertainty_mps','sequence'}
        assert set(b)==required and b['schema']=='coact-speed-witness-v1' and b['issuer']=='roadside-test'
        assert type(packet['signature']) is str and len(packet['signature'])==128
        public_key.verify(bytes.fromhex(packet['signature']),DOMAIN+canonical_json(b).encode())
        for k,v in r['challenge'].items():assert type(b[k]) is type(v) and b[k]==v
        assert b['request_digest']==digest(r['request']) and b['policy_digest']==r['policy_digest']
        assert b['epoch']==r['epoch_before'] and b['vehicle_id']==r['request']['vehicle_id']
        assert type(b['challenge']) is str and len(bytes.fromhex(b['challenge']))==32
        assert b['expires_ms']-b['issued_ms']==cfg['challenge_lifetime_ms']
        for k in ['sample_ms','sequence','epoch','issued_ms','expires_ms']:
            assert type(b[k]) is int and 0<=b[k]<=2**53-1
        now=r['time_ms'];motion=context['native']['time_ms']
        assert b['issued_ms']<=b['sample_ms']<=motion<=now<=b['expires_ms']
        age=now-b['sample_ms']
        assert age<=cfg['maximum_age_ms']
        assert type(b['speed_mps']) in (int,float) and math.isfinite(b['speed_mps']) and 0<=b['speed_mps']<=100
        assert type(b['uncertainty_mps']) in (int,float) and math.isfinite(b['uncertainty_mps']) and 0<=b['uncertainty_mps']<=cfg['maximum_uncertainty_mps']
        uncertainty=cfg['maximum_uncertainty_mps']+cfg['maximum_acceleration_mps2']*age/1000
        bound=cfg['agreement_mps']+uncertainty
        model=context['model']
        assert max(abs(context['native']['gnss_speed']-model.gnss_offset-b['speed_mps']),
                   abs(context['native']['wheel_speed']-b['speed_mps']))<=bound
        return True,dict(sample=b['sample_ms'],sequence=b['sequence'],id=sha(canonical_json(packet).encode()),
                         physical_bound=cfg['agreement_mps']+2*uncertainty)
    except (AssertionError,KeyError,ValueError,TypeError,InvalidSignature):return False,None


def audit(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    hashes=json.loads((out/'file_hashes.json').read_bytes())
    assert set(hashes)=={'frozen_protocol.json','contexts.jsonl','requests.jsonl','summary.json'}
    for n,h in hashes.items():assert sha((out/n).read_bytes())==h,n
    f=json.loads((out/'frozen_protocol.json').read_bytes());cfg=f['config']
    assert f['simulation_only'] is True
    for n,h in f['source_sha256'].items():assert sha((root/n).read_bytes().replace(b'\r\n',b'\n'))==h,n
    s=json.loads((out/'summary.json').read_bytes())
    assert s['config']==cfg and s['freeze_sha256']==sha((out/'frozen_protocol.json').read_bytes())
    assert s['status']=='SIMULATION_ONLY_NOT_PROMOTED'
    old=reference(root);m=ModelConfig(**old['model'])
    expected={}
    for seed in cfg['seeds']:
        original=trajectory(seed,cfg['frames'],m)
        for case in cfg['cases']:
            frames=changed_stream(original,case,m,old['session_config'])
            for current,(base,truth,noise) in zip(frames,original,strict=True):
                expected[seed,case,current.frame]=dict(seed=seed,case=case,frame=current.frame,
                    native=asdict(current),truth_speed_mps=truth,witness_noise_mps=noise,
                    period=period(current.frame,len(frames),case))
    contexts={}
    for line in (out/'contexts.jsonl').open():
        c=json.loads(line);key=c['seed'],c['case'],c['frame']
        assert key not in contexts and c==expected[key]
        contexts[key]=dict(c,model=m)
    assert contexts.keys()==expected.keys()
    pub=Ed25519PublicKey.from_public_bytes(bytes.fromhex(f['witness_public_key_hex']))
    counts=defaultdict(lambda:dict(requests=0,grants=0,physically_inaccurate_grants=0,releases=0))
    seen=defaultdict(set);watermarks={};release_at={};certificates=0;releases=0
    after_begin={};first_grant={};last_time={};confirmation={};epochs={}
    quorums={};warmup_started={};witness_enrolled=set();latches={}
    for total,line in enumerate((out/'requests.jsonl').open(),1):
        r=json.loads(line);key=r['seed'],r['case'],r['frame'];c=contexts[key]
        method=r['method'];assert method in cfg['methods'] and method not in seen[key]
        seen[key].add(method)
        group=r['seed'],r['case'],method
        assert r['request']==dict(request_id=f"s{r['seed']}-{r['case']}-{r['frame']}",
            nonce=f"n{r['seed']}-{r['case']}-{r['frame']}",subject_id='native-reader',vehicle_id='native-ego',
            resource='vehicle.telemetry.longitudinal',operation='read')
        check_frames(r['issue_frames'],r,c)
        if r['issued_outcome']=='ALLOW':
            assert r['use_frames'] is not None and r['use_time_ms']==r['time_ms']
            check_frames(r['use_frames'],r,c,True);check_capability(r)
        else:assert r['use_frames'] is None and r['capability'] is None
        assert r['time_ms']==c['native']['time_ms'] and r['context_digest']==digest(c['native'])
        assert r['time_ms']>last_time.get(group,-1);last_time[group]=r['time_ms']
        assert r['period']==c['period']
        error=max(abs(c['native']['gnss_speed']-m.gnss_offset-c['truth_speed_mps']),abs(c['native']['wheel_speed']-c['truth_speed_mps']))
        assert abs(error-r['physical_error_mps'])<1e-12
        if r['granted']:
            assert r['issued_outcome']=='ALLOW' and r['use_outcome']=='ALLOW'
            assert r['issued_quorum']>=3 and r['consumed_quorum']>=3
        if r['case'] in {'identity_revoked','postissue_revoked'} and r['period']=='active':assert not r['granted']
        diag=r['diagnostics'];released=diag.get('witness_release',diag.get('released',False))
        # Distinct motion epochs increment quorum; identity denial, release and
        # candidate quarantine reset it. Saved integer quorum claims are not proof.
        identity_denied=r['case']=='identity_revoked' and r['period']=='active'
        expected_quorum=0 if identity_denied else min(3,quorums.get(group,0)+1)
        if released or (method=='witness_recovery' and diag['quarantined']):expected_quorum=0
        assert r['issued_quorum']==expected_quorum,'issued_quorum_reconstruction'
        consumed_quorum=0 if r['issued_outcome']=='ALLOW' and r['case']=='postissue_revoked' and r['period']=='active' else expected_quorum
        assert r['consumed_quorum']==consumed_quorum
        quorums[group]=consumed_quorum
        if released:
            assert r['issued_outcome']=='CHALLENGE' and not r['granted'] and r['issued_quorum']==0
            release_at[group]=r['frame'];releases+=1
        if group in release_at and r['granted']:assert r['frame']-release_at[group]>=3
        if method=='witness_recovery':
            assert r['quarantine_before']==latches.get(group,False)
            assert diag['motion_quorum']==expected_quorum
            if diag['witness_check']['reason']!='witness_not_checked':
                warmup_started.setdefault(group,r['time_ms'])
            if r['issued_outcome']=='ALLOW' and group not in witness_enrolled:
                assert r['time_ms']-warmup_started[group]>=old['session_config']['stage7_horizon']['horizons'][0],'initial_horizon_warmup'
            if released:witness_enrolled.add(group)
            assert r['epoch_before']==epochs.get(group,0)
            count,begin,prior_time,prior_id=confirmation.get(group,(0,-1,-1,''))
            valid,proof=valid_certificate(r,c,pub,cfg['witness'])
            if diag['witness_check']['accepted']:
                assert valid and proof['id']==diag['witness_check']['certificate_id']
                assert abs(proof['physical_bound']-diag['witness_check']['physical_bound'])<1e-12
                prior=watermarks.get(group)
                if prior and proof['id']!=prior['id']:
                    assert proof['sequence']>prior['sequence'] and proof['sample']>prior['sample']
                watermarks[group]=proof;certificates+=1
            if diag['witness_check']['accepted'] and r['quarantine_before']:
                if count and c['native']['interval_start_ms']!=prior_time:
                    count,begin=0,-1
                if not count:begin=r['time_ms']
                if proof['id']!=prior_id:count+=1
                prior_id=proof['id']
            elif not diag['witness_check']['accepted']:
                count,begin,prior_id=0,-1,''
            if r['issued_outcome']=='ALLOW' or released:
                assert valid and diag['witness_check']['accepted']
            if r['granted']:
                assert r['use_diagnostics']['witness_check']['accepted']
                use_row=dict(r,certificate=r['use_certificate'],challenge=r['use_challenge'],time_ms=r['use_time_ms'])
                use_valid,use_proof=valid_certificate(use_row,c,pub,cfg['witness'])
                assert use_valid and use_proof['id']==proof['id']
                if r['case']!='witness_compromised':assert error<=proof['physical_bound']+1e-12
            if released:
                assert r['quarantine_before'] and diag['witness_epoch']==r['epoch_before']+1
                assert count>=5 and r['time_ms']-begin>=1000
                count,begin,prior_id=0,-1,''
            assert diag['witness_confirmation_epochs']==count
            expected_epoch=r['epoch_before']+int(released or (not r['quarantine_before'] and diag['quarantined']))
            assert diag['witness_epoch']==expected_epoch
            use_diag=r['use_diagnostics']
            if use_diag:
                use_entered=not diag['quarantined'] and use_diag['quarantined']
                assert use_diag['witness_epoch']==expected_epoch+int(use_entered)
                expected_epoch=use_diag['witness_epoch']
                if not use_diag['witness_check']['accepted']:count,begin,prior_id=0,-1,''
            confirmation[group]=count,begin,r['time_ms'],prior_id
            epochs[group]=expected_epoch
            latches[group]=(use_diag if use_diag else diag)['quarantined']
            if r['case'] in {'witness_missing','witness_replay','witness_forged'} and r['period']=='active':
                assert not r['granted'] and not released
        counter=counts[r['seed'],r['case'],method,r['period']]
        counter['requests']+=1;counter['grants']+=r['granted']
        counter['physically_inaccurate_grants']+=r['granted'] and error>1
        counter['releases']+=released
        if r['period']=='after':
            after_begin.setdefault(group,r['time_ms'])
            if r['granted']:first_grant.setdefault(group,r['time_ms'])
    assert all(v==set(cfg['methods']) for v in seen.values()) and seen.keys()==contexts.keys()
    calculated=[dict(seed=k[0],case=k[1],method=k[2],period=k[3],**v) for k,v in sorted(counts.items())]
    assert calculated==s['metrics'] and total==s['requests']==len(contexts)*len(cfg['methods'])
    result=dict(status='PASS',evidence='SYNTHETIC_ONLY',requests=total,contexts=len(contexts),
        frozen_sources=len(f['source_sha256']),content_hashes=len(hashes),
        witness_certificate_checks=certificates,releases_checked=releases,
        policy_reexecution=False,grant_time_signature_and_physical_bounds_verified=True,
        issue_and_use_source_authentication_verified=True,use_certificate_rechecked_independently=True,
        quorum_and_latch_continuity_reconstructed=True,
        all_metrics_reconstructed=True,candidate_status='SIMULATION_ONLY_NOT_PROMOTED',
        first_honest_after_grant_delay_ms=[dict(seed=k[0],case=k[1],method=k[2],
            delay_ms=first_grant[k]-v if k in first_grant else None) for k,v in sorted(after_begin.items())],
        summary_sha256=sha((out/'summary.json').read_bytes()))
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.root,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='first_honest_after_grant_delay_ms'},indent=2))
