"""Freeze and run an explicitly synthetic witness comparison once per output path."""
import argparse
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, canonical_json, digest
from coact_cps.vehicle.protected_recovery_policy import ProtectedRecoveryPolicyEngine
from coact_cps.vehicle.window_recovery_policy import WindowRecoveryPolicyEngine
from coact_cps.vehicle.witness_certificate import DOMAIN, WitnessConfig
from coact_cps.vehicle.witness_recovery_policy import WitnessRecoveryPolicyEngine
from run_vehicle_native_stage6 import packets
from run_vehicle_native_stage11 import SOURCE_FILES as OLD_SOURCES
from vehicle_witness_simulation12 import changed_stream, period, reference, trajectory

NEW_SOURCES = ['src/coact_cps/vehicle/witness_certificate.py',
    'src/coact_cps/vehicle/witness_recovery_policy.py',
    'scripts/vehicle_witness_simulation12.py','scripts/run_vehicle_witness_stage12.py',
    'scripts/audit_vehicle_witness_stage12.py','configs/vehicle_witness_stage12.json',
    'requirements_vehicle_witness_stage12.txt',
    'data/reference/vehicle_native_stage10b/stage10b_linux_frozen_transfer.json']
SOURCE_FILES = OLD_SOURCES+NEW_SOURCES
PRIVATE_SEED = hashlib.sha256(b'COACT Stage12 PUBLIC SIMULATION KEY ONLY').digest()
PUBLIC_KEY = Ed25519PrivateKey.from_private_bytes(PRIVATE_SEED).public_key().public_bytes_raw()


def sha(b):return hashlib.sha256(b).hexdigest()


def freeze(root,cfg,files=None):
    files = SOURCE_FILES if files is None else files
    return dict(schema='coact-witness-freeze-v1',config=cfg,
        source_sha256={n:sha((root/n).read_bytes().replace(b'\r\n',b'\n')) for n in files},
        witness_public_key_hex=PUBLIC_KEY.hex(),simulation_only=True)


def verify_freeze(root,f):
    for n,h in f['source_sha256'].items():
        assert sha((root/n).read_bytes().replace(b'\r\n',b'\n'))==h,n


def engine(method,cfg,f):
    keys={s:hashlib.sha256(('Stage12 simulation collector '+s).encode()).digest() for s in SOURCES}
    args=(keys,b'CAPABILITY SIMULATION KEY STAGE12! ',
        {('native-reader','native-ego','vehicle.telemetry.longitudinal','read')},
        ModelConfig(**f['model']),f['session_config']['civic_thresholds'])
    horizon=f['session_config']['stage7_horizon']
    if method=='civic_horizon':e=IntegrityPolicyEngine(*args,'horizon_dual',horizon)
    elif method=='civic_recovery':e=WindowRecoveryPolicyEngine(*args,horizon)
    elif method=='protected_recovery':e=ProtectedRecoveryPolicyEngine(*args,horizon,f['session_config']['anchor'])
    elif method=='witness_recovery':e=WitnessRecoveryPolicyEngine(*args,horizon,{'roadside-test':PUBLIC_KEY},WitnessConfig(**cfg['witness']))
    else:raise ValueError('unknown_stage12_method')
    return e,keys


def certificate(challenge,speed,now,sequence,forged=False):
    # This signer sees simulated truth; the policy sees only the signed report.
    body=dict(challenge,schema='coact-speed-witness-v1',issuer='roadside-test',
              speed_mps=speed,uncertainty_mps=.1,sample_ms=now,sequence=sequence)
    key=Ed25519PrivateKey.from_private_bytes(bytes(32) if forged else PRIVATE_SEED)
    return canonical_json(dict(body=body,signature=key.sign(DOMAIN+canonical_json(body).encode()).hex()))


def metric_summary(rows):
    return [dict(seed=k[0],case=k[1],method=k[2],period=k[3],**v) for k,v in sorted(rows.items())]


def run(root,out,cfg=None):
    root,out=Path(root).resolve(),Path(out).resolve()
    cfg=cfg or json.loads((root/'configs/vehicle_witness_stage12.json').read_bytes())
    out.mkdir(parents=True,exist_ok=False)
    f=freeze(root,cfg)
    (out/'frozen_protocol.json').write_text(json.dumps(f,indent=2)+'\n')
    verify_freeze(root,f)
    old=reference(root);model=ModelConfig(**old['model'])
    summary=defaultdict(lambda:dict(requests=0,grants=0,physically_inaccurate_grants=0,releases=0))
    requests=0
    with (out/'contexts.jsonl').open('w') as contexts,(out/'requests.jsonl').open('w') as output:
        for seed in cfg['seeds']:
            original=trajectory(seed,cfg['frames'],model)
            for case in cfg['cases']:
                frames=changed_stream(original,case,model,old['session_config'])
                engines={m:engine(m,cfg,old) for m in cfg['methods']}
                previous_cert=None
                for frame,(unperturbed,truth,noise) in zip(frames,original,strict=True):
                    p=period(frame.frame,len(frames),case)
                    on=p=='active'
                    current=asdict(frame)
                    context=dict(seed=seed,case=case,frame=frame.frame,native=current,
                        truth_speed_mps=truth,witness_noise_mps=noise,period=p)
                    contexts.write(canonical_json(context)+'\n')
                    for method,(e,keys) in engines.items():
                        q=VehicleRequest(f's{seed}-{case}-{frame.frame}',f'n{seed}-{case}-{frame.frame}',
                            'native-reader','native-ego','vehicle.telemetry.longitudinal','read')
                        cert=None;challenge=None
                        before=e.diagnostics('native-ego')
                        if method=='witness_recovery':
                            challenge=e.challenge(q,frame.time_ms)
                            speed=truth+noise+(5 if case=='witness_compromised' else 0)
                            cert=certificate(challenge,speed,frame.time_ms,frame.frame+1,case=='witness_forged' and on)
                            if case=='witness_replay' and on:cert=previous_cert
                            if case=='witness_missing' and on:cert=None
                            if not on:previous_cert=cert
                        wire=packets(frame,3*frame.frame+1,keys,revoked=case=='identity_revoked' and on)
                        if method=='witness_recovery':d=e.evaluate(q,wire,frame.time_ms,cert)
                        else:d=e.evaluate(q,wire,frame.time_ms)
                        diag=e.diagnostics('native-ego')
                        issued_quorum=e._states['native-ego'].quorum
                        granted=False;use=None;cap=None;use_frames=None
                        if d.outcome=='ALLOW':
                            cap=e.issue_capability(d)
                            use_frames=packets(frame,3*frame.frame+2,keys,
                                revoked=(case in {'identity_revoked','postissue_revoked'} and on))
                            use=e.consume_capability(cap,q,use_frames,frame.time_ms)
                            granted=use.authorized
                        after=e.diagnostics('native-ego')
                        physical_error=max(abs(frame.gnss_speed-model.gnss_offset-truth),abs(frame.wheel_speed-truth))
                        row=dict(seed=seed,case=case,method=method,frame=frame.frame,time_ms=frame.time_ms,
                            period=p,context_digest=digest(current),request=asdict(q),
                            policy_digest=e.policy_digest,issued_outcome=d.outcome,issued_reasons=d.reasons,
                            granted=granted,issued_quorum=issued_quorum,
                            consumed_quorum=e._states['native-ego'].quorum,
                            use_outcome=use.current_decision.outcome if use and use.current_decision else None,
                            use_reasons=use.reasons if use else [],
                            quarantine_before=before.get('quarantined',False),
                            epoch_before=before.get('witness_epoch',0),
                            diagnostics=diag,use_diagnostics=after if method=='witness_recovery' and use else None,
                            issue_frames=wire,use_frames=use_frames,capability=cap,
                            use_time_ms=frame.time_ms if use else None,
                            use_certificate=cert if method=='witness_recovery' and use else None,
                            use_challenge=challenge if method=='witness_recovery' and use else None,
                            certificate=cert,challenge=challenge,physical_error_mps=physical_error)
                        output.write(canonical_json(row)+'\n');requests+=1
                        counts=summary[seed,case,method,p];counts['requests']+=1
                        counts['grants']+=granted
                        counts['physically_inaccurate_grants']+=granted and physical_error>1
                        counts['releases']+=diag.get('witness_release',diag.get('released',False))
                print(f'seed {seed}: {case} complete',flush=True)
    verify_freeze(root,f)
    s=dict(schema='coact-witness-synthetic-result-v1',status='SIMULATION_ONLY_NOT_PROMOTED',
        requests=requests,config=cfg,environment=dict(python=platform.python_version(),platform=platform.platform()),
        metrics=metric_summary(summary),source_count=len(f['source_sha256']),
        freeze_sha256=sha((out/'frozen_protocol.json').read_bytes()))
    (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    names=['frozen_protocol.json','contexts.jsonl','requests.jsonl','summary.json']
    (out/'file_hashes.json').write_text(json.dumps({n:sha((out/n).read_bytes()) for n in names},indent=2)+'\n')
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
