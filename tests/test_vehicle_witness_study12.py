import hashlib
import json
from pathlib import Path

import pytest


def test_generator_is_causal_bounded_and_reproducible():
    from vehicle_witness_simulation12 import trajectory
    a=trajectory(17,120);b=trajectory(17,120)
    assert a==b and len(a)==120
    for f,truth,noise in a:
        assert 0<=truth<=100 and abs(noise)<=.1
        assert f.interval_start_ms<f.time_ms


def test_smoke_run_audit_and_refusal_to_overwrite(tmp_path):
    from run_vehicle_witness_stage12 import run
    from audit_vehicle_witness_stage12 import audit
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'study'
    cfg=json.loads((root/'configs/vehicle_witness_stage12.json').read_text())
    cfg.update(seeds=[17],frames=24,cases=['normal','witness_missing'])
    run(root,out,cfg)
    result=audit(root,out)
    assert result['status']=='PASS' and result['requests']==24*2*4
    with pytest.raises(FileExistsError):run(root,out,cfg)
    p=out/'requests.jsonl';p.write_text(p.read_text().replace('"granted":false','"granted":true',1))
    with pytest.raises(AssertionError):audit(root,out)
    # Even an attacker who recomputes content hashes cannot hide a false grant.
    h=out/'file_hashes.json';hashes=json.loads(h.read_text())
    hashes['requests.jsonl']=hashlib.sha256(p.read_bytes()).hexdigest()
    h.write_text(json.dumps(hashes))
    with pytest.raises(AssertionError):audit(root,out)


def test_freeze_rejects_changed_source(tmp_path):
    from run_vehicle_witness_stage12 import freeze, verify_freeze
    root=tmp_path/'project';root.mkdir()
    (root/'one.py').write_text('original\n')
    f=freeze(root,{},['one.py'])
    verify_freeze(root,f)
    (root/'one.py').write_text('changed\n')
    with pytest.raises(AssertionError):verify_freeze(root,f)


@pytest.mark.parametrize('target_frame',[0,2])
def test_audit_rejects_rehashed_early_grant_with_valid_witness(tmp_path,target_frame):
    from run_vehicle_witness_stage12 import run
    from audit_vehicle_witness_stage12 import audit
    root=Path(__file__).resolve().parents[1];out=tmp_path/'early'
    cfg=json.loads((root/'configs/vehicle_witness_stage12.json').read_text())
    cfg.update(seeds=[17],frames=3,cases=['normal'],methods=['witness_recovery'])
    run(root,out,cfg)
    p=out/'requests.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
    r=rows[target_frame]
    r.update(issued_outcome='ALLOW',use_outcome='ALLOW',granted=True,issued_quorum=3,consumed_quorum=3)
    r['diagnostics']['motion_quorum']=3
    r['use_diagnostics']=dict(r['diagnostics'])
    from coact_cps.vehicle.policy import SOURCES, CAPABILITY_DOMAIN, _mac, canonical_json
    from run_vehicle_native_stage6 import packets
    from vehicle_native_data import NativeFrame
    contexts=[json.loads(s) for s in (out/'contexts.jsonl').read_text().splitlines()]
    frame=NativeFrame(**contexts[target_frame]['native'])
    keys={name:hashlib.sha256(('Stage12 simulation collector '+name).encode()).digest() for name in SOURCES}
    r['use_frames']=packets(frame,3*target_frame+2,keys)
    r.update(use_time_ms=r['time_ms'],use_certificate=r['certificate'],use_challenge=r['challenge'])
    body=dict(request=r['request'],mode='stream',context_digest='0'*64,policy_digest=r['policy_digest'],
              issued_ms=r['time_ms'],expires_ms=r['time_ms']+250)
    r['capability']=canonical_json(dict(body=body,mac=_mac(body,b'CAPABILITY SIMULATION KEY STAGE12! ',CAPABILITY_DOMAIN)))
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sp=out/'summary.json';s=json.loads(sp.read_text())
    for m in s['metrics']:
        if m['period']==r['period']:m['grants']+=1
    sp.write_text(json.dumps(s))
    hp=out/'file_hashes.json';h=json.loads(hp.read_text())
    for n in h:h[n]=hashlib.sha256((out/n).read_bytes()).hexdigest()
    hp.write_text(json.dumps(h))
    reason='issued_quorum_reconstruction' if target_frame==0 else 'initial_horizon_warmup'
    with pytest.raises(AssertionError,match=reason):audit(root,out)


def test_audit_rejects_false_use_acceptance_flag_with_bad_evidence(tmp_path):
    from run_vehicle_witness_stage12 import run
    from audit_vehicle_witness_stage12 import audit
    root=Path(__file__).resolve().parents[1];out=tmp_path/'use'
    cfg=json.loads((root/'configs/vehicle_witness_stage12.json').read_text())
    cfg.update(seeds=[17],frames=24,cases=['normal'],methods=['witness_recovery'])
    run(root,out,cfg)
    p=out/'requests.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
    r=next(r for r in rows if r['granted'])
    r['use_certificate']='{}'
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    hp=out/'file_hashes.json';h=json.loads(hp.read_text())
    h['requests.jsonl']=hashlib.sha256(p.read_bytes()).hexdigest();hp.write_text(json.dumps(h))
    with pytest.raises(AssertionError):audit(root,out)
