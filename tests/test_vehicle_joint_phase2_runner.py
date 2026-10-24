"""Experiment boundary tests with real saved-context bytes and causal frames."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def test_role_loader_never_requires_evaluation_payloads(tmp_path):
    import run_vehicle_joint_phase2 as r
    folder=tmp_path/'artifacts/vehicle_transfer_stage32/contexts';folder.mkdir(parents=True)
    np.savez(folder/'rav31_02__41.npz',features=np.ones((50,3)),time_ms=np.arange(1,51)*100,
             start_ms=np.arange(50)*100,source_time_ms=np.tile(np.arange(1,51)[:,None]*100,(1,3)),
             maximum_gap_ms=np.full(50,10))
    p={'development_segments':[dict(trip='rav31_02',segment=41,role='fit'),
                               dict(trip='rav31_06',segment=4,role='development_evaluate')]}
    data=r.load_role(tmp_path,p,'fit')
    assert list(data)==['rav31_02__41']
    assert data['rav31_02__41']['positions'][0]==36
    with pytest.raises(FileNotFoundError):r.load_role(tmp_path,p,'development_evaluate')


def test_native_frame_roundtrip_restores_offsets_exactly_once():
    import run_vehicle_joint_phase2 as r
    from vehicle_ai_data13 import features
    from coact_cps.vehicle.native_detectors import ModelConfig
    with np.load(ROOT/'artifacts/vehicle_transfer_stage32/contexts/rav24_01__37.npz') as z:
        c={k:z[k] for k in z.files}
    model=ModelConfig(gnss_offset=.17194416787889377,imu_offset=-.6035401722836787)
    frames=r.frames_from_context(c,'route',model)
    np.testing.assert_allclose(features(frames,model),c['features'],rtol=0,atol=1e-14)
    assert [f.gnss_time_ms for f in frames]==c['source_time_ms'][:,0].tolist()
    assert frames[0].frame==0 and frames[-1].frame==len(frames)-1


def test_fit_training_decreases_loss_and_has_no_target_input():
    import run_vehicle_joint_phase2 as r
    x=torch.zeros(32,31,3); y=torch.ones(32,3)
    model,losses=r.train_history(x,y,np.array(['a']*8+['b']*24),7)
    assert losses[-1]<losses[0]
    assert not model.training
    assert sum(p.numel() for p in model.parameters())>0


def test_worst_route_is_not_rescued_by_macro_or_seed_average():
    import run_vehicle_joint_phase2 as r
    rows=[]
    for seed in (7,19,43):
        for trip,alarms in [('a',0),('b',6 if seed==7 else 0)]:
            for method in ('joint_history_gru6','raw_history_gru3'):
                rows.append(dict(case='normal',trip=trip,segment=1,seed=seed,method=method,
                    metrics=dict(normal_eligible_epochs=100,normal_alarm_epochs=alarms,
                    normal_eligible_denied=alarms,normal_epochs=110,normal_total_denied=10+alarms,
                    epochs=110,scored_epochs=100,unknown_scored_epochs=0)))
    report=r.normal_gates(rows)
    assert report['per_seed']['7']['macro_alarm_fraction']==pytest.approx(.03)
    assert report['per_seed']['7']['worst_route_alarm_fraction']==pytest.approx(.06)
    assert not report['per_seed']['7']['normal_limits_pass']
    assert not report['all_normal_gates_pass']


def test_attack_comparison_detects_later_grant_escape_even_after_early_alarm():
    import run_vehicle_joint_phase2 as r
    from coact_cps.vehicle.joint_phase2 import evaluate_decisions
    t=np.arange(1,6,dtype=np.int64)*100
    context=dict(time_ms=t,start_ms=t-100,positions=np.arange(5),labels=np.ones(5,dtype=bool),base_grant=np.ones(5,dtype=bool))
    rows=[]
    for method,score in [('original_nn_frozen',[2.,2.,2.,2.,2.]),
                         ('raw_history_gru3',[2.,2.,2.,2.,2.]),
                         ('joint_history_gru6',[2.,0.,0.,0.,2.])]:
        _,metrics=evaluate_decisions(context,np.array(score),1.)
        rows.append(dict(case='joint_jump',trip='a',segment=1,seed=7,method=method,metrics=metrics))
    checks=r.attack_gates(rows)
    assert not checks['all_attack_gates_pass']
    assert any('longer_grant_run' in f['reasons'] for f in checks['failed_cells'])


def test_pinned_protocol_rejects_changed_route_roles(tmp_path):
    import run_vehicle_joint_phase2 as r
    source=ROOT/'research/phase1_method_decision/PHASE2_PROTOCOL.json'
    p=json.loads(source.read_text());p['development_roles']['fit'].append('rav31_07')
    file=tmp_path/'protocol.json';file.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Phase1 protocol pin'):r.load_protocol(file)


def test_physics_reference_reports_native_and_matched_policy_decisions():
    import run_vehicle_joint_phase2 as r
    time=np.arange(1,6,dtype=np.int64)*100
    context=dict(time_ms=time,start_ms=time-100,positions=np.arange(1,5),labels=np.ones(5,dtype=bool),
                 base_grant=np.array([False,True,False,True,True]))
    report=r.physics_reference(context)
    assert report['native']['attack_granted_epochs']==3
    assert report['native']['policy_missed_events']==0
    assert report['native']['events'][0]['first_policy_denial_delay_ms']==100
    assert report['matched']['attack_longest_grant_run_epochs']==2
