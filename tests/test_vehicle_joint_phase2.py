"""Behavioral checks for causal features, calibration and fail-closed accounting."""
import numpy as np
import pytest
import torch


def motion(n=85):
    time = np.arange(1, n + 1, dtype=np.int64) * 100
    start = time - 100
    v = 5 + 2 * time / 1000
    return np.column_stack((v, v, np.full(n, 2.0))), time, start


def test_joint_features_match_physics_and_retain_common_offset():
    from coact_cps.vehicle import joint_phase2 as m
    from coact_cps.vehicle.innovation29 import innovations
    x, t, s = motion()
    result = m.joint_features(torch.tensor(x), t, s).numpy()
    np.testing.assert_allclose(result[:, 3:], innovations(x, t, s), equal_nan=True, atol=1e-12)
    shifted = x.copy(); shifted[:, :2] += 8
    other = m.joint_features(torch.tensor(shifted), t, s).numpy()
    np.testing.assert_allclose(other[5:, 3:], result[5:, 3:], atol=1e-12)
    np.testing.assert_allclose(other[:, :2] - result[:, :2], 8)


def test_irregular_integral_and_gap_reset():
    from coact_cps.vehicle import joint_phase2 as m
    from coact_cps.vehicle.innovation29 import innovations
    x, t, s = motion()
    t[30:] += 900; s[30:] += 900
    j = m.joint_features(torch.tensor(x), t, s)
    np.testing.assert_allclose(j[:, 3:].numpy(), innovations(x, t, s), equal_nan=True, atol=1e-12)
    p = m.valid_positions(j, t, s)
    assert not np.any((p >= 30) & (p < 66))
    assert p[0] == 66


def test_forecast_input_excludes_target_and_future():
    from coact_cps.vehicle import joint_phase2 as m
    x, t, s = motion(); j = m.joint_features(torch.tensor(x), t, s)
    p = m.valid_positions(j, t, s)
    assert p[0] == 36
    first, y = m.forecast_windows(j, p, [0, 1, 2])
    changed = torch.tensor(x); changed[36:, :] += 10
    other = m.joint_features(changed, t, s)
    second, y2 = m.forecast_windows(other, p, [0, 1, 2])
    torch.testing.assert_close(first[0], second[0])
    assert not torch.equal(y[0], y2[0])


def test_attacked_innovations_are_recomputed_and_differentiable():
    from coact_cps.vehicle import joint_phase2 as m
    x, t, s = motion(); x[40, 0] += 3
    v = torch.tensor(x, requires_grad=True)
    j = m.joint_features(v, t, s)
    assert j[40, 3].item() == pytest.approx(3)
    j[40, 4].backward()
    assert v.grad[40, 0] == pytest.approx(2)
    assert v.grad[35, 0] == pytest.approx(-2)
    assert v.grad[41:].abs().sum() == 0


def test_scaler_balances_routes_rather_than_windows():
    from coact_cps.vehicle import joint_phase2 as m
    data = np.r_[np.zeros((1, 2)), np.full((9, 2), 10.)]
    labels = np.array(['short'] + ['long'] * 9)
    scaler = m.fit_scaler(data, labels)
    np.testing.assert_allclose(scaler['mean'], [5, 5])
    np.testing.assert_allclose(scaler['scale'], [5, 5])
    weights = m.route_weights(labels)
    assert weights[:1].sum() == pytest.approx(weights[1:].sum())


def test_weighted_quantile_uses_cumulative_mass():
    from coact_cps.vehicle import joint_phase2 as m
    assert m.weighted_quantile([0, 2, 8], [0.25, 0.25, 0.5], .5) == 2
    with pytest.raises(ValueError): m.weighted_quantile([0, 2], [1, -1], .5)


def test_training_and_inference_share_float64_scaling_before_float32_cast():
    from coact_cps.vehicle import joint_phase2 as m
    x,t,s=motion();j=m.joint_features(torch.tensor(x),t,s);p=m.valid_positions(j,t,s)
    scaler=dict(mean=[.12345678912345678]*6,scale=[.9876543219876543]*6)
    history,target=m.standardized_windows(j,p,(0,1,2),scaler)
    raw_x,raw_y=m.forecast_windows(j,p,(0,1,2))
    expected_x=((raw_x-.12345678912345678)/.9876543219876543).float()
    expected_y=((raw_y-.12345678912345678)/.9876543219876543).float()
    assert torch.equal(history,expected_x)
    assert torch.equal(target,expected_y)


def test_calibration_uses_worst_route_and_strict_comparison():
    from coact_cps.vehicle import joint_phase2 as m
    threshold = m.calibrate({'a': np.arange(100), 'b': np.arange(100) + 20})
    assert threshold == 115
    scores = np.array([115., 116., np.nan, -np.inf])
    alarm, grant, known = m.decisions(scores, threshold, np.ones(4, dtype=bool))
    assert alarm.tolist() == [False, True, False, False]
    assert grant.tolist() == [True, False, False, False]
    assert known.tolist() == [True, True, False, False]


def test_base_policy_denial_cannot_be_overridden():
    from coact_cps.vehicle import joint_phase2 as m
    _, grant, _ = m.decisions(np.zeros(3), 1., np.array([True, False, False]))
    assert grant.tolist() == [True, False, False]


def test_unknown_cost_and_later_escape_are_not_hidden():
    from coact_cps.vehicle import joint_phase2 as m
    time = np.arange(1, 7, dtype=np.int64) * 100; start = time - 100
    context = dict(time_ms=time,start_ms=start,positions=np.arange(1,6),
                   labels=np.array([False,False,True,True,True,True]),base_grant=np.ones(6,dtype=bool))
    score = np.array([np.nan, 2., 0., 0., 2.])
    block, metric = m.evaluate_decisions(context, score, 1.)
    assert metric['unknown_scored_epochs'] == 1
    assert metric['normal_total_denied'] == 2
    assert metric['normal_eligible_denied'] == 1
    assert metric['attack_longest_grant_run_epochs'] == 2
    assert metric['attack_granted_observed_span_ms'] == 200
    assert metric['events'][0]['first_neural_alarm_delay_ms'] == 100
    assert metric['events'][0]['granted_epochs_after_first_denial'] == 2
    assert not block['grant'][0]


def test_grant_runs_reset_on_clock_gaps():
    from coact_cps.vehicle import joint_phase2 as m
    time=np.array([100,200,1200,1300],dtype=np.int64); start=time-100
    c=dict(time_ms=time,start_ms=start,positions=np.arange(4),labels=np.ones(4,dtype=bool),base_grant=np.ones(4,dtype=bool))
    _,r=m.evaluate_decisions(c,np.zeros(4),1.)
    assert r['attack_longest_grant_run_epochs']==2
    assert r['attack_events']==2
    assert r['neural_missed_events']==2


def test_role_guard_rejects_locked_and_quarantined_segments():
    from coact_cps.vehicle import joint_phase2 as m
    rows=[dict(trip='rav31_02',segment=41,role='fit')]
    m.require_role(rows,'rav31_02',41,'fit')
    for trip,segment,role in [('rav31_01',26,'fit'),('rav31_07',30,'fit'),('rav31_02',41,'development_evaluate')]:
        with pytest.raises(ValueError):m.require_role(rows,trip,segment,role)


def test_projected_attack_preserves_budget_and_nonactive_samples():
    from coact_cps.vehicle import joint_phase2 as m
    x=torch.tensor([[10.,10.,0.],[10.,10.,0.],[10.,10.,0.]],dtype=torch.float64)
    active=np.array([False,True,False])
    attacked,trace=m.projected_attack(x,active,lambda a: ((a[1,:2]-14)**2).mean(),0.)
    torch.testing.assert_close(attacked[[0,2]],x[[0,2]])
    assert torch.all(attacked[1,:2]>=13) and torch.all(attacked[1,:2]<=18)
    assert trace['best_objective'] < trace['objectives'][0]
    assert len(trace['objectives'])==21
    assert trace['iterates'].shape==(21,3,3)
    np.testing.assert_allclose(trace['iterates'][trace['best_iteration']],attacked.numpy())


def test_model_output_depends_only_on_provided_history_and_roundtrips(tmp_path):
    from coact_cps.vehicle import joint_phase2 as m
    torch.manual_seed(7)
    model=m.HistoryGRU(6).eval(); history=torch.randn(4,31,6)
    before=model(history).detach()
    path=tmp_path/'state.pt';torch.save(model.state_dict(),path)
    other=m.HistoryGRU(6).eval();other.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(before,other(history))
    assert before.shape==(4,6)


def test_future_changes_do_not_change_past_features():
    from coact_cps.vehicle import joint_phase2 as m
    x,t,s=motion();a=m.joint_features(torch.tensor(x),t,s)
    x[50:]+=1; b=m.joint_features(torch.tensor(x),t,s)
    torch.testing.assert_close(a[:50],b[:50],equal_nan=True)
