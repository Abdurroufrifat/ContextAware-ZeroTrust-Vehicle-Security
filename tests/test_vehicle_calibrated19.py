from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coact_cps.vehicle.calibrated19 import calibration_loss, fine_tune, make_windows, require_clean


def context():
    n = 40
    t = np.arange(1, n + 1, dtype=np.int64) * 200
    return {
        "features": np.column_stack((t / 1000, t / 1000, np.ones(n))),
        "time_ms": t,
        "start_ms": t - 200,
        "positions": np.arange(31, n),
        "labels": np.zeros(n, dtype=bool),
    }


SCALER = {"mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def test_fine_tuning_masks_target_and_updates_all_channels_reproducibly():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(3))

        def forward(self, masked, _adjacency):
            assert torch.all(masked[:, -1] == 0)
            prediction = self.value[None, None, :, None].expand(len(masked), 32, 3, 1)
            return SimpleNamespace(reconstructed=prediction)

    x, dt = physics_inputs()
    a, b = SimpleNamespace(model=Decoder()), SimpleNamespace(model=Decoder())
    first = fine_tune(a, x.numpy(), dt.numpy(), SCALER, 7, 0.5, 0.2)
    second = fine_tune(b, x.numpy(), dt.numpy(), SCALER, 7, 0.5, 0.2)
    assert len(first) == 5
    assert np.isfinite(first).all()
    assert first == second
    assert torch.all(a.model.value != 0)
    torch.testing.assert_close(a.model.value, b.model.value, rtol=0, atol=0)


def test_windows_keep_original_scored_endpoints():
    c = context()
    x, dt = make_windows(c, SCALER)
    assert x.shape == (9, 32, 3)
    np.testing.assert_array_equal(x[:, -1], c["features"][c["positions"]].astype(np.float32))
    np.testing.assert_allclose(dt, 0.2)


def test_gap_and_wrong_endpoint_fail_closed():
    for mode in ("gap", "position"):
        c = context()
        if mode == "gap":
            c["start_ms"][20] -= 1
        else:
            c["positions"][0] = 0
        with pytest.raises(ValueError):
            make_windows(c, SCALER)


def test_invalid_scaler_and_nonfinite_features_rejected():
    c = context()
    with pytest.raises(ValueError):
        make_windows(c, {"mean": [0] * 3, "scale": [0] * 3})
    c["features"][0, 2] = np.nan
    with pytest.raises(ValueError):
        make_windows(c, SCALER)


def test_attack_rows_never_enter_fitting():
    c = context()
    c["labels"][35] = True
    with pytest.raises(ValueError, match="normal"):
        require_clean(c)


def physics_inputs():
    # speed derivative = .5 * observed IMU + .2 = 1 m/s².
    c = context()
    c["features"][:, 2] = 1.6
    x, dt = make_windows(c, SCALER)
    return torch.from_numpy(x), torch.from_numpy(dt)


def test_calibrated_physics_matches_known_kinematics():
    x, dt = physics_inputs()
    p = x[:, -1].clone()
    loss = calibration_loss(p, x, dt, SCALER, 0.5, 0.2)
    assert loss.item() < 1e-10
    assert calibration_loss(p, x, dt, SCALER, 1.0, 0.0).item() > 0.1


def test_physics_uses_past_imu_not_current_target():
    x, dt = physics_inputs()
    p = x[:, -1].clone()
    before = calibration_loss(p, x, dt, SCALER, 0.5, 0.2)
    x[:, -1] += 500
    torch.testing.assert_close(before, calibration_loss(p, x, dt, SCALER, 0.5, 0.2))


def test_loss_remains_differentiable_and_mse_retains_imu():
    x, dt = physics_inputs()
    p = (x[:, -1] + 0.1).detach().requires_grad_()
    loss = (p - x[:, -1]).square().mean() + 0.05 * calibration_loss(p, x, dt, SCALER, 0.5, 0.2)
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert torch.all(p.grad[:, 2] != 0)


def test_original_scaling_is_reversed_for_physics():
    x, dt = physics_inputs()
    mean = torch.tensor([2.0, 2.0, 0.4])
    scale = torch.tensor([3.0, 3.0, 0.5])
    standardized = (x - mean) / scale
    s = {"mean": mean.tolist(), "scale": scale.tolist()}
    assert calibration_loss(standardized[:, -1], standardized, dt, s, 0.5, 0.2).item() < 1e-10
