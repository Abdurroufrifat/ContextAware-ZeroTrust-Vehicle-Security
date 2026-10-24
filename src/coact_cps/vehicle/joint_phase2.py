"""Fixed-development causal history candidate; never an active policy replacement."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from coact_cps.vehicle.temporal23 import summarize

ARMS = ('joint_history_gru6', 'raw_history_gru3', 'innovation_history_gru3', 'separate_gru_or')
CHANNELS = {'joint_history_gru6': tuple(range(6)), 'raw_history_gru3': (0, 1, 2),
            'innovation_history_gru3': (3, 4, 5)}


def require_role(rows, trip, segment, role):
    matches = [r for r in rows if r['trip'] == trip and r['segment'] == segment]
    if len(matches) != 1 or matches[0]['role'] != role:
        raise ValueError('reserved, quarantined, unlisted, or wrong-role segment')


def joint_features(values, time, start):
    """Differentiable equivalent of Stage29 innovations, retaining absolute values."""
    x = values if torch.is_tensor(values) else torch.as_tensor(values, dtype=torch.float64)
    t, s = np.asarray(time), np.asarray(start)
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x) or not x.is_floating_point()
            or not bool(torch.isfinite(x).all()) or t.shape != (len(x),) or s.shape != t.shape
            or t.dtype.kind not in 'iu' or s.dtype.kind not in 'iu'):
        raise ValueError('finite three-channel values and integer clocks required')
    t, s = t.astype(np.int64), s.astype(np.int64)
    if (np.any(t <= s) or np.any(t - s > 500) or np.any(np.diff(t) <= 0)
            or np.any(s[1:] < t[:-1])):
        raise ValueError('ordered intervals no longer than 500ms required')
    if len(x) <= 5:
        return torch.cat((x, torch.full_like(x, torch.nan)), dim=1)
    dt = torch.as_tensor((t - s) / 1000, dtype=x.dtype, device=x.device)
    integral = torch.cat((x.new_zeros(1), torch.cumsum(x[:, 2] * dt, 0)))
    accum = integral[6:] - integral[1:-5]
    duration = torch.as_tensor((t[5:] - t[:-5]) / 1000, dtype=x.dtype, device=x.device)
    tail = torch.stack((x[5:, 0] - x[5:, 1],
                        (x[5:, 0] - x[:-5, 0] - accum) / duration,
                        (x[5:, 1] - x[:-5, 1] - accum) / duration), dim=1)
    contiguous = s[1:] == t[:-1]
    ok = np.array([bool(contiguous[i - 5:i].all()) for i in range(5, len(x))])
    tail = torch.where(torch.as_tensor(ok, device=x.device)[:, None], tail,
                       torch.full_like(tail, torch.nan))
    innovations = torch.cat((torch.full_like(x[:5], torch.nan), tail), dim=0)
    return torch.cat((x, innovations), dim=1)


def valid_positions(joint, time, start):
    x = joint.detach().cpu().numpy() if torch.is_tensor(joint) else np.asarray(joint)
    t, s = np.asarray(time), np.asarray(start)
    result = []
    for i in range(31, len(x)):
        if np.isfinite(x[i-31:i+1]).all() and np.array_equal(s[i-30:i+1], t[i-31:i]):
            result.append(i)
    return np.asarray(result, dtype=np.int64)


def forecast_windows(joint, positions, channels):
    p = np.asarray(positions, dtype=np.int64)
    if p.ndim != 1 or not len(p) or np.any(p < 31) or np.any(p >= len(joint)):
        raise ValueError('nonempty valid forecast endpoints required')
    history_indices = torch.as_tensor(p[:, None] - np.arange(31, 0, -1), device=joint.device)
    target_indices = torch.as_tensor(p, device=joint.device)
    selected = joint[:, list(channels)]
    history, targets = selected[history_indices], selected[target_indices]
    if not bool(torch.isfinite(history).all() and torch.isfinite(targets).all()):
        raise ValueError('unavailable history or target')
    return history, targets


def route_weights(routes):
    labels = np.asarray(routes)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if labels.ndim != 1 or not len(labels):
        raise ValueError('nonempty route identifiers required')
    return len(labels) / (len(unique) * counts[inverse])


def fit_scaler(values, routes):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) != len(routes) or not np.isfinite(x).all():
        raise ValueError('finite fit-only feature rows and route IDs required')
    weights = route_weights(routes)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)), .01)
    return dict(mean=mean.tolist(), scale=scale.tolist())


def weighted_quantile(values, weights, p):
    x, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if (x.ndim != 1 or x.shape != w.shape or not len(x) or not np.isfinite(x).all()
            or not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0 or not 0 <= p <= 1):
        raise ValueError('finite values and nonnegative positive-total weights required')
    order = np.argsort(x, kind='stable')
    cumulative = np.cumsum(w[order]); index = np.searchsorted(cumulative, p * cumulative[-1], side='left')
    return float(x[order[min(int(index), len(x) - 1)]])


def fit_profile(residuals, routes):
    x = np.asarray(residuals, dtype=float)
    if x.ndim != 2 or len(x) != len(routes) or not np.isfinite(x).all():
        raise ValueError('finite fit residuals required')
    weights = route_weights(routes)
    quantiles = np.array([[weighted_quantile(x[:, j], weights, q) for j in range(x.shape[1])]
                          for q in (.25, .5, .75)])
    return dict(median=quantiles[1].tolist(), scale=np.maximum(quantiles[2]-quantiles[0],.01).tolist())


def calibrate(route_scores):
    if len(route_scores) != 2:
        raise ValueError('exactly two declared calibration routes required')
    thresholds = []
    for values in route_scores.values():
        x = np.asarray(values)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('finite complete calibration scores required')
        thresholds.append(float(np.quantile(x, .95, method='higher')))
    return max(thresholds)


class HistoryGRU(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.gru = nn.GRU(dimensions, 32, num_layers=1, batch_first=True, dropout=0.)
        self.head = nn.Linear(32, dimensions)

    def forward(self, history):
        _, hidden = self.gru(history)
        return self.head(hidden[-1])


def standardized_windows(joint, positions, channels, scaler):
    indices = list(channels)
    mean = joint.new_tensor(scaler['mean'])[indices]
    scale = joint.new_tensor(scaler['scale'])[indices]
    x, y = forecast_windows(joint, positions, channels)
    return ((x-mean)/scale).float(), ((y-mean)/scale).float()


def model_score(model, joint, positions, channels, scaler, profile):
    x,y=standardized_windows(joint,positions,channels,scaler)
    prediction = torch.cat([model(b) for b in x.split(128)])
    residual = (y - prediction).abs()
    median = residual.new_tensor(profile['median']); spread = residual.new_tensor(profile['scale'])
    contributions = torch.clamp((residual-median)/spread,min=0)
    return contributions.max(dim=1).values, residual, prediction, contributions


def decisions(scores, threshold, base):
    scores, base = np.asarray(scores), np.asarray(base)
    if scores.ndim != 1 or scores.shape != base.shape or base.dtype.kind != 'b' or not np.isfinite(threshold):
        raise ValueError('aligned scores, boolean base permits and finite threshold required')
    known = np.isfinite(scores)
    alarm = known & (scores > threshold)
    return alarm, base & known & ~alarm, known


def evaluate_decisions(context, scores, threshold):
    p = context['positions']; base = context['base_grant']; labels = context['labels']
    alarm, selected_grant, known = decisions(scores, threshold, base[p])
    grant = np.zeros(len(base), dtype=bool); grant[p] = selected_grant
    effective_base = base.copy(); effective_base[p] &= known
    temporal = summarize(context['time_ms'],context['start_ms'],labels,p,alarm,effective_base,grant)
    active = labels[p]; normal = ~labels
    eligible = np.zeros(len(base), dtype=bool); eligible[p] = True
    result = dict(temporal,
        scored_attack_epochs=int(active.sum()), attack_alarm_epochs=int(alarm[active].sum()),
        active_endpoint_tpr=float(alarm[active].mean()) if active.any() else None,
        normal_alarm_epochs=int(alarm[~active].sum()), normal_eligible_epochs=int((normal & eligible).sum()),
        normal_eligible_denied=int((normal & eligible & ~grant).sum()),
        normal_total_denied=int((normal & ~grant).sum()), unknown_scored_epochs=int((~known).sum()))
    block = dict(score=np.asarray(scores),alarm=alarm,known=known,grant=grant,effective_base=effective_base)
    return block, result


def projected_attack(original, active, objective, gnss_offset):
    """Offline full-trace PGD; exact fixed budget, no model/threshold modification."""
    raw = original.detach().clone(); mask = torch.as_tensor(active, dtype=torch.bool, device=raw.device)
    if mask.shape != (len(raw),) or not mask.any():
        raise ValueError('nonempty active interval required')
    lower = raw.new_tensor([3.,3.,-.4]).expand_as(raw).clone()
    upper = raw.new_tensor([8.,8.,.4]).expand_as(raw).clone()
    upper[:,0] = torch.minimum(upper[:,0],100.-raw[:,0]-gnss_offset)
    upper[:,1] = torch.minimum(upper[:,1],100.-raw[:,1])
    if bool((upper[mask] < lower[mask]).any()):
        raise ValueError('infeasible declared attack domain')
    delta = raw.new_tensor([3.,3.,0.]).expand_as(raw).clone()
    delta[~mask] = 0.; delta.requires_grad_(True)
    steps = raw.new_tensor([.25,.25,.02])
    best = None; best_value = float('inf'); best_iteration = None; values=[]; iterates=[]
    for iteration in range(21):
        value = objective(raw + delta)
        number = float(value.detach())
        if not np.isfinite(number):
            raise ValueError('nonfinite white-box attack objective')
        values.append(number)
        iterates.append((raw+delta).detach().cpu().numpy().copy())
        if number < best_value:
            best_value = number; best = (raw+delta).detach().clone(); best_iteration=iteration
        if iteration == 20:break
        gradient, = torch.autograd.grad(value, delta)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError('nonfinite white-box gradient')
        with torch.no_grad():
            delta -= steps * gradient.sign()
            delta.copy_(torch.maximum(lower,torch.minimum(upper,delta)))
            delta[~mask] = 0.
    return best, dict(objectives=values,best_objective=best_value,best_iteration=best_iteration,
                      attack_is_offline_noncausal=True,iterations=20,iterates=np.stack(iterates))
