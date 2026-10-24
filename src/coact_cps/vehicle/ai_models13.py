"""Normal-only vehicle models. Published-method adaptations are explicit."""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer

METHODS = (
    "coact_physics",
    "coact_no_graph",
    "coact_no_physics",
    "gdn_adapted",
    "catch_adapted",
    "isolation_forest",
    "lof",
    "persistence",
)


def adjacency(no_graph=False):
    a = torch.eye(3).repeat(2, 1, 1)
    if not no_graph:
        a[0, 0, 1] = a[0, 1, 0] = 1
        a[1, 0, 2] = a[1, 1, 2] = a[1, 2, 0] = a[1, 2, 1] = 1
    return a


def physics_loss(pred, x, dt, scaling):
    mean = torch.as_tensor(scaling.mean, dtype=pred.dtype, device=pred.device)
    scale = torch.as_tensor(scaling.scale, dtype=pred.dtype, device=pred.device)
    p = pred * scale + mean
    previous = x[:, -2] * scale + mean
    return (
        (p[:, 0] - p[:, 1]).square()
        + (p[:, :2].mean(-1) - previous[:, :2].mean(-1) - p[:, 2] * dt).square()
    ).mean()


class DenseGDN(nn.Module):
    """Dense 3-node equivalent of MIT author GDN message passing at9853899.

    Compact dimension32/topk2/common31lags are declared adaptations. Original
    source/license retained under data/reference/vehicle_ai_stage13/gdn_original.
    """

    def __init__(self, window=32, dim=32, topk=2):
        super().__init__()
        self.topk = topk
        self.embedding = nn.Embedding(3, dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        self.projection = nn.Linear(window - 1, dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        self.att_i = nn.Parameter(torch.empty(dim))
        self.att_j = nn.Parameter(torch.empty(dim))
        self.emb_i = nn.Parameter(torch.zeros(dim))
        self.emb_j = nn.Parameter(torch.zeros(dim))
        for a in (self.att_i, self.att_j):
            nn.init.uniform_(a, -math.sqrt(6 / (1 + dim)), math.sqrt(6 / (1 + dim)))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bn = nn.BatchNorm1d(dim)
        self.bn_out = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        z = self.projection(x[:, :-1].transpose(1, 2))
        e = self.embedding.weight
        detached = e.detach()
        norm = detached.norm(dim=-1).clamp_min(1e-12)
        cosine = (detached @ detached.T) / (norm[:, None] * norm[None, :])
        top = cosine.topk(self.topk, dim=-1).indices
        mask = torch.zeros(3, 3, dtype=torch.bool, device=x.device).scatter_(1, top, True)
        mask.fill_diagonal_(True)
        ai = (z * self.att_i + e * self.emb_i).sum(-1)
        aj = (z * self.att_j + e * self.emb_j).sum(-1)
        attention = torch.nn.functional.leaky_relu(ai[:, :, None] + aj[:, None, :], 0.2)
        attention = torch.softmax(attention.masked_fill(~mask, -torch.inf), dim=-1)
        message = attention @ z + self.bias
        h = torch.relu(self.bn(message.reshape(-1, z.shape[-1]))).reshape(z.shape)
        h = h * e
        h = torch.relu(self.bn_out(h.reshape(-1, z.shape[-1]))).reshape(z.shape)
        return self.out(self.dropout(h)).squeeze(-1)


def catch_config(window):
    return SimpleNamespace(
        c_in=3,
        seq_len=window,
        pred_len=window,
        affine=0,
        subtract_last=0,
        patch_size=16,
        patch_stride=8,
        cf_dim=32,
        e_layers=1,
        n_heads=2,
        d_ff=128,
        head_dim=16,
        dropout=0.2,
        d_model=16,
        regular_lambda=0.5,
        temperature=0.07,
        individual=0,
        head_dropout=0.1,
        auxi_mode="fft",
        auxi_type="complex",
        auxi_loss="MAE",
        module_first=True,
        mask=False,
        inference_patch_size=16,
        inference_patch_stride=1,
    )


def catch_modules(root):
    from prepare_vehicle_ai_stage13 import verify_upstream

    verify_upstream(root)
    path = str(Path(root).resolve() / "third_party/vehicle_ai_stage13/catch")
    if path not in sys.path:
        sys.path.insert(0, path)
    m = importlib.import_module("ts_benchmark.baselines.catch.models.CATCH_model")
    f = importlib.import_module("ts_benchmark.baselines.catch.utils.fre_rec_loss")
    if not Path(m.__file__).resolve().is_relative_to(Path(path)):
        raise ValueError("unexpected_CATCH_import_location")
    return m.CATCHModel, f.frequency_loss, f.frequency_criterion


@dataclass
class Adapter:
    method: str
    model: object
    window: int
    root: Path


def create_adapter(method, cfg, root, seed):
    if method not in METHODS:
        raise ValueError("unknown_method")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    window = cfg.get("window", 32)
    if method.startswith("coact"):
        model = TemporalHeterogeneousGraphTransformer(
            3,
            1,
            2,
            window,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.1,
            detection_mode="normal_only",
            score_top_k=1,
        )
    elif method == "gdn_adapted":
        model = DenseGDN(window)
    elif method == "catch_adapted":
        model = catch_modules(root)[0](catch_config(window))
    else:
        model = None
    return Adapter(method, model, window, Path(root))


def summary_features(x):
    return np.concatenate([x[:, -1], x.mean(1), x.std(1), x[:, -1] - x[:, 0]], axis=-1)


def fit_adapter(method, x, dt, scaling, cfg, seed, root):
    x = np.asarray(x, dtype=np.float32)
    dt = np.asarray(dt, dtype=np.float32)
    if (
        x.ndim != 3
        or x.shape[1:] != (cfg.get("window", 32), 3)
        or len(x) != len(dt)
        or len(x) < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError("finite_training_windows_required")
    a = create_adapter(method, cfg, root, seed)
    history = []
    if method in {"isolation_forest", "lof"}:
        a.model = (
            IsolationForest(
                n_estimators=200, max_samples=min(256, len(x)), random_state=seed, n_jobs=1
            )
            if method == "isolation_forest"
            else LocalOutlierFactor(n_neighbors=min(20, len(x) - 1), novelty=True, n_jobs=1)
        )
        a.model.fit(summary_features(x))
        return a, history
    if method == "persistence":
        return a, history
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(dt)),
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if method == "catch_adapted":
        freq = catch_modules(root)[1](catch_config(a.window))
        optimizer = torch.optim.Adam(
            [p for n, p in a.model.named_parameters() if "mask_generator" not in n], lr=0.0001
        )
        mask_optimizer = torch.optim.Adam(a.model.mask_generator.parameters(), lr=0.00001)
        cadence = max(1, min(len(loader) // 10, 100))
    else:
        optimizer = torch.optim.Adam(a.model.parameters(), lr=0.001)
    a.model.train()
    for _epoch in range(cfg.get("epochs", 20)):
        total = 0.0
        seen = 0
        for i, (xb, d) in enumerate(loader):
            optimizer.zero_grad()
            if method.startswith("coact"):
                masked = xb.clone()
                masked[:, -1] = 0
                pred = a.model(
                    masked[:, :, :, None], adjacency(method == "coact_no_graph")
                ).reconstructed[:, -1, :, 0]
                loss = (pred - xb[:, -1]).square().mean()
                if method != "coact_no_physics":
                    loss = loss + 0.05 * physics_loss(pred, xb, d, scaling)
            elif method == "gdn_adapted":
                loss = (a.model(xb) - xb[:, -1]).square().mean()
            else:
                pred, complex_pred, discovery = a.model(xb)
                normalized = a.model.revin_layer(xb, "transform")
                loss = (
                    (pred - xb).square().mean()
                    + 0.005 * discovery
                    + 0.005 * freq(complex_pred, normalized)
                )
                if (i + 1) % cadence == 0:
                    mask_optimizer.step()
                    mask_optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite_training_loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        history.append(total / seen)
    a.model.eval()
    return a, history


@torch.no_grad()
def score_nodes(a, x, cfg, seed):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (a.window, 3) or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite_scoring_windows_required")
    if a.method == "persistence":
        return np.abs(x[:, -1] - x[:, -2]).astype(float)
    if a.method in {"isolation_forest", "lof"}:
        return -a.model.score_samples(summary_features(x))[:, None]
    torch.manual_seed(seed)
    a.model.eval()
    result = []
    if a.method == "catch_adapted":
        freq = catch_modules(a.root)[2](catch_config(a.window))
    for start in range(0, len(x), cfg.get("batch_size", 64)):
        xb = torch.from_numpy(x[start : start + cfg.get("batch_size", 64)])
        if a.method.startswith("coact"):
            masked = xb.clone()
            masked[:, -1] = 0
            pred = a.model(
                masked[:, :, :, None], adjacency(a.method == "coact_no_graph")
            ).reconstructed[:, -1, :, 0]
            score = torch.abs(pred - xb[:, -1])
        elif a.method == "gdn_adapted":
            score = torch.abs(a.model(xb) - xb[:, -1])
        else:
            pred, _, _ = a.model(xb)
            score = (pred - xb).square()[:, -1] + 0.05 * freq(xb, pred)[:, -1]
        result.append(score.numpy().astype(float))
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_AI_score")
    return values


def fit_node_calibration(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite_clean_calibration_required")
    return dict(
        median=np.median(x, axis=0).tolist(),
        scale=np.maximum(
            np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0), 0.01
        ).tolist(),
    )


def aggregate_scores(nodes, calibration, method):
    x = (np.asarray(nodes, dtype=float) - np.asarray(calibration["median"])) / np.asarray(
        calibration["scale"]
    )
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_calibrated_score")
    if method == "gdn_adapted":
        total = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        x = np.stack([(total[i + 1] - total[max(0, i - 3)]) / min(4, i + 1) for i in range(len(x))])
    return x.mean(-1) if method == "catch_adapted" else x.max(-1)


def threshold(scores):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite_clean_threshold_scores_required")
    return float(np.quantile(x, 0.95))
