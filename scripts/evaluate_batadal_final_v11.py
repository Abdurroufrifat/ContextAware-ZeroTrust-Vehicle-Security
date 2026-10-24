"""Evaluate existing checkpoints without training or threshold selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from coact_cps.data.windowing import make_windows
from coact_cps.metrics import binary_detection_metrics
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer

ROOT = Path(__file__).resolve().parents[1]
INTERVALS = [
    ('2017-01-16 09:00', '2017-01-19 06:00', 70),
    ('2017-01-30 08:00', '2017-02-02 00:00', 65),
    ('2017-02-09 03:00', '2017-02-10 09:00', 31),
    ('2017-02-12 01:00', '2017-02-13 07:00', 31),
    ('2017-02-24 05:00', '2017-02-28 08:00', 100),
    ('2017-03-10 14:00', '2017-03-13 21:00', 80),
    ('2017-03-25 20:00', '2017-03-27 01:00', 30),
]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'artifacts/batadal_v11_final'
    if (out / 'final_results.json').exists():
        raise RuntimeError('Final results already exist. Preserve them before another evaluation.')
    source = ROOT / 'data/raw/batadal/BATADAL_test_dataset.csv'
    meta_path = ROOT / 'data/processed/batadal_v11/metadata.json'
    meta = json.loads(meta_path.read_text())
    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    times = pd.to_datetime(frame['DATETIME'], dayfirst=True)
    assert len(frame) == 2089 and not times.duplicated().any()
    assert (times.diff().dropna() == pd.Timedelta(hours=1)).all()
    labels = np.zeros(len(frame), dtype=np.int64)
    for start, end, count in INTERVALS:
        mask = times.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
        assert mask.sum() == count
        labels[mask] = 1
    assert labels.sum() == 407
    features = meta['feature_names']
    raw = frame[features].to_numpy(dtype=np.float32)
    assert np.isfinite(raw).all()
    # Match StandardScaler's float32 subtract-then-divide operations.
    raw -= np.asarray(meta['scaler_mean'])
    raw /= np.asarray(meta['scaler_scale'])
    windows = make_windows(raw, labels, np.full(len(labels), -1),
                           window_size=24, stride=1, label_rule='any')
    end_rows = windows.end_indices
    hourly_y = labels[end_rows]
    out.mkdir(parents=True, exist_ok=True)
    results = {'protocol': 'BATADAL final evaluation, fixed models and thresholds',
               'test_sha256': digest(source), 'metadata_sha256': digest(meta_path),
               'intervals_inclusive': INTERVALS, 'raw_attack_hours': 407,
               'warmup_hours_excluded': 23, 'runs': []}
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for seed in [7, 17, 29, 43, 61]:
        folder = ROOT / f'artifacts/batadal_v11/seeds/seed_{seed}'
        checkpoint = folder / 'detector.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['feature_names'] == features
        model = TemporalHeterogeneousGraphTransformer(**saved['model_parameters'])
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        adjacency = torch.from_numpy(np.load(folder / 'adjacency.npy', allow_pickle=False))
        probabilities = []
        with torch.no_grad():
            for offset in range(0, len(windows.x), 32):
                probability, _ = model.predict_point(
                    torch.from_numpy(windows.x[offset:offset + 32]), adjacency)
                probabilities.extend(probability.numpy().reshape(-1).tolist())
        p = np.asarray(probabilities)
        assert np.isfinite(p).all()
        episodes = []
        for attack_id, (start, end, _) in enumerate(INTERVALS, 8):
            mask = times.iloc[end_rows].between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
            detected = np.flatnonzero(mask & (p >= 0.5))
            delay = None if not len(detected) else (
                times.iloc[end_rows[detected[0]]] - pd.Timestamp(start)).total_seconds() / 3600
            episodes.append({'attack_id': attack_id, 'detected': bool(len(detected)),
                             'delay_hours': delay})
        pd.DataFrame({'timestamp': times.iloc[end_rows].to_numpy(),
                      'hourly_label': hourly_y, 'window_any_label': windows.y,
                      'probability': p, 'alarm': p >= 0.5}).to_csv(
                          out / f'seed_{seed}_predictions.csv', index=False)
        results['runs'].append({'seed': seed, 'checkpoint_sha256': digest(checkpoint),
                                'hourly_metrics': binary_detection_metrics(hourly_y, p),
                                'window_any_metrics': binary_detection_metrics(windows.y, p),
                                'episodes': episodes})
        print(f'Seed {seed} evaluated.', flush=True)
    (out / 'final_results.json').write_text(json.dumps(clean(results), indent=2, allow_nan=False))
    print('Saved artifacts/batadal_v11_final. No training performed.')


if __name__ == '__main__':
    main()
