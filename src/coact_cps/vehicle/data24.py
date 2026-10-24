"""Metadata-only native route selection and unmodified channel validation."""

import numpy as np

CHANNELS = {
    'GNSS/live_gnss_ublox': 6,
    'CAN/wheel_speed': 4,
    'IMU/accelerometer': 3,
    'IMU/gyro': 3,
}


def select_trips(records, prior_routes):
    names = [r['name'] for r in records]
    if len(set(names)) != len(names):
        raise ValueError('duplicate archive entries')
    prior_days = {r.split('|')[1][:10] for r in prior_routes}
    available = {}
    for name in names:
        parts = name.split('/')
        if len(parts) != 7 or parts[0] != 'Chunk_2' or parts[3] != 'processed_log':
            continue
        route = parts[1]
        if not route.startswith('b0c9d2329ad1606b|') or route in prior_routes:
            continue
        day = route.split('|')[1][:10]
        if day in prior_days or not parts[2].isdigit():
            continue
        available.setdefault(route, {}).setdefault(int(parts[2]), set()).add('/'.join(parts[4:]))
    required = {f'{channel}/{kind}' for channel in CHANNELS for kind in ('t', 'value')}
    trips, days = [], set()
    for route, segments in sorted(available.items()):
        complete = {s for s, files in segments.items() if required <= files}
        pairs = sorted(s for s in complete if s + 1 in complete)
        day = route.split('|')[1][:10]
        if not pairs or day in days:
            continue
        days.add(day)
        i = len(trips)
        trips.append(dict(id=f'rav24_{i + 1:02d}', route_id=route, day=day,
                          segments=[pairs[0], pairs[0] + 1],
                          role='development' if i < 3 else 'reserved_validation'))
        if len(trips) == 5:
            return trips
    raise ValueError('five complete RAV4 dates absent from prior inventory required')


def check_channel(time, values, columns, allow_ties):
    t, v = np.asarray(time), np.asarray(values)
    if (t.ndim != 1 or len(t) < 2 or v.shape != (len(t), columns)
            or t.dtype.kind not in 'fiu' or v.dtype.kind not in 'fiu'
            or not np.isfinite(t).all() or not np.isfinite(v).all()):
        raise ValueError('finite aligned native timestamp/value arrays required')
    diff = np.diff(t.astype(float))
    if np.any(diff < 0) or (not allow_ties and np.any(diff == 0)):
        raise ValueError('native timestamps out of permitted order')
    return dict(samples=len(t), first_timestamp_seconds=float(t[0]),
                last_timestamp_seconds=float(t[-1]),
                duplicate_timestamp_count=int((diff == 0).sum()),
                maximum_gap_seconds=float(diff.max()))
