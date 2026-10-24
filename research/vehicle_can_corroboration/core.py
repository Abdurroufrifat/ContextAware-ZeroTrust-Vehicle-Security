"""Fixed causal CAN corroboration. No learned parameters and no trust promotion."""
import collections
import math

WHEELS = ('FL_wheel_speed', 'FR_wheel_speed', 'RL_wheel_speed', 'RR_wheel_speed')
RPM_TO_MPS = 2 * math.pi * .305 / 60
MAX_RECEIPT_AGE_US = 100_000


def numeric(v):
    try:
        return type(v) in (int, float) and math.isfinite(v)
    except (OverflowError, ValueError):
        return False


def wheel_speed(row):
    if not isinstance(row, dict) or not all(numeric(row.get(k)) for k in WHEELS):
        return None
    values = sorted(abs(row[k] * RPM_TO_MPS) for k in WHEELS)
    result = values[1] / 2 + values[2] / 2
    return result if math.isfinite(result) else None


def pose_speed(row):
    v = row.get('vel') if isinstance(row, dict) else None
    if not isinstance(v, list) or len(v) != 3 or not all(numeric(x) for x in v):
        return None
    s = math.hypot(v[0], v[1])
    return s if math.isfinite(s) else None


def check_clock(rows, stream):
    previous = -1
    for row in rows:
        t = row.get('utime') if isinstance(row, dict) else None
        if type(t) is not int or t < 0 or t <= previous:
            raise ValueError(stream + '_invalid_clock')
        previous = t


def pair(poses, wheels):
    check_clock(poses, 'pose'); check_clock(wheels, 'wheel')
    rows = []; j = -1
    for i, p in enumerate(poses):
        t = p['utime']
        while j+1 < len(wheels) and wheels[j+1]['utime'] <= t:
            j += 1
        ps = pose_speed(p)
        wt = wheels[j]['utime'] if j >= 0 else None
        age = t-wt if wt is not None else None
        ws = wheel_speed(wheels[j]) if j >= 0 else None
        if ps is None:
            status = 'invalid_pose'
        elif j < 0:
            status = 'no_past_wheel'
        elif age > MAX_RECEIPT_AGE_US:
            status = 'stale_receipt'
        elif ws is None:
            status = 'invalid_latest_wheel'
        else:
            status = 'valid'
        rows.append(dict(pose_index=i, pose_utime=t, wheel_index=j if j >= 0 else None,
                         wheel_utime=wt, receipt_age_us=age, pose_speed_mps=ps,
                         wheel_speed_mps=ws, status=status,
                         signed_disagreement_mps=ws-ps if status == 'valid' else None))
    return rows


def quantile(values, q):
    if not values:
        return None
    s = sorted(values); f = (len(s)-1)*q; lo = math.floor(f); hi = math.ceil(f)
    return s[lo] + (s[hi]-s[lo])*(f-lo)


def summarize(rows):
    n = len(rows)
    errors = [r['signed_disagreement_mps'] for r in rows if r['status'] == 'valid']
    absolute = [abs(e) for e in errors]; k = len(errors)
    age = [r['receipt_age_us']/1000 for r in rows if r.get('receipt_age_us') is not None]
    p95 = quantile(absolute, .95); coverage = 100*k/n if n else None
    return dict(pose_rows=n, valid_pairs=k, coverage_percent=coverage,
                status_counts=dict(sorted(collections.Counter(r['status'] for r in rows).items())),
                signed_bias_mps=math.fsum(errors)/k if k else None,
                mae_disagreement_mps=math.fsum(absolute)/k if k else None,
                rmse_disagreement_mps=math.hypot(*errors)/math.sqrt(k) if k else None,
                p95_abs_disagreement_mps=p95, max_abs_disagreement_mps=max(absolute) if k else None,
                disagreement_gt_0_5_mps_count=sum(e > .5 for e in absolute),
                disagreement_gt_2_mps_count=sum(e > 2 for e in absolute),
                receipt_age_ms=dict(selected_rows=len(age), median=quantile(age, .5),
                                    p95=quantile(age, .95), maximum=max(age) if age else None),
                diagnostic_gates_pass=bool(n and coverage >= 95 and p95 is not None and p95 <= .5 and not any(e > 2 for e in absolute)),
                independent_source_trust_established=False, physical_witness_enabled=False,
                promotion_allowed=False, production_policy_changed=False)
