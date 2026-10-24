"""Independent scalar reconstruction: bisect pairing, separate conversion and reducers.
Only filesystem safety, inventory hashing and JSON I/O are shared with the runner.
"""
import argparse
import bisect
import gzip
import json
import math
from pathlib import Path
import sys
import run_can as io


def finite(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def scalar_clock(rows):
    times = []
    for r in rows:
        t = r.get('utime') if isinstance(r, dict) else None
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            return None
        if times and t <= times[-1]:
            return None
        times.append(t)
    return times


def scalar_rows(p, w):
    wt = scalar_clock(w)
    rows = []
    for i in range(len(p)):
        t = p[i]['utime']; j = bisect.bisect_right(wt, t)-1
        vector = p[i].get('vel')
        ps = None
        if isinstance(vector, list) and len(vector) == 3 and all(finite(v) for v in vector):
            ps = math.sqrt(vector[0]**2 + vector[1]**2)
            if not math.isfinite(ps):
                ps = None
        ws = None
        if j >= 0:
            fields = ['FL_wheel_speed', 'FR_wheel_speed', 'RL_wheel_speed', 'RR_wheel_speed']
            values = [w[j].get(k) for k in fields]
            if all(finite(v) for v in values):
                speeds = sorted([abs(v) * (2*math.pi*.305/60) for v in values])
                ws = (speeds[1]+speeds[2])/2
                if not math.isfinite(ws):
                    ws = None
        stamp = wt[j] if j >= 0 else None
        age = t-stamp if stamp is not None else None
        status = ('invalid_pose' if ps is None else 'no_past_wheel' if j < 0
                  else 'stale_receipt' if age > 100000 else 'invalid_latest_wheel' if ws is None else 'valid')
        rows.append(dict(pose_index=i, pose_utime=t, wheel_index=j if j >= 0 else None,
                         wheel_utime=stamp, receipt_age_us=age, pose_speed_mps=ps,
                         wheel_speed_mps=ws, status=status,
                         signed_disagreement_mps=ws-ps if status == 'valid' else None))
    return rows


def percentile(values, fraction):
    if len(values) == 0:
        return None
    ordered = sorted(values); position = (len(ordered)-1)*fraction
    lower = int(position); upper = min(lower+1, len(ordered)-1); weight = position-lower
    return ordered[lower]*(1-weight)+ordered[upper]*weight


def scalar_summary(rows):
    errors = []; ages = []; counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0)+1
        if r['status'] == 'valid':
            errors.append(r['signed_disagreement_mps'])
        if r['receipt_age_us'] is not None:
            ages.append(r['receipt_age_us'] / 1000)
    n = len(rows); k = len(errors); a = [abs(v) for v in errors]
    coverage = k/n*100 if n else None; p95 = percentile(a, .95)
    over2 = len([v for v in a if v > 2])
    return dict(pose_rows=n, valid_pairs=k, coverage_percent=coverage,
                status_counts=dict(sorted(counts.items())),
                signed_bias_mps=sum(errors)/k if k else None,
                mae_disagreement_mps=sum(a)/k if k else None,
                rmse_disagreement_mps=math.sqrt(sum(v*v for v in errors)/k) if k else None,
                p95_abs_disagreement_mps=p95, max_abs_disagreement_mps=max(a) if k else None,
                disagreement_gt_0_5_mps_count=len([v for v in a if v > .5]),
                disagreement_gt_2_mps_count=over2,
                receipt_age_ms=dict(selected_rows=len(ages), median=percentile(ages, .5),
                                    p95=percentile(ages, .95), maximum=max(ages) if ages else None),
                diagnostic_gates_pass=bool(n and coverage >= 95 and p95 is not None and p95 <= .5 and over2 == 0),
                independent_source_trust_established=False, physical_witness_enabled=False,
                promotion_allowed=False, production_policy_changed=False)


def stream(data, names, label):
    if len(names) != 1:
        return None, []
    try:
        value = json.loads((data/names[0]).read_text(encoding='utf-8'))
        if not isinstance(value, list):
            return None, [label+'_not_list']
        return value, []
    except (ValueError, OSError, UnicodeError):
        return None, [label+'_unreadable']


def independent_scene(data, scene, entries):
    names = {k: sorted(e['name'] for e in entries if e['stream'] == k)
             for k in ('pose', 'zoe_veh_info')}
    p, pi = stream(data, names['pose'], 'pose')
    w, wi = stream(data, names['zoe_veh_info'], 'wheel')
    issues = pi+wi
    if any(len(v) > 1 for v in names.values()):
        state = 'duplicate_alias'; issues.append('duplicate_stream_alias')
    elif any(len(v) == 0 for v in names.values()):
        state = 'missing_stream'; issues.append('missing_stream_partner')
    elif p is None or w is None:
        state = 'invalid_scene'
    else:
        state = 'audited'
        for records, label in ((p, 'pose'), (w, 'wheel')):
            if scalar_clock(records) is None:
                issues.append(label+'_invalid_clock'); state = 'invalid_scene'
    if state == 'audited':
        rows = scalar_rows(p, w)
    else:
        rows = [dict(pose_index=i, pose_utime=None, wheel_index=None, wheel_utime=None,
                     receipt_age_us=None, pose_speed_mps=None, wheel_speed_mps=None,
                     signed_disagreement_mps=None, status=state)
                for i in range(len(p) if p is not None else 0)]
    return dict(scene=scene, files=names, status=state, issues=sorted(issues),
                pose_rows=len(p) if p is not None else None,
                wheel_rows=len(w) if w is not None else None,
                summary=scalar_summary(rows)), rows


def equal(actual, expected, path='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(path+': keys differ')
        for k in expected:
            equal(actual[k], expected[k], path+'.'+k)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(path+': lengths differ')
        for i, (a, e) in enumerate(zip(actual, expected)):
            equal(a, e, path+'['+str(i)+']')
    elif isinstance(expected, float):
        if not finite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(path+': scalar disagreement')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(path+': values differ')


def verify(data, output):
    data = io.safe(data); output = io.safe(output)
    frozen = io.read(output/'input_snapshot.json')
    equal(io.snapshot(data), frozen, 'input_snapshot')
    keys = sorted(set(e['scene'] for e in frozen['files']))
    hashes = io.read(output/'result_hashes.json')
    required = {'input_snapshot.json', 'batch_summary.json'} | {s+'_pairs.jsonl.gz' for s in keys}
    if set(hashes) != required:
        raise ValueError('Incomplete or unexpected result file inventory')
    for name in sorted(required):
        if io.digest(output/name) != hashes[name]:
            raise ValueError('Result hash mismatch: '+name)
    pooled = []; scenes = []; verified = 0
    for scene in keys:
        s, rows = independent_scene(data, scene, [e for e in frozen['files'] if e['scene'] == scene])
        with gzip.open(output/(scene+'_pairs.jsonl.gz'), 'rt', encoding='utf-8') as f:
            stored = [json.loads(line) for line in f]
        equal(stored, rows, scene+'.pairs')
        scenes.append(s); pooled.extend(rows); verified += len(rows)
    unknown = sum(s['pose_rows'] is None for s in scenes)
    expected = dict(protocol_id='can-corroboration-v1', input_snapshot_sha256=io.digest(output/'input_snapshot.json'),
                    cohort='all locally available matching files; not proof of complete dataset or independent logs',
                    interpretation='CAN channel agreement only; no physical truth, held-out confirmation or policy availability claim',
                    scene_count=len(scenes), audited_scene_count=sum(s['status'] == 'audited' for s in scenes),
                    unknown_pose_denominator_scenes=unknown, pooled_denominator_complete_for_inventory=not bool(unknown),
                    inventory_diagnostic_gates_pass=bool(not unknown and all(s['status'] == 'audited' for s in scenes) and scalar_summary(pooled)['diagnostic_gates_pass']),
                    pooled=scalar_summary(pooled), scenes=scenes, scene_0001_role='development pilot',
                    independent_source_trust_established=False, physical_witness_enabled=False,
                    promotion_allowed=False, production_policy_changed=False)
    equal(io.read(output/'batch_summary.json'), expected, 'batch_summary')
    return dict(verified_pair_rows=verified, verified_scenes=len(scenes),
                input_snapshot_sha256=expected['input_snapshot_sha256'],
                method='independent scalar conversion, bisect pairing and summary reconstruction',
                promotion_allowed=False, verification_passed=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(verify(a.data_dir, a.output_dir), indent=2))
    except (ValueError, OSError) as e:
        print('VERIFICATION_FAILED: '+str(e), file=sys.stderr); sys.exit(1)
