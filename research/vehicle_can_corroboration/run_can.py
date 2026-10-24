"""Freeze input/code hashes before scoring all available CAN scenes. Stdlib only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import core

HERE = Path(__file__).resolve().parent
CODE_FILES = ('core.py', 'run_can.py', 'verify_can.py', 'protocol.json', 'test_can.py')
PATTERN = re.compile(r'scene[-_](\d{4})_(pose|zoe_veh_info)\.json\Z')
RESERVED = ('urban02', 'rav4', 'vehicle_final_evaluation')


def safe(path):
    p = Path(path)
    for spelling in (str(p.absolute()).lower(), str(p.resolve()).lower()):
        if any(x in spelling for x in RESERVED):
            raise ValueError('protected_location_refused')
    return p.resolve()


def digest(p):
    h = hashlib.sha256()
    with safe(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(safe(p).read_text(encoding='utf-8'))


def write(p, value):
    p = safe(p); tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, p)


def inventory(data):
    data = safe(data)
    if not data.is_dir():
        raise ValueError('CAN directory missing: '+str(data))
    records = []
    for p in sorted(data.iterdir(), key=lambda p: p.name):
        m = PATTERN.fullmatch(p.name)
        if not m:
            continue
        if p.is_symlink() or not p.is_file():
            raise ValueError('matching input must be an ordinary file: '+p.name)
        records.append(dict(name=p.name, scene='scene-'+m[1], stream=m[2],
                            bytes=p.stat().st_size, sha256=digest(p)))
    if not records:
        raise ValueError('No matching native CAN files in '+str(data))
    return records


def snapshot(data):
    return dict(protocol_id='can-corroboration-v1', files=inventory(data),
                code_sha256={n: digest(HERE/n) for n in CODE_FILES})


def load_stream(data, names, label):
    if len(names) != 1:
        return None, []
    try:
        rows = read(data/names[0])
        if not isinstance(rows, list):
            return None, [label+'_not_list']
        return rows, []
    except (ValueError, OSError, UnicodeError):
        return None, [label+'_unreadable']


def unavailable_rows(count, status):
    return [dict(pose_index=i, pose_utime=None, wheel_index=None, wheel_utime=None,
                 receipt_age_us=None, pose_speed_mps=None, wheel_speed_mps=None,
                 signed_disagreement_mps=None, status=status) for i in range(count)]


def scene_result(data, scene, entries):
    names = {k: sorted(e['name'] for e in entries if e['stream'] == k)
             for k in ('pose', 'zoe_veh_info')}
    p, pi = load_stream(data, names['pose'], 'pose')
    w, wi = load_stream(data, names['zoe_veh_info'], 'wheel')
    issues = pi+wi
    if any(len(v) > 1 for v in names.values()):
        status = 'duplicate_alias'; issues.append('duplicate_stream_alias')
    elif any(not v for v in names.values()):
        status = 'missing_stream'; issues.append('missing_stream_partner')
    elif p is None or w is None:
        status = 'invalid_scene'
    else:
        status = 'audited'
        for data_rows, label in ((p, 'pose'), (w, 'wheel')):
            try:
                core.check_clock(data_rows, label)
            except ValueError as e:
                issues.append(str(e)); status = 'invalid_scene'
    pairs = core.pair(p, w) if status == 'audited' else unavailable_rows(len(p) if p is not None else 0, status)
    return dict(scene=scene, files=names, status=status, issues=sorted(issues),
                pose_rows=len(p) if p is not None else None,
                wheel_rows=len(w) if w is not None else None,
                summary=core.summarize(pairs)), pairs


def build_summary(scenes, pooled, lock):
    unknown = sum(s['pose_rows'] is None for s in scenes)
    return dict(protocol_id='can-corroboration-v1', input_snapshot_sha256=lock,
                cohort='all locally available matching files; not proof of complete dataset or independent logs',
                interpretation='CAN channel agreement only; no physical truth, held-out confirmation or policy availability claim',
                scene_count=len(scenes), audited_scene_count=sum(s['status'] == 'audited' for s in scenes),
                unknown_pose_denominator_scenes=unknown,
                pooled_denominator_complete_for_inventory=not bool(unknown),
                inventory_diagnostic_gates_pass=bool(not unknown and all(s['status'] == 'audited' for s in scenes) and core.summarize(pooled)['diagnostic_gates_pass']),
                pooled=core.summarize(pooled), scenes=scenes,
                scene_0001_role='development pilot', independent_source_trust_established=False,
                physical_witness_enabled=False, promotion_allowed=False, production_policy_changed=False)


def run(data, output):
    data = safe(data); output = safe(output)
    if data == output or data in output.parents:
        raise ValueError('Output must be outside native input directory')
    frozen = snapshot(data)  # Finish the whole inventory before any speed outcome.
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output/'input_snapshot.json'
    if lock_path.exists():
        if read(lock_path) != frozen:
            raise ValueError('Frozen input/code inventory changed; refusing automatic rescore')
    else:
        if any(output.iterdir()):
            raise ValueError('Refusing nonempty output without an input lock')
        write(lock_path, frozen)
    if (output/'result_hashes.json').exists() or (output/'batch_summary.json').exists():
        import verify_can
        return verify_can.verify(data, output)
    scenes = []; pooled = []; result_files = []
    keys = sorted({e['scene'] for e in frozen['files']})
    for scene in keys:
        entry, rows = scene_result(data, scene, [e for e in frozen['files'] if e['scene'] == scene])
        scenes.append(entry); pooled.extend(rows)
        name = scene+'_pairs.jsonl.gz'
        with gzip.open(output/name, 'wt', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        result_files.append(name)
    if snapshot(data) != frozen:
        raise ValueError('Inputs/code changed during audit; results not finalized')
    summary = build_summary(scenes, pooled, digest(lock_path))
    write(output/'batch_summary.json', summary)
    result_files.extend(['input_snapshot.json', 'batch_summary.json'])
    write(output/'result_hashes.json', {n: digest(output/n) for n in result_files})
    import verify_can
    verified = verify_can.verify(data, output)
    write(output/'verification.json', verified)
    print(json.dumps(dict(scene_count=len(scenes), pose_rows=summary['pooled']['pose_rows'],
                          valid_pairs=summary['pooled']['valid_pairs'],
                          p95_abs_disagreement_mps=summary['pooled']['p95_abs_disagreement_mps'],
                          unknown_pose_denominator_scenes=summary['unknown_pose_denominator_scenes'],
                          verified_pair_rows=verified['verified_pair_rows'], promotion_allowed=False)))
    print('CAN_CORROBORATION_COMPLETE_NO_PROMOTION')
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parents[2]/'data'/'nuscenes_canbus'/'can_bus')
    parser.add_argument('--output-dir', type=Path, default=HERE/'local_batch')
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except (ValueError, OSError) as e:
        print('AUDIT_STOPPED: '+str(e), file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
