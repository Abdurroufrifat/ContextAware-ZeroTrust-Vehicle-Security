"""Compare two development experiments without pairing different PGD streams."""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import run_vehicle_coverage_phase2 as recipe
from audit_vehicle_joint_phase2 import read_npz, independent_features


def cross_run_pairable(case):
    # Each model produces its own white-box stream. Within-run gates remain paired.
    return not case.startswith('pgd_')


def fit_speed_support(root, protocol):
    values = []
    for row in protocol['development_segments']:
        if row['role'] != 'fit':
            continue
        c = read_npz(root / 'artifacts/vehicle_transfer_stage32/contexts' /
                     f"{row['trip']}__{row['segment']}.npz")
        joint, positions = independent_features(c['features'], c['time_ms'], c['start_ms'])
        union = sorted({k for i in positions for k in range(int(i)-31, int(i)+1)})
        values.extend(joint[union, 0])
    return [float(min(values)), float(max(values))]


def digest_summary(folder):
    s = json.loads((folder / 'summary.json').read_text())
    lock = json.loads((folder / 'model_lock.json').read_text())
    failures = s['attack']['failed_cells']
    thresholds = {str(seed): lock['thresholds'][f'joint_history_gru6__{seed}']
                  for seed in (7, 19, 43)}
    tolerances = {}
    for seed in (7, 19, 43):
        p = lock['profiles'][f'joint_history_gru6__{seed}']
        tolerance = (np.asarray(p['median']) + thresholds[str(seed)] * np.asarray(p['scale']))
        tolerances[str(seed)] = (tolerance * np.asarray(lock['scaler']['scale'])).tolist()
    return dict(status=s['status'], primary_gates_pass=s['primary_gates_pass'],
                rows=s['rows_observed'], errors=s['errors'], normal=s['normal'],
                attack_gate_cells=s['attack']['checked_cells'],
                attack_failed_cells=len(failures),
                attack_failures_by_reference=dict(Counter(v['reference'] for v in failures)),
                attack_failure_reasons=dict(Counter(r for v in failures for r in v['reasons'])),
                joint_thresholds=thresholds,
                joint_forecast_absolute_residual_tolerance_native_units=tolerances)


def compare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = recipe.verify_protocol_family(root)
    before = root / 'artifacts/vehicle_joint_phase2'
    lock = json.loads((output / 'model_lock.json').read_text())
    if lock['coverage_protocol_sha256'] != recipe.PROTOCOL_SHA:
        raise ValueError('coverage lock mismatch')
    original = json.loads((before / 'summary.json').read_text())
    current = json.loads((output / 'summary.json').read_text())
    if any(current[k] for k in ('reserved_validation_accessed', 'stage16_final_routes_accessed', 'eligible_for_validation', 'active_policy_changed')):
        raise ValueError('unexpected promotion or validation access')
    context_count = reference_count = 0
    for name in sorted({r['context'] for r in original['rows'] if cross_run_pairable(r['case'])}):
        a, b = read_npz(before / 'contexts' / name), read_npz(output / 'contexts' / name)
        if a.keys() != b.keys():
            raise ValueError('fixed stream schema differs')
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
        context_count += 1
    for row in original['rows']:
        if row['method'] != 'original_nn_frozen' or not cross_run_pairable(row['case']):
            continue
        a, b = read_npz(before / 'blocks' / row['block']), read_npz(output / 'blocks' / row['block'])
        if a.keys() != b.keys():
            raise ValueError('original-reference block schema differs')
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
        reference_count += 1
    parent = json.loads((root / recipe.PARENT_PROTOCOL_PATH).read_text())
    return dict(schema='coact-coverage-comparison-v1',
                before=digest_summary(before), after=digest_summary(output),
                fit_corrected_gnss_support_before=fit_speed_support(root, parent),
                fit_corrected_gnss_support_after=fit_speed_support(root, protocol),
                fixed_normal_and_attack_streams_exact=context_count,
                original_neural_fixed_stream_blocks_exact=reference_count,
                cross_run_pgd_streams_paired=False,
                pgd_comparison='Each run reoptimizes its own attacks; all arms within each run see identical saved streams. Cross-run PGD outcomes are descriptive, not a paired treatment effect.',
                interpretation='Previously inspected development evidence only. The intervention changes the fitting cohort, scaler, learned weights/profiles and calibrated thresholds through the same fixed recipe. It does not isolate their individual causal effects.',
                architecture_or_hyperparameter_search=False,
                whole_split_is_chronological=False, validation_accessed=False,
                eligible_for_validation=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    p.add_argument('--save', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.root, a.output or a.root / 'artifacts/vehicle_coverage_phase2')
    recipe.write_json(a.save, result)
    print(json.dumps(result, indent=2))
