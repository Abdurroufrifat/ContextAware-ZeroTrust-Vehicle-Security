"""Verify the design archive; does not train or evaluate CO-ACT-CPS."""
from pathlib import Path
import hashlib
import json

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['sha256'].items():
        path = root / relative
        if not path.is_file():
            raise SystemExit('FAIL: missing ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit('FAIL: changed ' + relative)
    p = json.loads((root / 'PHASE2_PROTOCOL.json').read_text(encoding='utf-8'))
    roles = p['development_roles']
    flat = [x for values in roles.values() for x in values]
    assert len(flat) == len(set(flat)) == 8, 'route roles overlap or missing'
    assert [len(roles[k]) for k in ('fit','calibrate','development_evaluate')] == [4,2,2]
    locked = {r['trip'] for r in p['locked_rav4_metadata_only']}
    quarantined = {r['trip'] for r in p['quarantine']}
    assert not set(flat) & (locked | quarantined), 'forbidden route assigned'
    assert len(p['development_segments']) == 16
    assert len({(r['trip'],r['segment']) for r in p['development_segments']}) == 16
    for row in p['development_segments']:
        assert row['trip'] in roles[row['role']]
        assert row['previously_inspected'] is True
    assert p['allow_locked_payload_access_in_phase2'] is False
    assert p['training']['attack_labels_used_for_fit'] is False
    assert p['training']['hyperparameter_search'] is False
    assert p['status'] == 'DESIGN_ONLY_NOT_IMPLEMENTED_NOT_PROMOTED'
    assert p['promotion_gates']['normal_alarm_worst_route_max'] == 0.05
    assert p['promotion_gates']['relaxation_allowed'] is False
    sources = json.loads((root / 'PUBLISHED_METHOD_COMPARISON.json').read_text(encoding='utf-8'))['sources']
    assert len(sources) == 6 and all(s['url'].startswith('https://') for s in sources)
    print('PASS — package hashes and Phase 1 protocol consistency.')
    print('Design only: no model training, security evaluation or validation access.')

if __name__ == '__main__':
    main()
