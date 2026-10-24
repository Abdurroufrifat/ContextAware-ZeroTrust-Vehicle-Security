"""Boundary checks for the one normal-coverage intervention."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    import run_vehicle_coverage_phase2 as r
    return r.load_protocol(ROOT / r.PROTOCOL_PATH)


def test_only_the_proposed_pair_of_normal_roles_changes():
    import run_vehicle_coverage_phase2 as r
    p = protocol()
    original = json.loads((ROOT / r.PARENT_PROTOCOL_PATH).read_text())
    r.validate_protocol(p, original)
    changed = [(a['trip'], a['segment']) for a, b in zip(
        p['development_segments'], original['development_segments'], strict=True)
        if a['role'] != b['role']]
    assert changed == [('rav24_02', 4), ('rav24_02', 5), ('rav31_04', 7), ('rav31_04', 8)]
    assert p['development_roles']['development_evaluate'] == ['rav24_03', 'rav31_06']
    assert p['intervention']['whole_split_is_chronological'] is False


@pytest.mark.parametrize('field', ['training', 'scoring', 'promotion_gates', 'additional_stress'])
def test_other_recipe_changes_are_rejected(field):
    import run_vehicle_coverage_phase2 as r
    p = deepcopy(protocol())
    original = json.loads((ROOT / r.PARENT_PROTOCOL_PATH).read_text())
    p[field]['unauthorized_change'] = True
    with pytest.raises(ValueError, match='Only the frozen normal-role exchange'):
        r.validate_protocol(p, original)


def test_locked_route_injection_is_rejected():
    import run_vehicle_coverage_phase2 as r
    p = deepcopy(protocol())
    original = json.loads((ROOT / r.PARENT_PROTOCOL_PATH).read_text())
    p['development_segments'][0]['trip'] = 'rav31_07'
    with pytest.raises(ValueError):
        r.validate_protocol(p, original)


def test_serialized_protocol_is_pinned_before_payload_access(tmp_path):
    import run_vehicle_coverage_phase2 as r
    p = protocol()
    p['development_roles']['fit'].append('rav31_07')
    target = tmp_path / 'changed.json'
    target.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='coverage protocol pin'):
        r.load_protocol(target)


def test_coverage_auditor_binds_the_new_protocol_not_only_parent(tmp_path):
    import run_vehicle_coverage_phase2 as r
    import audit_vehicle_coverage_phase2 as a
    out = tmp_path / 'out'
    out.mkdir()
    source = ROOT / r.PROTOCOL_PATH
    (out / 'coverage_protocol.json').write_bytes(source.read_bytes())
    lock = {'phase1_protocol_sha256': r.PARENT_PROTOCOL_SHA,
            'coverage_protocol_sha256': r.PROTOCOL_SHA}
    assert a.verify_lock_protocol(ROOT, out, lock)['development_roles'] == protocol()['development_roles']
    lock['coverage_protocol_sha256'] = r.PARENT_PROTOCOL_SHA
    with pytest.raises(ValueError, match='coverage protocol binding'):
        a.verify_lock_protocol(ROOT, out, lock)


def test_fixed_streams_are_paired_but_new_pgd_streams_are_separate():
    import compare_vehicle_coverage_phase2 as c
    assert c.cross_run_pairable('joint_jump') is True
    assert c.cross_run_pairable('normal') is True
    assert c.cross_run_pairable('pgd_joint_history_gru6') is False
