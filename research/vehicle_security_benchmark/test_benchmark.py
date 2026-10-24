"""Behavioral regressions for benchmark coverage and honest attack accounting."""
import unittest
import benchmark_core as b

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.p = {'profile_id': 'p010', 'speed_mps': 1, 'acceleration_mps2': 2, 'yaw_rate_rps': -0.1}

    def test_profile_grid_is_complete(self):
        self.assertEqual(len(b.profiles()), 36)
        self.assertEqual(len({tuple(p.values()) for p in b.profiles()}), 36)

    def test_all_sources_receive_each_fault(self):
        names = {c['id'] for c in b.ingress_catalog()}
        for source in b.SOURCES:
            for fault in ('missing', 'stale', 'future', 'replay', 'tamper', 'bad_type'):
                self.assertIn(f'{fault}__{source}', names)

    def test_partial_replays_block_after_valid_admission(self):
        for source in b.SOURCES:
            row = b.ingress_trial(self.p, f'replay__{source}', 'full_policy')
            self.assertFalse(row['granted'])
            self.assertTrue(any('replay' in x for x in row['reasons']))

    def test_benign_context_is_allowed_on_all_profiles(self):
        for profile in b.profiles():
            self.assertEqual(b.ingress_trial(profile, 'benign', 'full_policy')['outcome'], 'ALLOW')

    def test_common_compromised_speed_is_counted_as_attack_success(self):
        row = b.ingress_trial(self.p, 'common_speed_7', 'full_policy')
        self.assertTrue(row['granted'])
        self.assertTrue(row['attack_success'])
        self.assertTrue(row['contract_passed'])

    def test_restart_replay_stays_visible(self):
        row = b.ingress_trial(self.p, 'restart_replay', 'full_policy')
        self.assertTrue(row['attack_success'])

    def test_snapshot_is_a_grant(self):
        row = b.ingress_trial(self.p, 'moderate_read', 'full_policy')
        self.assertEqual(row['outcome'], 'RESTRICT')
        self.assertTrue(row['granted'])

    def test_source_boundary_contracts(self):
        for case in ('ttl_at_limit', 'skew_at_limit'):
            self.assertTrue(b.ingress_trial(self.p, case, 'full_policy')['granted'])
        for case in ('ttl_over_limit', 'skew_over_limit', 'clock_rollback'):
            self.assertFalse(b.ingress_trial(self.p, case, 'full_policy')['granted'])

    def test_ablations_reveal_independent_layers(self):
        self.assertTrue(b.ingress_trial(self.p, 'tamper__gnss', 'acl_only')['granted'])
        self.assertFalse(b.ingress_trial(self.p, 'tamper__gnss', 'authenticated_context_only')['granted'])
        self.assertTrue(b.ingress_trial(self.p, 'single_speed_7', 'authenticated_context_only')['granted'])
        self.assertFalse(b.ingress_trial(self.p, 'single_speed_7', 'full_policy')['granted'])

    def test_acl_only_ignores_source_transport_and_claims(self):
        for case in ('malformed', 'revoked', 'failed_attestation', 'bad_type__identity'):
            self.assertTrue(b.ingress_trial(self.p, case, 'acl_only')['granted'])

    def test_context_ablation_preserves_clock_rollback_protection(self):
        self.assertFalse(b.ingress_trial(self.p, 'clock_rollback', 'authenticated_context_only')['granted'])

    def test_capability_expiry_is_exclusive(self):
        self.assertTrue(b.capability_trial(self.p, 'lease_249')['authorized'])
        self.assertFalse(b.capability_trial(self.p, 'lease_250')['authorized'])

    def test_capability_nonce_reuse_and_restart(self):
        self.assertFalse(b.capability_trial(self.p, 'token_reuse')['authorized'])
        self.assertTrue(b.capability_trial(self.p, 'restart_token_reuse')['attack_success'])

    def test_capability_downgrade_and_no_upgrade(self):
        for case in ('downgrade', 'snapshot_no_upgrade'):
            row = b.capability_trial(self.p, case)
            self.assertTrue(row['authorized'])
            self.assertEqual(row['mode'], 'snapshot')

    def test_revocation_revalidated_at_consumption(self):
        self.assertFalse(b.capability_trial(self.p, 'revoked')['authorized'])

    def test_results_are_reproducible(self):
        self.assertEqual(b.ingress_trial(self.p, 'replay__gnss', 'full_policy'), b.ingress_trial(self.p, 'replay__gnss', 'full_policy'))
        self.assertEqual(b.capability_trial(self.p, 'fresh'), b.capability_trial(self.p, 'fresh'))

    def test_reducer_keeps_denominator_and_failed_security(self):
        rows = [b.ingress_trial(self.p, 'common_speed_7', 'full_policy'), b.ingress_trial(self.p, 'single_speed_7', 'full_policy')]
        result = b.reduce_rows(rows)
        self.assertEqual(result['trials'], 2)
        self.assertEqual(result['attack_trials'], 2)
        self.assertEqual(result['attack_grants'], 1)
        self.assertEqual(result['attack_grant_fraction'], 0.5)
        self.assertTrue(result['all_contracts_passed'])

    def test_empty_group_does_not_invent_zero_rate(self):
        self.assertIsNone(b.reduce_rows([])['attack_grant_fraction'])

    def test_verifier_rejects_missing_or_duplicate_trial(self):
        import verify_benchmark as v
        rows = [b.ingress_trial(self.p, 'benign', 'full_policy')]
        expected = {('p010', 'benign', 'full_policy')}
        v.validate_rows(rows, expected, {c['id']: c for c in b.ingress_catalog()}, False)
        for altered in ([], rows + rows):
            with self.assertRaises(ValueError):
                v.validate_rows(altered, expected, {c['id']: c for c in b.ingress_catalog()}, False)

    def test_verifier_rejects_hidden_attack_grant(self):
        import verify_benchmark as v
        row = b.ingress_trial(self.p, 'common_speed_7', 'full_policy')
        row['attack_success'] = False
        with self.assertRaises(ValueError):
            v.validate_rows([row], {('p010', 'common_speed_7', 'full_policy')}, {c['id']: c for c in b.ingress_catalog()}, False)

    def test_integrity_check_never_opens_protected_measurements(self):
        import verify_benchmark as v
        import hashlib, json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'safe.py').write_bytes(b'safe')
            protected = ['research/vehicle_final_evaluation/packets.jsonl.gz', 'data/Urban02_night.npz', 'data/RAV4.npz']
            manifest = {'safe.py': hashlib.sha256(b'safe').hexdigest(), **{name: 'closed' for name in protected}}
            (root / 'prior_snapshot_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            original = v.sha
            def guarded(path):
                if path.name != 'safe.py':
                    raise AssertionError('protected measurement opened')
                return original(path)
            with patch.object(b, 'ROOT', root), patch.object(b, 'PROJECT', root), patch.object(v, 'sha', guarded):
                self.assertEqual(v.verify_prior(True), (1, 0, 3))

if __name__ == '__main__':
    unittest.main()
