"""Protocol boundary tests: removing each check must fail its named fixture."""
import json
import math
import tempfile
import unittest
from pathlib import Path
import core
import run_can
import verify_can


def pose(t, v=1):
    return {'utime': t, 'vel': [v, 0, 0]}


def wheel(t, rpm=60):
    return dict(utime=t, **{k: rpm for k in core.WHEELS})


class PairTests(unittest.TestCase):
    def test_rpm_units(self):
        self.assertAlmostEqual(core.wheel_speed(wheel(0)), 2*math.pi*.305)

    def test_reverse_and_median(self):
        w = wheel(0)
        w.update(dict(zip(core.WHEELS, [-60, 120, -180, 240])))
        self.assertAlmostEqual(core.wheel_speed(w), 2.5*2*math.pi*.305)

    def test_future_not_used(self):
        r = core.pair([pose(99)], [wheel(100)])
        self.assertEqual(r[0]['status'], 'no_past_wheel')
        self.assertIsNone(r[0]['wheel_index'])

    def test_absolute_clock_not_separately_normalized(self):
        r = core.pair([pose(200_000)], [wheel(0)])
        self.assertEqual(r[0]['status'], 'stale_receipt')

    def test_age_boundary(self):
        r = core.pair([pose(100_000), pose(100_001)], [wheel(0)])
        self.assertEqual([x['status'] for x in r], ['valid', 'stale_receipt'])

    def test_latest_including_equal(self):
        r = core.pair([pose(10)], [wheel(0), wheel(10)])
        self.assertEqual(r[0]['wheel_index'], 1)
        self.assertEqual(r[0]['receipt_age_us'], 0)

    def test_no_fallback_past_invalid_latest(self):
        w = wheel(10); w[core.WHEELS[0]] = '60'
        r = core.pair([pose(10)], [wheel(0), w])
        self.assertEqual(r[0]['status'], 'invalid_latest_wheel')
        self.assertEqual(r[0]['wheel_index'], 1)

    def test_invalid_numeric_fields(self):
        for v in [True, '60', None, float('nan'), float('inf')]:
            w = wheel(0); w[core.WHEELS[1]] = v
            self.assertIsNone(core.wheel_speed(w))

    def test_pose_horizontal_and_denominator(self):
        p = pose(1); p['vel'] = [3, 4, 99]
        bad = pose(2); bad['vel'] = [True, 0, 0]
        r = core.pair([p, bad], [wheel(0)])
        self.assertEqual(r[0]['pose_speed_mps'], 5)
        self.assertEqual(r[1]['status'], 'invalid_pose')
        self.assertEqual(core.summarize(r)['pose_rows'], 2)
        self.assertEqual(core.summarize(r)['valid_pairs'], 1)

    def test_clock_reject_no_sort(self):
        for ts in [[1, 1], [2, 1], [True, 2], [1.0, 2], ['1', 2]]:
            with self.assertRaises(ValueError):
                core.pair([pose(t) for t in ts], [wheel(0)])

    def test_empty_and_single_quantile(self):
        self.assertIsNone(core.summarize([])['p95_abs_disagreement_mps'])
        self.assertEqual(core.quantile([2], .95), 2)
        self.assertAlmostEqual(core.quantile([0, 10], .95), 9.5)

    def test_agreement_gates_never_promote(self):
        s = core.summarize(core.pair([pose(0, 0)], [wheel(0, 0)]))
        self.assertTrue(s['diagnostic_gates_pass'])
        self.assertFalse(s['promotion_allowed'])
        self.assertFalse(s['independent_source_trust_established'])
        self.assertFalse(s['physical_witness_enabled'])

    def test_invalid_pose_precedence(self):
        p = pose(0); p['vel'] = []
        self.assertEqual(core.pair([p], [wheel(1)])[0]['status'], 'invalid_pose')


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/'can_bus'; self.data.mkdir()
        self.out = self.root/'audit'

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        (self.data/name).write_text(json.dumps(value), encoding='utf-8')

    def fixture(self):
        self.write('scene-0001_pose.json', [pose(100_000), pose(100_001)])
        self.write('scene-0001_zoe_veh_info.json', [wheel(0)])

    def test_missing_partner_retained(self):
        self.write('scene-0001_pose.json', [pose(0), pose(1)])
        run_can.run(self.data, self.out)
        s = json.loads((self.out/'batch_summary.json').read_text())
        self.assertEqual(s['scenes'][0]['status'], 'missing_stream')
        self.assertEqual(s['pooled']['pose_rows'], 2)
        self.assertEqual(s['pooled']['valid_pairs'], 0)

    def test_duplicate_alias_invalid(self):
        self.fixture(); self.write('scene_0001_pose.json', [pose(0)])
        run_can.run(self.data, self.out)
        s = json.loads((self.out/'batch_summary.json').read_text())
        self.assertEqual(s['scenes'][0]['status'], 'duplicate_alias')
        self.assertEqual(s['unknown_pose_denominator_scenes'], 1)

    def test_bad_clock_retained(self):
        self.fixture(); self.write('scene-0001_pose.json', [pose(2), pose(1)])
        run_can.run(self.data, self.out)
        s = json.loads((self.out/'batch_summary.json').read_text())
        self.assertEqual(s['scenes'][0]['status'], 'invalid_scene')
        self.assertEqual(s['pooled']['pose_rows'], 2)

    def test_scalar_verifier_and_repeat(self):
        self.fixture(); run_can.run(self.data, self.out)
        self.assertEqual(verify_can.verify(self.data, self.out)['verified_pair_rows'], 2)
        before = (self.out/'batch_summary.json').read_bytes()
        run_can.run(self.data, self.out)
        self.assertEqual(before, (self.out/'batch_summary.json').read_bytes())

    def test_input_tamper_rejected(self):
        self.fixture(); run_can.run(self.data, self.out)
        self.write('scene-0001_pose.json', [pose(0)])
        with self.assertRaises(ValueError):
            run_can.run(self.data, self.out)

    def test_result_tamper_rejected(self):
        self.fixture(); run_can.run(self.data, self.out)
        p = self.out/'batch_summary.json'
        v = json.loads(p.read_text()); v['pooled']['valid_pairs'] = 100
        p.write_text(json.dumps(v))
        with self.assertRaises(ValueError):
            verify_can.verify(self.data, self.out)

    def test_new_scene_after_lock_rejected(self):
        self.fixture(); run_can.run(self.data, self.out)
        self.write('scene-0002_pose.json', [pose(0)])
        with self.assertRaises(ValueError):
            run_can.run(self.data, self.out)

    def test_protected_location_rejected_before_read(self):
        d = self.root/'vehicle_final_evaluation'; d.mkdir()
        with self.assertRaises(ValueError):
            run_can.inventory(d)

    def test_independent_verifier_detects_forgery_with_updated_hash(self):
        self.fixture(); run_can.run(self.data, self.out)
        p = self.out/'batch_summary.json'
        v = json.loads(p.read_text()); v['pooled']['valid_pairs'] = 100
        p.write_text(json.dumps(v))
        hashes = run_can.read(self.out/'result_hashes.json')
        hashes['batch_summary.json'] = run_can.digest(p)
        run_can.write(self.out/'result_hashes.json', hashes)
        with self.assertRaises(ValueError):
            verify_can.verify(self.data, self.out)

    def test_unknown_denominator_cannot_pass_inventory_gates(self):
        self.write('scene-0001_zoe_veh_info.json', [wheel(0)])
        run_can.run(self.data, self.out)
        s = run_can.read(self.out/'batch_summary.json')
        self.assertEqual(s['unknown_pose_denominator_scenes'], 1)
        self.assertFalse(s['inventory_diagnostic_gates_pass'])

    def test_wheel_duplicate_keeps_readable_pose_denominator(self):
        self.fixture(); self.write('scene_0001_zoe_veh_info.json', [wheel(0)])
        run_can.run(self.data, self.out)
        s = run_can.read(self.out/'batch_summary.json')
        self.assertEqual(s['pooled']['pose_rows'], 2)
        self.assertEqual(s['pooled']['valid_pairs'], 0)
        self.assertFalse(s['inventory_diagnostic_gates_pass'])

    def test_malformed_pose_json_unknown_denominator(self):
        self.fixture(); (self.data/'scene-0001_pose.json').write_text('{broken')
        run_can.run(self.data, self.out)
        s = run_can.read(self.out/'batch_summary.json')
        self.assertEqual(s['scenes'][0]['status'], 'invalid_scene')
        self.assertEqual(s['unknown_pose_denominator_scenes'], 1)

    def test_empty_directory_rejected(self):
        with self.assertRaises(ValueError):
            run_can.run(self.data, self.out)

    def test_symlink_rejected(self):
        target = self.root/'outside.json'; target.write_text('[]')
        (self.data/'scene-0001_pose.json').symlink_to(target)
        with self.assertRaises(ValueError):
            run_can.inventory(self.data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
