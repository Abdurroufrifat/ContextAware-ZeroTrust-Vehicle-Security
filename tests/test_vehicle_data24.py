import unittest

import numpy as np

from coact_cps.vehicle.data24 import CHANNELS, check_channel, select_trips


def records(days=7):
    out = []
    for day in range(1, days + 1):
        route = f'b0c9d2329ad1606b|2018-09-{day:02d}--10-00-00'
        for segment in (2, 3):
            for channel in CHANNELS:
                for kind in ('t', 'value'):
                    out.append(dict(name=f'Chunk_2/{route}/{segment}/processed_log/{channel}/{kind}'))
    return out


class DataTests(unittest.TestCase):
    def test_selection_excludes_whole_prior_date(self):
        prior = ['b0c9d2329ad1606b|2018-09-01--09-00-00']
        trips = select_trips(records(), prior)
        self.assertEqual(trips[0]['day'], '2018-09-02')
        self.assertEqual([t['role'] for t in trips], ['development'] * 3 + ['reserved_validation'] * 2)

    def test_metadata_order_does_not_change_selection(self):
        self.assertEqual(select_trips(records(), []), select_trips(records()[::-1], []))

    def test_wrong_vehicle_is_not_selected(self):
        wrong = [dict(name=r['name'].replace('b0c9d2329ad1606b', '99c94dc769b5d96e'))
                 for r in records()]
        with self.assertRaises(ValueError):
            select_trips(wrong, [])

    def test_missing_required_channel_prevents_pair(self):
        r = [x for x in records(5) if not ('2018-09-01' in x['name'] and '/gyro/' in x['name'])]
        with self.assertRaises(ValueError):
            select_trips(r, [])

    def test_duplicate_entries_are_rejected(self):
        r = records()
        with self.assertRaises(ValueError):
            select_trips(r + [r[0]], [])

    def test_finite_aligned_channel_accepts_native_times(self):
        r = check_channel(np.array([1., 2., 3.]), np.zeros((3, 3)), 3, False)
        self.assertEqual(r['samples'], 3)
        self.assertEqual(r['duplicate_timestamp_count'], 0)

    def test_wheel_ties_preserved_but_imu_ties_rejected(self):
        t = np.array([1., 1., 2.]);v = np.zeros((3, 4))
        self.assertEqual(check_channel(t, v, 4, True)['duplicate_timestamp_count'], 1)
        with self.assertRaises(ValueError):
            check_channel(t, v, 4, False)

    def test_nonfinite_sensor_value_rejected(self):
        with self.assertRaises(ValueError):
            check_channel(np.array([1., 2.]), np.array([[0., 0., 0.], [0., np.nan, 0.]]), 3, False)

    def test_time_value_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            check_channel(np.array([1., 2., 3.]), np.zeros((2, 3)), 3, False)


if __name__ == '__main__':
    unittest.main()
