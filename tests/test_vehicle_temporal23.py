import unittest

import numpy as np

from coact_cps.vehicle.temporal23 import runs, summarize


class TemporalTests(unittest.TestCase):
    @staticmethod
    def evaluate(labels, alarm, grant, positions=None, base=None, gap=False):
        n = len(labels)
        t = np.arange(1, n + 1, dtype=np.int64) * 200
        start = t - 200
        if gap:
            t[2:] += 200
            start[2:] += 200
        positions = np.arange(n) if positions is None else np.array(positions)
        base = np.ones(n, dtype=bool) if base is None else np.array(base, dtype=bool)
        return summarize(
            t,
            start,
            np.array(labels, dtype=bool),
            positions,
            np.array(alarm, dtype=bool),
            base,
            np.array(grant, dtype=bool),
        )

    def test_missing_interval_splits_streaks(self):
        mask = np.ones(4, dtype=bool)
        self.assertEqual(
            runs(mask, np.array([0, 200, 600, 800]), np.array([200, 400, 800, 1000])),
            [(0, 2), (2, 4)],
        )

    def test_delay_uses_event_start_and_decision_endpoint(self):
        r = self.evaluate([False, True, True], [False, False, True], [True, True, False])
        self.assertEqual(r["events"][0]["first_neural_alarm_delay_ms"], 400)
        self.assertEqual(r["events"][0]["first_denial_delay_ms"], 400)

    def test_missed_event_delay_is_null(self):
        r = self.evaluate([True, True], [False, False], [True, True])
        self.assertEqual(r["neural_missed_events"], 1)
        self.assertIsNone(r["events"][0]["first_neural_alarm_delay_ms"])
        self.assertEqual(r["attack_granted_observed_span_ms"], 400)

    def test_startup_denial_does_not_count_as_neural_alarm(self):
        r = self.evaluate([True, True, True], [False], [False, False, True], positions=[2])
        self.assertEqual(r["unscored_epochs"], 2)
        self.assertEqual(r["events"][0]["first_denial_delay_ms"], 200)
        self.assertIsNone(r["events"][0]["first_neural_alarm_delay_ms"])
        self.assertEqual(r["events"][0]["granted_epochs_after_first_denial"], 1)

    def test_first_denial_does_not_hide_later_grants(self):
        r = self.evaluate([True, True, True], [True, False, True], [False, True, False])
        self.assertEqual(r["events"][0]["granted_epochs_after_first_denial"], 1)
        self.assertFalse(r["events"][0]["fully_denied"])

    def test_normal_alarm_bursts_break_at_clear_epoch(self):
        r = self.evaluate([False] * 4, [True, True, False, True], [False, False, True, False])
        self.assertEqual(r["normal_neural_alarm_bursts"], 2)
        self.assertEqual(r["normal_neural_longest_alarm_run_epochs"], 2)
        self.assertEqual(r["normal_longest_denial_run_epochs"], 2)

    def test_gaps_split_attack_events(self):
        r = self.evaluate([True] * 4, [False] * 4, [True] * 4, gap=True)
        self.assertEqual(r["attack_events"], 2)
        self.assertEqual(r["attack_longest_grant_run_epochs"], 2)

    def test_no_attack_has_no_events(self):
        r = self.evaluate([False], [False], [True])
        self.assertEqual(r["events"], [])
        self.assertEqual(r["attack_events"], 0)

    def test_saved_grant_must_match_base_and_alarm(self):
        with self.assertRaisesRegex(ValueError, "grant"):
            self.evaluate([True], [True], [True])

    def test_base_denial_is_preserved(self):
        r = self.evaluate([True], [False], [False], base=[False])
        self.assertTrue(r["events"][0]["fully_denied"])

    def test_overlapping_intervals_rejected(self):
        with self.assertRaises(ValueError):
            summarize(
                np.array([200, 300]),
                np.array([0, 100]),
                np.zeros(2, dtype=bool),
                np.arange(2),
                np.zeros(2, dtype=bool),
                np.ones(2, dtype=bool),
                np.ones(2, dtype=bool),
            )

    def test_duplicate_scored_positions_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate([False, False], [False, False], [True, False], positions=[0, 0])


if __name__ == "__main__":
    unittest.main()
