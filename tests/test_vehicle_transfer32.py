"""Quarantine enforcement and comparable endpoint/route aggregation."""
import unittest
import numpy as np
from coact_cps.vehicle.transfer32 import require_eligible,alarm_summary,aggregate


class ExpandedTransferTests(unittest.TestCase):
    def test_quarantine_rejected_even_with_valid_three_channel_data(self):
        trip=dict(id='rav31_01',role='development_pool',segments=[26,27])
        with self.assertRaises(ValueError):require_eligible(trip,26,{('rav31_02',41)})

    def test_reserved_role_is_rejected(self):
        trip=dict(id='rav31_07',role='reserved_validation',segments=[30,31])
        with self.assertRaises(ValueError):require_eligible(trip,30,{('rav31_07',30)})

    def test_segment_not_in_trip_is_rejected(self):
        trip=dict(id='rav31_02',role='development_pool',segments=[41,42])
        with self.assertRaises(ValueError):require_eligible(trip,43,{('rav31_02',43)})

    def test_exact_eligible_segment_is_allowed(self):
        require_eligible(dict(id='rav31_02',role='development_pool',segments=[41,42]),41,{('rav31_02',41)})

    def test_alarm_runs_reset_on_unscored_positions(self):
        t=np.arange(1,7,dtype=np.int64)*100;s=t-100;p=np.array([1,2,4,5])
        result=alarm_summary(np.ones(4,dtype=bool),p,t,s)
        self.assertEqual(result['longest_alarm_run_epochs'],2)
        self.assertEqual(result['alarm_endpoints'],4)

    def test_alarm_runs_reset_on_clock_gap(self):
        t=np.array([100,200,500,600]);s=np.array([0,100,400,500]);p=np.arange(4)
        self.assertEqual(alarm_summary(np.ones(4,dtype=bool),p,t,s)['longest_alarm_run_epochs'],2)

    def test_macro_routes_are_not_weighted_by_endpoint_count(self):
        rows=[dict(trip='a',segment=1,seed=7,scored_endpoints=10,alarm_endpoints=5),
              dict(trip='b',segment=1,seed=7,scored_endpoints=100,alarm_endpoints=0)]
        result=aggregate(rows)
        self.assertEqual(result['macro_alarm_fraction'],.25)
        self.assertAlmostEqual(result['pooled_model_endpoint_alarm_fraction'],5/110)

    def test_segments_pool_before_seed_mean(self):
        rows=[dict(trip='a',segment=1,seed=7,scored_endpoints=10,alarm_endpoints=10),
              dict(trip='a',segment=2,seed=7,scored_endpoints=90,alarm_endpoints=0),
              dict(trip='a',segment=1,seed=19,scored_endpoints=10,alarm_endpoints=0),
              dict(trip='a',segment=2,seed=19,scored_endpoints=90,alarm_endpoints=0)]
        self.assertEqual(aggregate(rows)['trip_alarm_fractions']['a'],.05)
        self.assertEqual(aggregate(rows)['unique_sensor_endpoints'],100)

    def test_duplicate_rows_and_unmatched_seed_coverage_fail(self):
        r=dict(trip='a',segment=1,seed=7,scored_endpoints=10,alarm_endpoints=0)
        with self.assertRaises(ValueError):aggregate([r,r])
        with self.assertRaises(ValueError):aggregate([r,dict(trip='a',segment=2,seed=19,scored_endpoints=10,alarm_endpoints=0)])

    def test_invalid_decisions_or_counts_fail(self):
        with self.assertRaises(ValueError):alarm_summary(np.array([1]),np.array([0]),np.array([1]),np.array([0]))
        with self.assertRaises(ValueError):aggregate([dict(trip='a',segment=1,seed=7,scored_endpoints=2,alarm_endpoints=3)])


if __name__=='__main__':unittest.main()
