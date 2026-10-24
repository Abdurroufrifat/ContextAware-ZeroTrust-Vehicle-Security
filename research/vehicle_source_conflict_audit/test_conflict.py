import unittest
import conflict_audit as m

class Tests(unittest.TestCase):
    def test_strict_threshold_and_contiguous_runs(self):
        self.assertEqual(m.runs([False,True,True,False,True]),[(1,2),(4,4)])
        self.assertEqual(m.runs([]),[])

    def test_equal_values_with_new_timestamps(self):
        self.assertEqual(m.longest_equal([1.,2.,2.,2.,3.],[0,100000,200000,300000,400000]),
                         {'first_row':1,'last_row':3,'rows':3,'elapsed_us':200000,'speed_mps':2.})

    def test_single_and_empty_run(self):
        self.assertIsNone(m.longest_equal([],[]))
        self.assertEqual(m.longest_equal([0.],[123])['elapsed_us'],0)

    def test_internal_position_check_is_only_a_diagnostic(self):
        self.assertAlmostEqual(m.surface_distance(0,0,0,0),0.)
        self.assertAlmostEqual(m.surface_distance(0,0,0,1),111195.0802335329,places=6)

    def test_no_claim_of_truth_from_consensus(self):
        self.assertFalse(m.CLAIMS['independent_physical_truth_established'])
        self.assertFalse(m.CLAIMS['gnss_fault_proven'])
        self.assertFalse(m.CLAIMS['policy_changed'])

    def test_fresh_receipt_can_accompany_large_source_conflict(self):
        c=m.describe(28.5,26.,5000,[28.4,28.5,28.5,28.6])
        self.assertEqual(c['signed_wheel_minus_gnss_mps'],2.5)
        self.assertTrue(c['recorded_age_gate_pass'])
        self.assertTrue(c['large_gnss_disagreement'])
        self.assertAlmostEqual(c['four_wheel_spread_mps'],.2)

    def test_exact_boundaries(self):
        self.assertFalse(m.describe(4.,2.,100000,[4.]*4)['large_gnss_disagreement'])
        self.assertTrue(m.describe(4.,2.,100000,[4.]*4)['recorded_age_gate_pass'])
        self.assertFalse(m.describe(4.,2.,100001,[4.]*4)['recorded_age_gate_pass'])

if __name__=='__main__':unittest.main()
