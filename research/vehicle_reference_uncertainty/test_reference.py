import math,unittest
from fractions import Fraction
import reference_bounds as m

class ReferenceTests(unittest.TestCase):
    def test_endpoint_bounds_contain_every_synthetic_truth(self):
        for output in (0.,1.,3.,8.):
            for center in (0.,2.,6.):
                for u in (0.,.25,1.,3.):
                    lo,hi=m.error_interval(output,center,u)
                    for k in range(21):
                        truth=center-u+2*u*k/20
                        error=abs(output-truth)
                        self.assertGreaterEqual(error+1e-12,float(lo));self.assertLessEqual(error,float(hi)+1e-12)

    def test_unknown_bound_does_not_become_zero(self):
        self.assertIsNone(m.error_interval(1.,1.,None))

    def test_invalid_bounds_and_nonfinite_values(self):
        for u in (-1,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError):m.error_interval(1.,1.,u)
        with self.assertRaises(ValueError):m.error_interval(float('inf'),1.,0.)

    def test_position_accuracy_is_not_speed_accuracy(self):
        self.assertEqual(m.position_difference_budget(Fraction(3,10),Fraction(3,10),Fraction(1,5)),Fraction(3))
        with self.assertRaises(ValueError):m.position_difference_budget(.3,.3,0.)

    def test_fractional_boundary_is_exact(self):
        self.assertEqual(m.error_interval(Fraction(1,4),0,Fraction(1,4))[1],Fraction(1,2))

    def test_rounding_cannot_admit_a_bound_above_half(self):
        u=Fraction(1,2)+Fraction(1,10**30)
        self.assertFalse(m.summarize([(0.,0.,u)])['conditional_numeric_gates_pass'])

    def test_missing_reference_preserves_accepted_denominator(self):
        r=m.summarize([(1.,1.,.1),(1.,None,.1),(None,None,None)])
        self.assertEqual(r['requests'],3);self.assertEqual(r['accepted'],2)
        self.assertEqual(r['accepted_with_qualified_numeric_bound'],1)
        self.assertFalse(r['conditional_numeric_gates_pass'])
        self.assertIsNone(r['p95_upper_absolute_error_mps'])

    def test_zero_disagreement_still_fails_if_uncertainty_too_large(self):
        r=m.summarize([(5.,5.,.6)])
        self.assertFalse(r['conditional_numeric_gates_pass'])

    def test_numerical_success_is_not_physical_appraisal(self):
        r=m.summarize([(5.,5.,.25)])
        self.assertTrue(r['conditional_numeric_gates_pass'])
        self.assertFalse(r['physical_reference_appraised'])
        self.assertFalse(r['promotion_allowed'])

    def test_possible_and_unavoidable_large_errors_differ(self):
        r=m.summarize([(2.,0.,.25),(2.5,0.,.25)])
        self.assertEqual(r['possible_error_gt2_count'],2)
        self.assertEqual(r['unavoidable_error_gt2_count'],1)

    def test_point_two_is_not_over_large_error_threshold(self):
        self.assertEqual(m.summarize([(2.,0.,0.)])['possible_error_gt2_count'],0)

    def test_p95_interval_order(self):
        r=m.summarize([(float(i),0.,.1) for i in range(100)])
        self.assertAlmostEqual(r['p95_lower_absolute_error_mps'],93.95)
        self.assertAlmostEqual(r['p95_upper_absolute_error_mps'],94.15)

if __name__=='__main__':unittest.main()
