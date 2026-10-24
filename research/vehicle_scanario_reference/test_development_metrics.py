import math, unittest
import development_metrics as metrics
class DevelopmentMetricsTests(unittest.TestCase):
    def test_coverage_counts_abstentions(self):
        fn=getattr(metrics,"summarize",None)
        self.assertTrue(callable(fn),"Missing descriptive metric implementation")
        result=fn([1.,None,3.,None],[1.,2.,3.,4.])
        self.assertEqual(result["coverage_pct"],50.)
        self.assertFalse(result["empirical_discrepancy_gate"])
        self.assertIsNone(result["physical_gate_pass"])
    def test_boundary_conditions(self):
        fn=getattr(metrics,"summarize",None)
        self.assertTrue(callable(fn))
        self.assertTrue(fn([.5]*20,[0.]*20)["empirical_discrepancy_gate"])
        self.assertFalse(fn([.500001]*20,[0.]*20)["empirical_discrepancy_gate"])
        self.assertEqual(fn([2.000001],[0.])["discrepancy_above_2_mps"],1)
    def test_no_accepted_measurements(self):
        fn=getattr(metrics,"summarize",None)
        self.assertTrue(callable(fn))
        result=fn([None],[1.])
        self.assertIsNone(result["p95_discrepancy_mps"])
        self.assertFalse(result["empirical_discrepancy_gate"])
    def test_nonfinite_and_shape_rejected(self):
        fn=getattr(metrics,"summarize",None)
        self.assertTrue(callable(fn))
        for a,b in [([math.nan],[1.]),([1.],[math.inf]),([1.],[1.,2.])]:
            with self.assertRaises(ValueError): fn(a,b)
    def test_rear_wheels_ignore_mean_and_front(self):
        fn=getattr(metrics,"rear_wheel_speed",None)
        self.assertTrue(callable(fn))
        wheels=[dict(id=i,speed=dict(mean=v,availability=dict(data=1))) for i,v in [(0,99.),(1,8.),(2,9.),(3,2.),(4,4.)]]
        self.assertEqual(fn(dict(wheel_speed=wheels)),3.)
        wheels[-1]["speed"]["availability"]["data"]=0
        self.assertIsNone(fn(dict(wheel_speed=wheels)))
    def test_duplicate_wheel_ids_rejected(self):
        fn=getattr(metrics,"rear_wheel_speed",None)
        self.assertTrue(callable(fn))
        rows=[dict(id=i,speed=dict(mean=2.,availability=dict(data=1))) for i in [3,3,4]]
        with self.assertRaises(ValueError): fn(dict(wheel_speed=rows))
    def test_fixed_strata_boundaries(self):
        fn=getattr(metrics,"bin_name",None)
        self.assertTrue(callable(fn))
        self.assertEqual(fn(2.,[0,2,5,15,1000]),"[2,5)")
        self.assertEqual(fn(.05,[0,.05,.15,1000]),"[0.05,0.15)")
if __name__=="__main__": unittest.main()
