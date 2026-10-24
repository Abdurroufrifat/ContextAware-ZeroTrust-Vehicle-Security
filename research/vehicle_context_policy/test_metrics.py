import math,unittest
from run_experiment import metrics
class MetricsTests(unittest.TestCase):
    def test_unknown_reference_is_not_a_zero_error(self):
        result=metrics([dict(accept=True,reference=None,value=1.,active=False,reason="example")])
        self.assertEqual(result["unscorable_accepted"],1)
        self.assertIsNone(result["empirical_all_request_gate"])
    def test_p95_rounding_cannot_admit_an_exceeded_boundary(self):
        values=[.5]*19+[math.nextafter(.5,1.)]
        rows=[dict(accept=True,reference=0.,value=v,active=False,reason="example") for v in values]
        self.assertFalse(metrics(rows)["empirical_all_request_gate"])
if __name__=="__main__": unittest.main()
