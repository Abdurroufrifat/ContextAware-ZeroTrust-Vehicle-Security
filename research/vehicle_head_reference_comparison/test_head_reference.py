import unittest
import numpy as np
from head_reference_core import eligibility

def inputs():return [np.array([5.]),np.array([5.]),np.array([True]),np.array([0],np.int64),np.array([True])]
class EligibilityTests(unittest.TestCase):
    def test_valid_endpoint_is_eligible(self):self.assertEqual(eligibility(*inputs()).tolist(),[True])
    def test_source_rejection_is_not_revived(self):
        a=inputs();a[2][0]=False;self.assertEqual(eligibility(*a).tolist(),[False])
    def test_invalid_imu_source_is_not_revived(self):
        for status in [1,2,3,4]:
            a=inputs();a[3][0]=status;self.assertEqual(eligibility(*a).tolist(),[False])
    def test_dirty_history_is_not_revived(self):
        a=inputs();a[4][0]=False;self.assertEqual(eligibility(*a).tolist(),[False])
    def test_peer_limit_is_inclusive(self):
        a=inputs();a[0][0]=6.;self.assertTrue(eligibility(*a)[0]);a[0][0]=6.00000001;self.assertFalse(eligibility(*a)[0])
    def test_nonfinite_predictions_and_peer_are_rejected(self):
        for field in [0,1]:
            for value in [np.nan,np.inf,-np.inf]:
                a=inputs();a[field][0]=value;self.assertFalse(eligibility(*a)[0])
    def test_negative_speeds_are_rejected(self):
        for field in [0,1]:
            a=inputs();a[field][0]=-1.;self.assertFalse(eligibility(*a)[0])
    def test_misaligned_rows_are_rejected(self):
        a=inputs();a[1]=np.zeros(2)
        with self.assertRaises(ValueError):eligibility(*a)
    def test_boolean_masks_are_not_silently_coerced(self):
        a=inputs();a[2]=np.array([1])
        with self.assertRaises(ValueError):eligibility(*a)
    def test_inputs_are_not_changed(self):
        a=inputs();copies=[v.copy() for v in a];eligibility(*a)
        for x,y in zip(a,copies):np.testing.assert_array_equal(x,y)

if __name__=='__main__':unittest.main()
