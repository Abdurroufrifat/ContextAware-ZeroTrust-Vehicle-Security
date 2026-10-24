"""Behavior tests for source selection; every branch keeps its own eligibility."""
import unittest
import numpy as np
from fallback_core import select

def branch(status,values=None,latest=None,age=None):
    status=np.asarray(status,np.uint8);n=len(status)
    return {'status':status,'prediction_mps':np.asarray(values if values is not None else np.where(status==0,4.,np.nan),float),'latest_endpoint':np.asarray(latest if latest is not None else np.arange(n),np.int64),'age_ns':np.asarray(age if age is not None else np.zeros(n),np.int64)}

class SelectionTests(unittest.TestCase):
    def test_primary_preferred_when_both_eligible(self):
        out=select(branch([0],[8]),branch([0],[3]),True)
        self.assertEqual(out['route'].tolist(),[1]);self.assertEqual(out['prediction_mps'].tolist(),[8.])
    def test_radar_only_after_primary_denied(self):
        out=select(branch([4,2]),branch([0,0],[3,5]),True)
        self.assertEqual(out['route'].tolist(),[2,2]);self.assertEqual(out['prediction_mps'].tolist(),[3.,5.])
    def test_both_denied_remain_abstained(self):
        out=select(branch([1,2,3,4]),branch([4,3,2,1]),True)
        self.assertTrue(np.isnan(out['prediction_mps']).all());self.assertEqual(out['route'].tolist(),[0]*4)
        self.assertEqual(out['status'].tolist(),[1,2,3,4]);self.assertEqual(out['latest_endpoint'].tolist(),[-1]*4)
    def test_withhold_policy_does_not_use_radar(self):
        out=select(branch([4]),branch([0]),False)
        self.assertEqual(int(out['route'][0]),0);self.assertTrue(np.isnan(out['prediction_mps'][0]))
    def test_selected_source_provenance_preserved(self):
        out=select(branch([0,4],[8,np.nan],[7,8],[10,20]),branch([0,0],[3,5],[2,3],[30,40]),True)
        self.assertEqual(out['latest_endpoint'].tolist(),[7,3]);self.assertEqual(out['age_ns'].tolist(),[10,40])
    def test_original_rejections_are_retained(self):
        out=select(branch([4]),branch([0]),True)
        self.assertEqual(out['primary_status'].tolist(),[4]);self.assertEqual(out['radar_status'].tolist(),[0])
    def test_no_silent_shape_broadcast(self):
        with self.assertRaises(ValueError):select(branch([0,4]),branch([0]),True)
    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):select(branch([5]),branch([0]),True)
    def test_accepted_nonfinite_or_negative_values_rejected(self):
        for value in [np.nan,np.inf,-1.]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):select(branch([0],[value]),branch([0]),True)
                with self.assertRaises(ValueError):select(branch([4]),branch([0],[value]),True)
    def test_accepted_age_boundary_is_inclusive(self):
        self.assertEqual(int(select(branch([4]),branch([0],age=[100_000_000]),True)['route'][0]),2)
        for age in [-1,100_000_001]:
            with self.assertRaises(ValueError):select(branch([4]),branch([0],age=[age]),True)
    def test_accepted_source_requires_endpoint(self):
        with self.assertRaises(ValueError):select(branch([4]),branch([0],latest=[-1]),True)
    def test_denied_radar_prediction_never_selected(self):
        out=select(branch([4]),branch([2],[999.]),True)
        self.assertEqual(out['route'].tolist(),[0]);self.assertTrue(np.isnan(out['prediction_mps'][0]))
    def test_input_arrays_are_not_changed(self):
        primary=branch([4,0]);radar=branch([0,4]);snap=[{k:v.copy() for k,v in x.items()} for x in [primary,radar]]
        select(primary,radar,True)
        for original,copy in zip([primary,radar],snap):
            for key in original:np.testing.assert_array_equal(original[key],copy[key])

if __name__=='__main__':unittest.main()
