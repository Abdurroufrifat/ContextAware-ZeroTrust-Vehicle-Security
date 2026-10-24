import unittest
import numpy as np
import estimator_core as e
class CausalFeatureTests(unittest.TestCase):
 def radar(self,t,v):return {'bag_time_ns':t,'header_time_ns':t,'speed_raw':v}
 def test_future_measurement_cannot_change_past_features(self):
  rows=[self.radar(i*50000000,3.) for i in range(141)];a=e.grid_features(rows,6500000000,0)
  rows[-1]['speed_raw']=999.;b=e.grid_features(rows,6500000000,0)
  np.testing.assert_array_equal(a,b)
 def test_freshness_gap_fails_instead_of_filling_unknown_motion(self):
  rows=[self.radar(0,3.),self.radar(6500000000,3.)]
  self.assertIsNone(e.grid_features(rows,6500000000,0))
 def test_warmup_is_not_credited_with_ready_history(self):
  rows=[self.radar(i*50000000,3.) for i in range(100)]
  self.assertIsNone(e.grid_features(rows,2000000000,0))
 def test_stale_header_cannot_supply_valid_current_grid(self):
  rows=[self.radar(i*50000000,3.) for i in range(141)]
  for r in rows:r['header_time_ns']-=300000000
  self.assertIsNone(e.grid_features(rows,6500000000,0))
 def test_scalar_claims_are_not_a_feature_argument(self):
  rows=[self.radar(i*50000000,-3.) for i in range(141)];x=e.grid_features(rows,6500000000,0)
  self.assertEqual(x.shape,(4,128));self.assertAlmostEqual(x[0,-1],-.3);self.assertAlmostEqual(x[1,-1],.3)
class BaselineTests(unittest.TestCase):
 def test_endpoint_savgol_reproduces_quadratic_at_current_time(self):
  t=np.arange(-50,1,dtype=float);y=2+.1*t+.003*t*t
  self.assertAlmostEqual(float(e.sg_coefficients()@y),2.,places=8)
 def test_savgol_preserves_constant(self):self.assertAlmostEqual(float(e.sg_coefficients()@np.ones(51)),1.,places=10)
 def test_fit_refuses_calibration_labels(self):
  with self.assertRaises(ValueError):e.fit_affine(np.array([1.,2.]),np.array([3.,4.]),'calibration')
 def test_speed_bins_balance_training_mass(self):
  y=np.array([0.,0.,0.,3.,6.,12.]);w=e.balanced_weights(y)
  self.assertAlmostEqual(float(w[:3].sum()),float(w[3]))
if __name__=='__main__':unittest.main()
