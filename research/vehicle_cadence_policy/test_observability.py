import math,unittest
from observability import construct,check
class ObservabilityTests(unittest.TestCase):
 def test_one_mps_attack_can_share_observation_with_benign_world(self):
  x=construct();o=x['observation']
  self.assertEqual(o['gnss_mps'],x['benign_true_mps'])
  self.assertEqual(o['gnss_mps']-x['attacked_true_mps'],1.)
 def test_both_worlds_obey_the_same_witness_error_bound(self):
  x=construct();p=x['observation']['radar_prediction_mps']
  for key in ('benign_true_mps','attacked_true_mps'):self.assertLessEqual(abs(p-x[key]),x['epsilon_mps'])
 def test_constructed_worlds_have_identical_policy_permissions(self):
  x=check({'centers':[0,0,0,0],'allowance_mps':.1,'cusum_limit':5.,'hard_limit':2.5,'continuity_gap_ns':300000000})
  self.assertTrue(x['identical_permission_sequences']);self.assertGreater(x['granted_in_both_worlds'],0);self.assertEqual(x['injected_shift_mps'],1.)
 def test_invalid_physical_examples_rejected(self):
  for p,e in ((10,float('nan')),(10,-1),(.1,.5),(float('inf'),.5)):
   with self.assertRaises(ValueError):construct(p,e)
