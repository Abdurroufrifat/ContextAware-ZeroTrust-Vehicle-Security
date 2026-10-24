import unittest,hashlib,struct
import numpy as np
import night_core as n
class NightTests(unittest.TestCase):
 def test_allowed_sources(self):
  for route in n.ROUTES:
   self.assertTrue(n.check_source(route,'radars.bag'))
   self.assertTrue(n.check_source(route,'reference.bag'))
 def test_reserved_and_alternate_rejected(self):
  for route in ['Urban02/2023-11-23-17-53-52-Night',n.DEVELOPMENT,'../'+n.ROUTES[0],n.ROUTES[0]+'/']:
   with self.assertRaises(ValueError):n.check_source(route,'radars.bag')
 def test_other_payloads_rejected(self):
  for name in ['gnss.bag','imus.bag','../radars.bag','cameras.bag']:
   with self.assertRaises(ValueError):n.check_source(n.ROUTES[0],name)
 def test_publisher_hash_required(self):
  b=b'bag';self.assertEqual(n.verified_bytes(b,3,hashlib.sha256(b).hexdigest()),b)
  with self.assertRaises(ValueError):n.verified_bytes(b,3,'0'*64)
  with self.assertRaises(ValueError):n.verified_bytes(b,4,hashlib.sha256(b).hexdigest())
 def test_fixed_windows(self):
  w=n.windows(n.ROUTES[0],0,1000000000000)
  self.assertEqual([r['start_ns'] for r in w],[155000000000,455000000000,755000000000])
  self.assertEqual([r['end_ns']-r['start_ns'] for r in w],[90000000000]*3)
 def test_short_recording_rejected(self):
  with self.assertRaises(ValueError):n.windows(n.ROUTES[0],0,100000000000)
 def test_future_reference_excluded(self):
  ref=n.r.ReferenceIndex([{'header_time_ns':100,'status':3,'horizontal_mps':2.,'speed_3d_mps':2.}])
  self.assertIsNone(ref.latest(99,100));self.assertIsNotNone(ref.latest(200,100));self.assertIsNone(ref.latest(201,100))
 def test_bad_reference_status(self):
  ref=n.r.ReferenceIndex([{'header_time_ns':100,'status':2,'horizontal_mps':2.,'speed_3d_mps':2.}])
  self.assertIsNone(ref.latest(100,100))
 def test_fit_only_splits(self):
  n.validate_routes(list(n.ROUTES))
  for routes in [[n.DEVELOPMENT],[n.ROUTES[0],n.DEVELOPMENT],[n.ROUTES[0],n.ROUTES[0]]]:
   with self.assertRaises(ValueError):n.validate_routes(routes)
 def test_equal_full_batch_update_budget(self):
  for size in [129,300]:
   batches=n.epoch_ids(size,5,np.random.default_rng(1));self.assertEqual(batches.shape,(5,128));self.assertTrue(np.all(batches>=0));self.assertTrue(np.all(batches<size))
 def test_epoch_no_silent_drops(self):
  ids=n.epoch_ids(300,3,np.random.default_rng(1)).reshape(-1)
  self.assertEqual(len(set(ids[:300])),300)
 def test_original_weights_shared(self):
  day=np.array([0.,2.,6.,11.,12.]);self.assertEqual(n.fixed_weights(day,np.array([11.]))[0],n.fixed_weights(day,np.array([12.]))[0])
  self.assertAlmostEqual(n.fixed_weights(day,day).mean(),1.)
 def test_missing_weight_bin_rejected(self):
  with self.assertRaises(ValueError):n.fixed_weights(np.array([0.,2.,6.]),np.array([12.]))
 def test_malformed_bag_rejected(self):
  with self.assertRaises(ValueError):n.decode(b'not a ROS bag','radars.bag')
if __name__=='__main__':unittest.main()
