import unittest,numpy as np
import route_core as r
class RouteTests(unittest.TestCase):
 def population(self):
  groups=np.repeat(['Urban01','Urban03','Urban04'],8);y=np.tile([0.,2.,6.,12.],6);median=(y+.2)/1.6;day=np.tile([True,True,True,True,False,False,False,False],3)
  return {'y':y,'median_feature':median},groups,day
 def test_groups_disjoint_complete(self):
  d,g,day=self.population();parts=[r.split(g,k) for k in r.GROUPS]
  for train,test in parts:self.assertFalse(np.intersect1d(train,test).size);self.assertEqual(len(train)+len(test),len(g))
  np.testing.assert_array_equal(np.sort(np.concatenate([b for a,b in parts])),np.arange(24))
 def test_unknown_group_rejected(self):
  with self.assertRaises(ValueError):r.split(np.array(['Urban01','Urban02','Urban03']),'Urban01')
 def test_empty_group_rejected(self):
  with self.assertRaises(ValueError):r.split(np.array(['Urban01','Urban03']),'Urban01')
 def test_unknown_held_group_rejected(self):
  d,g,day=self.population()
  with self.assertRaises(ValueError):r.split(g,'Urban02')
 def test_routes_exact(self):
  for route in r.ROUTES:self.assertIn(r.group_of(route),r.GROUPS)
  for route in ['Urban02/2023-11-23-17-53-52-Night','Urban01/2023-11-23-18-24-14-Night','../'+r.ROUTES[0]]:
   with self.assertRaises(ValueError):r.group_of(route)
 def test_held_labels_do_not_change_anchor(self):
  d,g,day=self.population();one=r.calibrate(d,g,day,'Urban03');d2={**d,'y':d['y'].copy()};d2['y'][g=='Urban03']+=999.;two=r.calibrate(d2,g,day,'Urban03');self.assertEqual(one['anchor'],two['anchor']);np.testing.assert_array_equal(one['bin_counts'],two['bin_counts'])
 def test_training_night_labels_do_not_change_anchor(self):
  d,g,day=self.population();one=r.calibrate(d,g,day,'Urban03');d2={**d,'y':d['y'].copy()};d2['y'][~day]+=999.;two=r.calibrate(d2,g,day,'Urban03');self.assertEqual(one['anchor'],two['anchor'])
 def test_training_day_labels_can_change_anchor(self):
  d,g,day=self.population();one=r.calibrate(d,g,day,'Urban03');d2={**d,'y':d['y']*1.1};two=r.calibrate(d2,g,day,'Urban03');self.assertGreater(two['anchor']['slope'],one['anchor']['slope'])
 def test_anchor_uses_training_groups(self):
  d,g,day=self.population();cal=r.calibrate(d,g,day,'Urban03');self.assertTrue(np.all(g[cal['calibration_ids']]!='Urban03'));self.assertTrue(np.all(day[cal['calibration_ids']]));self.assertAlmostEqual(cal['anchor']['slope'],1.6)
 def test_fit_weight_counts_only_calibration_labels(self):
  d,g,day=self.population();cal=r.calibrate(d,g,day,'Urban03');self.assertEqual(cal['bin_counts'],[2,2,2,2]);np.testing.assert_allclose(r.weights(np.array([0.,2.,6.,12.]),cal),np.ones(4))
 def test_unsupported_bins_rejected(self):
  with self.assertRaises(ValueError):r.weights(np.array([12.]),{'bin_counts':[2,2,2,0],'calibration_ids':[0,1,2,3,4,5]})
 def test_duplicate_assignment_rejected(self):
  with self.assertRaises(ValueError):r.assign_membership([{'array_index':0,'role':'fit','time_ns':1,'window_id':'x','route':r.ROUTES[0]},{'array_index':0,'role':'fit','time_ns':2,'window_id':'x','route':r.ROUTES[0]}],1,0)
 def test_missing_assignment_rejected(self):
  with self.assertRaises(ValueError):r.assign_membership([],1,0)
 def test_wrong_role_rejected(self):
  with self.assertRaises(ValueError):r.assign_membership([{'array_index':0,'role':'calibration','time_ns':1,'window_id':'x','route':r.ROUTES[0]}],1,0)
 def test_audit_replays_frozen_calibration_after_numeric_refit_check(self):
  from verify_routes import validated_calibration
  saved={'anchor':{'slope':1.6,'intercept':-.2},'train_ids':[0,1]}
  rebuilt={'anchor':{'slope':float(np.nextafter(1.6,np.inf)),'intercept':-.2},'train_ids':[0,1]}
  self.assertIs(validated_calibration(rebuilt,saved),saved)
if __name__=='__main__':unittest.main()
