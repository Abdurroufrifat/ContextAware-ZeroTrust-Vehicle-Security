import unittest
import numpy as np
import guard_core as c
def source(count=40):
 t=1_000_000_000+np.arange(count,dtype=np.int64)*10_000_000;v=np.zeros((count,6));v[:,2]=9.8
 return {'bag_ns':t,'header_ns':t.copy(),'usable':np.ones(count,bool),'acceleration':v[:,:3].copy(),'angular_velocity':v[:,3:].copy(),'acceleration_norm':np.full(count,9.8),'angular_velocity_norm':np.zeros(count)}
class Checks(unittest.TestCase):
 def test_budget_inclusive(self):
  np.testing.assert_array_equal(c.is_fresh(np.array([100,100]),np.array([200,201]),budget=100),[True,False])
 def test_unchanged_content_keeps_first_stamp(self):
  s=source();np.testing.assert_array_equal(c.content_stamps(s),np.full(40,s['bag_ns'][0]))
 def test_any_channel_change_renews_content(self):
  s=source(3);s['angular_velocity'][1:,2]=.1;np.testing.assert_array_equal(c.content_stamps(s),s['bag_ns'][[0,1,1]])
 def test_invalid_record_does_not_renew(self):
  s=source(3);s['usable'][1]=False;s['acceleration'][1,0]=999;np.testing.assert_array_equal(c.content_stamps(s),[s['bag_ns'][0],-1,s['bag_ns'][0]])
 def test_float_clocks_rejected(self):
  s=source();s['bag_ns']=s['bag_ns'].astype(float)
  with self.assertRaises(ValueError):c.content_stamps(s)
 def test_signed_zero_does_not_renew(self):
  s=source(2);s['acceleration'][1,0]=-0.;np.testing.assert_array_equal(c.content_stamps(s),[s['bag_ns'][0]]*2)
 def test_jitter_is_explicit_evasion(self):
  s=source();s['acceleration'][:,0]=np.arange(40)%2*1e-12;stamp=c.content_stamps(s);np.testing.assert_array_equal(stamp,s['bag_ns'])
 def test_constant_control_has_false_denials(self):
  s=source();fresh=c.is_fresh(c.content_stamps(s),s['bag_ns']);self.assertEqual(int(fresh.sum()),11)
 def test_history_recovery_waits_for_contamination_to_leave(self):
  s=source(801);s['acceleration'][21:,0]=np.arange(780)+1.;s['acceleration_norm']=np.hypot.reduce(s['acceleration'],axis=1);t=np.array([1_000_000_000,1_150_000_000,7_500_000_000,7_600_000_000],dtype=np.int64);clean,stamp=c.history_clean(s,t,c.content_stamps(s));np.testing.assert_array_equal(clean,[True,False,False,True])
 def test_guard_cannot_revive_old_denial(self):
  old={'status':np.array([4],np.uint8),'prediction_mps':np.array([np.nan])};out=c.apply_guard(old,np.array([True]));self.assertEqual(out['status'][0],4);self.assertTrue(np.isnan(out['prediction_mps'][0]))
 def test_guard_withholds_and_marks_old_grant(self):
  old={'status':np.array([0],np.uint8),'prediction_mps':np.array([1.])};out=c.apply_guard(old,np.array([False]));self.assertEqual(out['status'][0],4);self.assertTrue(out['content_guard_rejected'][0]);self.assertTrue(np.isnan(out['prediction_mps'][0]));self.assertEqual(old['status'][0],0)
 def test_empty_source(self):
  s=source(0);self.assertEqual(c.content_stamps(s).shape,(0,))
if __name__=='__main__':unittest.main()
