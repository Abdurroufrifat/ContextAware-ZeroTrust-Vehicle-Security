import tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import fault_core as c
def source():
 t=np.arange(5,dtype=np.int64)*50_000_000+1_000_000_000;a=np.arange(15.).reshape(5,3);g=a/10
 return {'bag_ns':t,'header_ns':t.copy(),'usable':np.ones(5,bool),'acceleration':a,'angular_velocity':g,'acceleration_norm':np.linalg.norm(a,axis=1),'angular_velocity_norm':np.linalg.norm(g,axis=1)}
class Checks(unittest.TestCase):
 def test_dropout_interval(self):
  s=source();o=c.mutate(s,'dropout',int(s['bag_ns'][1]),int(s['bag_ns'][3]));np.testing.assert_array_equal(o['bag_ns'],s['bag_ns'][[0,3,4]])
 def test_stale_header(self):
  s=source();o=c.mutate(s,'stale_header',int(s['bag_ns'][1]),int(s['bag_ns'][3]));np.testing.assert_array_equal(o['bag_ns'],s['bag_ns']);np.testing.assert_array_equal(o['header_ns']-s['header_ns'],[0,-150000000,-150000000,0,0])
 def test_future_header(self):
  s=source();o=c.mutate(s,'future_header',int(s['bag_ns'][1]),int(s['bag_ns'][3]));np.testing.assert_array_equal(o['header_ns']-s['header_ns'],[0,150000000,150000000,0,0])
 def test_accel_bias_norm(self):
  s=source();o=c.mutate(s,'accel_bias',int(s['bag_ns'][1]),int(s['bag_ns'][3]));a=s['acceleration'].copy();a[1:3,0]+=.5;np.testing.assert_array_equal(o['acceleration'],a);np.testing.assert_array_equal(o['angular_velocity'],s['angular_velocity']);np.testing.assert_allclose(o['acceleration_norm'],np.linalg.norm(a,axis=1))
 def test_gyro_bias(self):
  s=source();o=c.mutate(s,'gyro_bias',int(s['bag_ns'][1]),int(s['bag_ns'][3]));a=s['angular_velocity'].copy();a[1:3,2]+=.1;np.testing.assert_array_equal(o['angular_velocity'],a)
 def test_frozen_keeps_clocks(self):
  s=source();o=c.mutate(s,'frozen_values',int(s['bag_ns'][1]),int(s['bag_ns'][3]));np.testing.assert_array_equal(o['acceleration'][1:3],np.tile(s['acceleration'][0],(2,1)));np.testing.assert_array_equal(o['header_ns'],s['header_ns'])
 def test_no_future_axes(self):
  s=source();t=s['bag_ns'][2:3];a=c.axes(s,t);np.testing.assert_allclose(a,[[3,4,5,.3,.4,.5]]);s['acceleration'][-1]=999;np.testing.assert_array_equal(c.axes(s,t),a)
 def test_request_binding(self):
  q={'endpoint_ns':np.array([100]),'status':np.array([0]),'bag_ns':np.array([0]),'header_ns':np.array([0])};self.assertFalse(c.binding(np.array([101]),np.array([0]),q,ttl=100)[0])
 def test_tree_equality(self):
  m={'baseline':0.,'trees':[[{'leaf':False,'feature':0,'threshold':1.,'left':1,'right':2},{'leaf':True,'value':2.},{'leaf':True,'value':3.}]]};np.testing.assert_array_equal(c.infer(m,np.array([[1.],[2.]])),[2.,3.])
 def test_complete_archive(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'replay.npz';c.write_replay(p,{'values':np.arange(100.)})
   with zipfile.ZipFile(p) as z:self.assertIsNone(z.testzip())
   with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['values'],np.arange(100.))
 def test_failed_write_preserves_destination(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'replay.npz';p.write_bytes(b'previous')
   with self.assertRaises(ValueError):c.write_replay(p,{'bad':object()})
   self.assertEqual(p.read_bytes(),b'previous')
if __name__=='__main__':unittest.main()
