import unittest
from types import SimpleNamespace as N
import numpy as np
from run_benchmark import extract
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
class SourceChecks(unittest.TestCase):
 def cloud(self):
  data=bytearray(2*(60*20+8))
  fields=[N(name=k,offset=4*i,datatype=7,count=1) for i,k in enumerate(("x","y","z","time","distance"))]
  for f in fields:
   a=np.ndarray((2,60),dtype="<f4",buffer=data,offset=f.offset,strides=(1208,20))
   a[:]=.01 if f.name=="time" else 10.
  return N(header=N(stamp=N(sec=1,nanosec=0)),width=60,height=2,point_step=20,row_step=1208,is_bigendian=False,fields=fields,data=data)
 def test_row_padding_and_original_time(self):
  msg=self.cloud()
  a=np.ndarray((2,60),dtype="<f4",buffer=msg.data,offset=12,strides=(1208,20));a[0,0]=.1;a[0,1]=0
  xyz,t,m=extract(msg,1050000000)
  self.assertEqual(xyz.shape,(119,3));self.assertEqual(m["future_points_excluded"],1)
  self.assertEqual(m["header_ns"],1000000000);self.assertEqual(m["end_ns"],1010000000)
  np.testing.assert_allclose(xyz,10)
 def test_truncated(self):
  msg=self.cloud();msg.data=msg.data[:-1]
  with self.assertRaises(ValueError):extract(msg,1050000000)
 def test_unknown_field(self):
  msg=self.cloud();msg.fields[0].datatype=6
  with self.assertRaises(ValueError):extract(msg,1050000000)
 def test_known_scene_translation(self):
  cfg=load_config(None);cfg.registration.max_num_threads=4
  model=KissICP(cfg)
  points=np.random.default_rng(413).uniform(-15,15,(15000,3))
  times=np.linspace(0,.1,len(points))
  model.register_frame(points,times)
  # Second static-world scan from a sensor translated 0.2 m.
  model.register_frame(points-np.array([.2,0,0]),times)
  np.testing.assert_allclose(model.last_pose[:3,3],[.2,0,0],atol=.003)
 def test_map_not_updated_before_registration(self):
  cfg=load_config(None);cfg.registration.max_num_threads=4;model=KissICP(cfg)
  original=model.registration.align_points_to_map;seen=[]
  def check(**kw):
   seen.append(model.local_map.empty());return original(**kw)
  model.registration.align_points_to_map=check
  points=np.random.default_rng(10).uniform(-10,10,(1000,3));t=np.linspace(0,.1,len(points))
  model.register_frame(points,t);model.register_frame(points,t)
  self.assertEqual(seen,[True,False])
if __name__=="__main__":unittest.main(verbosity=2)
