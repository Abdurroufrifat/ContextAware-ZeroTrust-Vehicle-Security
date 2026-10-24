"""Known packet/rigid-motion fixtures; no dataset/reference fitting."""
import math,struct,unittest
import numpy as np
try:
    from stream_decoder import decode
except ImportError: decode=None
try:
    from motion import estimate
except ImportError: estimate=None
def packet():
    p=bytearray(1206)
    for k in range(12):
        struct.pack_into("<HH",p,k*100,(0xeeff,0xddff,0xccff,0xbbff)[k%4],1000+(k//4)*21)
        for j in range(32): struct.pack_into("<HB",p,k*100+4+j*3,1000,10)
    p[1204]=55;p[1205]=161
    return p
def calibration():
    return {"lasers":[dict(laser_id=i,vert_correction=0.,rot_correction=0.) for i in range(128)]}
class DecoderTests(unittest.TestCase):
    def test_known_geometry_distance_and_firing_offset(self):
        self.assertTrue(callable(decode),"decoder missing")
        r=decode(packet(),1700000000123456789,calibration())
        self.assertEqual(len(r["xyz"]),384)
        np.testing.assert_allclose(r["xyz"][0],[4*math.cos(math.radians(9.97)),-4*math.sin(math.radians(9.97)),0.],atol=1e-10)
        self.assertEqual(int(r["time_ns"][0]),1700000000123449789)
        self.assertEqual(int(r["time_ns"][128])-1700000000123456789,51333)
    def test_unrecognized_model_and_mode_decline(self):
        self.assertTrue(callable(decode),"decoder missing")
        for pos,value in [(1204,57),(1205,0)]:
            p=packet();p[pos]=value
            with self.assertRaises(ValueError): decode(p,1000000,calibration())
    def test_bad_bank_and_truncation_decline(self):
        self.assertTrue(callable(decode),"decoder missing")
        p=packet();p[0]=0
        with self.assertRaises(ValueError): decode(p,1000000,calibration())
        with self.assertRaises(ValueError): decode(packet()[:-1],1000000,calibration())
    def test_zero_returns_are_not_points(self):
        self.assertTrue(callable(decode),"decoder missing")
        p=packet()
        for k in range(12):
            for j in range(32): struct.pack_into("<H",p,k*100+4+j*3,0)
        self.assertEqual(len(decode(p,1000000,calibration())["xyz"]),0)
class MotionTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(17);p=[]
        for x in [-5.,5.]:
            p.extend(np.column_stack((np.full(800,x),rng.uniform(-8,8,800),rng.uniform(-2,2,800))))
        for y in [-8.,8.]:
            p.extend(np.column_stack((rng.uniform(-5,5,800),np.full(800,y),rng.uniform(-2,2,800))))
        return np.array(p)
    def test_translation_speed_from_static_surfaces(self):
        self.assertTrue(callable(estimate),"estimator missing")
        old=self.scene();current=old-np.array([.4,.02,0.])
        r=estimate(current,old,.1,np.eye(4))
        self.assertTrue(r["accepted"],r)
        self.assertAlmostEqual(r["speed_mps"],math.hypot(4,.2),delta=.03)
    def test_yaw_and_lever_arm_rear_axis(self):
        self.assertTrue(callable(estimate),"estimator missing")
        old=self.scene();a=.01;c,s=math.cos(a),math.sin(a);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        current=(old-np.array([.3,.01,0.]))@rot
        extrinsic=np.eye(4);extrinsic[0,3]=2.
        r=estimate(current,old,.1,extrinsic)
        self.assertTrue(r["accepted"],r)
        self.assertAlmostEqual(r["yaw_rate"],.1,delta=.02)
    def test_ground_plane_alone_has_no_planar_velocity_observability(self):
        self.assertTrue(callable(estimate),"estimator missing")
        xy=np.random.default_rng(6).uniform(-5,5,(1000,2));p=np.column_stack((xy,np.zeros(1000)))
        self.assertFalse(estimate(p,p,.1,np.eye(4))["accepted"])
    def test_missing_nonfinite_and_invalid_interval_decline(self):
        self.assertTrue(callable(estimate),"estimator missing")
        p=self.scene()
        self.assertFalse(estimate(p[:10],p,.1,np.eye(4))["accepted"])
        self.assertFalse(estimate(p,p,0.,np.eye(4))["accepted"])
        q=p.copy();q[0,0]=float("nan")
        self.assertFalse(estimate(q,p,.1,np.eye(4))["accepted"])
if __name__=="__main__": unittest.main()
