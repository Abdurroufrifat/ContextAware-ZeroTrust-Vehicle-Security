import math,unittest
import numpy as np
try:
    from backend_motion import estimate
except ImportError: estimate=None
class BackendTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(17);p=[]
        for x in [-5.,5.]:
            p.extend(np.column_stack((np.full(700,x),rng.uniform(-8,8,700),rng.uniform(-2,2,700))))
        for y in [-8.,8.]:
            p.extend(np.column_stack((rng.uniform(-5,5,700),np.full(700,y),rng.uniform(-2,2,700))))
        p.extend(np.column_stack((rng.uniform(-5,5,700),rng.uniform(-8,8,700),np.full(700,-2.))))
        return np.array(p)
    def test_source_to_target_direction_and_velocity_units(self):
        self.assertTrue(callable(estimate),"backend adapter missing")
        old=self.scene();new=old-[.4,.02,0.]
        for method in ("ICP","PLANE_ICP","GICP"):
            r=estimate(new,old,.1,np.eye(4),method)
            self.assertTrue(r["accepted"],(method,r))
            self.assertAlmostEqual(r["speed_mps"],math.hypot(4,.2),delta=.08)
    def test_yaw_and_rear_axis_lever_arm(self):
        self.assertTrue(callable(estimate),"backend adapter missing")
        old=self.scene();a=.01;c,s=math.cos(a),math.sin(a)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        new=(old-[.3,.01,0.])@rot
        mount=np.eye(4);mount[0,3]=2.
        r=estimate(new,old,.1,mount,"GICP")
        self.assertTrue(r["accepted"],r)
        self.assertAlmostEqual(r["yaw_rate"],.1,delta=.02)
        self.assertAlmostEqual(r["rear_pose"][1][3],.01-2*math.sin(.01),delta=.005)
    def test_ground_only_does_not_certify_planar_speed(self):
        self.assertTrue(callable(estimate),"backend adapter missing")
        xy=np.random.default_rng(9).uniform(-5,5,(1000,2));p=np.column_stack((xy,np.zeros(1000)))
        self.assertFalse(estimate(p,p,.1,np.eye(4),"GICP")["accepted"])
    def test_missing_nonfinite_invalid_interval_and_unknown_method_decline(self):
        self.assertTrue(callable(estimate),"backend adapter missing")
        p=self.scene();bad=p.copy();bad[0,0]=float("nan")
        for a,b,dt,method in [(p[:10],p,.1,"GICP"),(bad,p,.1,"GICP"),(p,p,0.,"GICP"),(p,p,.1,"unknown")]:
            self.assertFalse(estimate(a,b,dt,np.eye(4),method)["accepted"])
if __name__=="__main__": unittest.main()
