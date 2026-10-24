import math,unittest
import numpy as np
from radar_motion import fit_velocity,quaternion_matrix
class MotionTests(unittest.TestCase):
    def scene(self,outliers=0):
        rng=np.random.default_rng(7);rows=[];values=[];sensors=[]
        state=np.array([4.,.2,.1])
        for sensor,(x,y) in enumerate([(2.,1.),(2.,-1.),(-1.,1.),(-1.,-1.)]):
            for angle in np.linspace(-math.pi,math.pi,30,endpoint=False):
                dx,dy=math.cos(angle),math.sin(angle)
                row=[dx,dy,-dx*y+dy*x]
                rows.append(row);values.append(float(np.dot(row,state)));sensors.append(sensor)
        for _ in range(outliers):
            rows.append(rng.normal(size=3));values.append(float(rng.uniform(-20,20)));sensors.append(int(rng.integers(0,4)))
        return np.array(rows),np.array(values),np.array(sensors),state
    def test_exact_independent_radar_solution(self):
        a,b,ids,state=self.scene()
        r=fit_velocity(a,b,ids)
        self.assertTrue(r["accepted"])
        np.testing.assert_allclose(r["state"],state,atol=1e-10)
    def test_moving_object_outliers_rejected(self):
        a,b,ids,state=self.scene(40)
        r=fit_velocity(a,b,ids)
        self.assertTrue(r["accepted"])
        np.testing.assert_allclose(r["state"],state,atol=1e-6)
    def test_single_sensor_not_independent_rigid_body_geometry(self):
        a,b,ids,_=self.scene()
        self.assertFalse(fit_velocity(a,b,np.zeros(len(ids),dtype=int))["accepted"])
    def test_singular_geometry_declines(self):
        a=np.tile([1.,0.,0.],(50,1))
        self.assertFalse(fit_velocity(a,np.zeros(50),np.arange(50)%4)["accepted"])
    def test_quaternion_rotation(self):
        r=quaternion_matrix(dict(x=0.,y=0.,z=math.sin(math.pi/4),w=math.cos(math.pi/4)))
        np.testing.assert_allclose(r@np.array([1.,0.,0.]),[0.,1.,0.],atol=1e-12)
    def test_nonfinite_measurements_decline(self):
        a,b,ids,_=self.scene();b[0]=float("nan")
        self.assertFalse(fit_velocity(a,b,ids)["accepted"])
if __name__=="__main__": unittest.main()
