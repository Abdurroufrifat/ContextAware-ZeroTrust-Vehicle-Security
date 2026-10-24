"""Equal-sensor regression: returning point-count RANSAC must fail dense clutter test."""
import math,unittest
import numpy as np
try:
    from radar_balanced import fit_velocity
except ImportError:
    from radar_motion import fit_velocity
class BalancedTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(502)
        rows=[];values=[];ids=[]
        true=np.array([8.,.1,.04]);wrong=np.array([16.,-.5,.1])
        for sensor in range(6):
            n=250 if sensor<3 else 25
            for i in range(n):
                x,y=(2. if sensor%2 else -1.),(1. if sensor<3 else -1.)
                # Exclude near-perpendicular rays where different physical velocities
                # have indistinguishable Doppler; exact-state expectation needs this.
                while True:
                    angle=rng.uniform(-math.pi,math.pi);dx,dy=np.cos(angle),np.sin(angle)
                    row=np.array([dx,dy,-dx*y+dy*x])
                    if abs(row@(true-wrong))>.75: break
                state=wrong if sensor<3 and i<150 else true
                rows.append(row);values.append(float(row@state));ids.append(sensor)
        return np.array(rows),np.array(values),np.array(ids),true
    def test_dense_moving_sensors_do_not_outvote_static_sensor_evidence(self):
        a,b,s,true=self.scene();r=fit_velocity(a,b,s)
        self.assertTrue(r["accepted"])
        np.testing.assert_allclose(r["state"],true,atol=1e-6)
    def test_sensor_renaming_changes_no_physical_solution(self):
        a,b,s,true=self.scene();s=np.array([83,9,41,7,55,32])[s]
        r=fit_velocity(a,b,s);self.assertTrue(r["accepted"])
        np.testing.assert_allclose(r["state"],true,atol=1e-6)
    def test_missing_sensor_geometry_declines(self):
        a,b,s,_=self.scene()
        self.assertFalse(fit_velocity(a,b,np.zeros(len(s),dtype=int))["accepted"])
    def test_nonfinite_declines(self):
        a,b,s,_=self.scene();b[0]=float("nan")
        self.assertFalse(fit_velocity(a,b,s)["accepted"])
if __name__=="__main__": unittest.main()
