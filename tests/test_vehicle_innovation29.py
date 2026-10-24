"""Causal physics, observability limits, and normal-only geometry fitting."""
import unittest
import numpy as np
from coact_cps.vehicle.innovation29 import innovations,fit_geometry,score


def motion():
    time=np.arange(1,13,dtype=np.int64)*100
    start=time-100
    speed=5+2*time/1000
    return np.column_stack((speed,speed,np.full(12,2.))),time,start


class InnovationTests(unittest.TestCase):
    def test_constant_acceleration_is_consistent(self):
        x,t,s=motion();result=innovations(x,t,s)
        self.assertTrue(np.isnan(result[:5]).all())
        np.testing.assert_allclose(result[5:],0,atol=1e-13)

    def test_integrates_irregular_right_endpoint_intervals(self):
        t=np.array([100,220,410,600,710,900],dtype=np.int64);s=np.r_[0,t[:-1]]
        a=np.arange(6,dtype=float);v=np.cumsum(a*(t-s)/1000)
        x=np.column_stack((v,v,a))
        np.testing.assert_allclose(innovations(x,t,s)[5],0,atol=1e-13)

    def test_signed_single_sensor_bias_is_retained(self):
        x,t,s=motion();x[:,0]+=3
        np.testing.assert_allclose(innovations(x,t,s)[5:],np.tile([3,0,0],(7,1)),atol=1e-13)

    def test_common_speed_offset_is_unobservable(self):
        x,t,s=motion();shift=x.copy();shift[:,:2]+=17
        np.testing.assert_allclose(innovations(x,t,s),innovations(shift,t,s),equal_nan=True,atol=1e-12)

    def test_coherent_speed_ramp_and_imu_bias_are_unobservable(self):
        x,t,s=motion();shift=x.copy();shift[:,:2]+=1.7*t[:,None]/1000;shift[:,2]+=1.7
        np.testing.assert_allclose(innovations(x,t,s),innovations(shift,t,s),equal_nan=True,atol=1e-12)

    def test_future_values_do_not_change_past_features(self):
        x,t,s=motion();changed=x.copy();changed[9:]+=100
        np.testing.assert_array_equal(innovations(x,t,s)[:9],innovations(changed,t,s)[:9])

    def test_gap_restarts_history(self):
        x,t,s=motion();t[6:]+=1000;s[6:]+=1000
        result=innovations(x,t,s)
        self.assertTrue(np.isnan(result[6:11]).all())
        self.assertTrue(np.isfinite(result[11]).all())

    def test_rejects_invalid_interval_or_timestamp_type(self):
        x,t,s=motion()
        for bad in (t.astype(float),t[::-1]):
            with self.assertRaises(ValueError):innovations(x,bad,s)
        bad=s.copy();bad[3]=t[3]
        with self.assertRaises(ValueError):innovations(x,t,bad)
        bad=s.copy();bad[3]=t[3]-501
        with self.assertRaises(ValueError):innovations(x,t,bad)

    def test_rejects_nonfinite_features(self):
        x,t,s=motion();x[0,0]=np.nan
        with self.assertRaises(ValueError):innovations(x,t,s)

    def test_regularization_handles_collinear_constant_data(self):
        profile=fit_geometry(np.ones((6,3)),np.zeros(6,dtype=bool))
        np.testing.assert_allclose(profile['scale'],.01)
        np.testing.assert_allclose(profile['covariance'],.1*np.eye(3))
        np.testing.assert_array_equal(score(np.ones((2,3)),profile,'innovation_covariance'),[0,0])

    def test_geometry_matches_direct_quadratic_form(self):
        x=np.array([[0,1,2],[1,3,1],[3,2,4],[2,0,3]],dtype=float)
        profile=fit_geometry(x,np.zeros(4,dtype=bool));z=(x-profile['median'])/profile['scale']
        c=.9*(z.T@z/4)+.1*np.eye(3)
        np.testing.assert_allclose(profile['covariance'],c)
        np.testing.assert_allclose(score(x,profile,'innovation_covariance'),np.sqrt(np.einsum('ij,jk,ik->i',z,np.linalg.inv(c),z)))
        np.testing.assert_allclose(score(x,profile,'innovation_max'),np.max(abs(z),axis=1))

    def test_fitting_rejects_attack_rows(self):
        with self.assertRaises(ValueError):fit_geometry(np.ones((3,3)),np.array([False,True,False]))
        with self.assertRaises(ValueError):fit_geometry(np.ones((3,3)),np.zeros(3,dtype=int))

    def test_scoring_rejects_invalid_or_unavailable_rows(self):
        profile=fit_geometry(np.ones((3,3)),np.zeros(3,dtype=bool))
        with self.assertRaises(ValueError):score([[np.nan,0,0]],profile,'innovation_max')
        with self.assertRaises(ValueError):score([[0,0,0]],profile,'unknown')
        profile['scale'][0]=0
        with self.assertRaises(ValueError):score([[0,0,0]],profile,'innovation_max')


if __name__=='__main__':unittest.main()
