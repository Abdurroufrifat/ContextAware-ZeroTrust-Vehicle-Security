import unittest
import numpy as np
from component_core import validate_native_record

def valid():return {'kappa_plan':0.,'kappa_exec':0.,'d_t':20.,'v_t':5.,'z_t':np.full(72,20.)}

class NativeInputTests(unittest.TestCase):
    def test_complete_finite_native_inputs_are_accepted(self):
        self.assertEqual(validate_native_record(valid()),{'eligible':True,'missing':[],'invalid':[]})
    def test_missing_signals_are_not_zero_filled(self):
        r=validate_native_record({});self.assertFalse(r['eligible']);self.assertEqual(set(r['missing']),set(valid()))
    def test_speed_and_imu_aliases_do_not_supply_native_cues(self):
        r=validate_native_record({'speed_mps':5.,'angular_velocity_z':0.,'acceleration_x':0.})
        self.assertEqual(len(r['missing']),5);self.assertFalse(r['eligible'])
    def test_missing_lidar_is_unavailable(self):
        for value in [None,[],np.zeros(71),np.zeros(73)]:
            with self.subTest(value=type(value).__name__):
                x=valid();x['z_t']=value;self.assertFalse(validate_native_record(x)['eligible'])
    def test_nonfinite_native_signals_are_rejected(self):
        for key in ['kappa_plan','kappa_exec','d_t','v_t']:
            for value in [np.nan,np.inf,-np.inf]:
                x=valid();x[key]=value;self.assertIn(key,validate_native_record(x)['invalid'])
    def test_boolean_and_string_are_not_numeric_signals(self):
        for value in [True,'5']:
            x=valid();x['v_t']=value;self.assertFalse(validate_native_record(x)['eligible'])
    def test_negative_distance_or_closing_speed_rejected(self):
        for key in ['d_t','v_t']:
            x=valid();x[key]=-1.;self.assertIn(key,validate_native_record(x)['invalid'])
    def test_nonfinite_or_negative_lidar_rejected(self):
        for value in [np.nan,np.inf,-1.]:
            x=valid();x['z_t'][10]=value;self.assertIn('z_t',validate_native_record(x)['invalid'])
    def test_record_is_not_mutated(self):
        x=valid();b=x['z_t'].copy();validate_native_record(x);np.testing.assert_array_equal(x['z_t'],b)
    def test_non_mapping_rejected(self):
        self.assertFalse(validate_native_record(None)['eligible'])

if __name__=='__main__':unittest.main()
