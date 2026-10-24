import unittest
from data_adapter import interpolate,online_rows
class AdapterTests(unittest.TestCase):
    def test_reference_interpolation_and_no_extrapolation(self):
        values=[(0,1.),(10_000,3.)]
        self.assertEqual(interpolate(values,5_000),2.)
        self.assertIsNone(interpolate(values,-1))
        self.assertIsNone(interpolate(values,10_001))
        self.assertIsNone(interpolate([(0,1.),(50_001,3.)],20_000))
    def test_chronological_purge_and_no_reference_features(self):
        rows=[]
        for i in range(101):
            measurement=lambda value:dict(mean=value,availability=dict(data=1))
            message=dict(header=dict(stamp=dict(sec=i,nanosec=0)),
                wheel_speed=[dict(id=j,speed=measurement(2.)) for j in (0,3,4)],
                onboard_yaw_rate_esp=measurement(.1),onboard_steering_angle=measurement(.2),acceleration=measurement(.3))
            rows.append(dict(topic="actuator/vehicle/feedback",receipt_ns=i*10**9,message=message))
        spec=dict(split=dict(stride=1,train_fraction=.45,calibration_end_fraction=.70,purge_seconds=1))
        output=online_rows(rows,"example",spec)
        roles={r["index"]:r["role"] for r in output}
        self.assertEqual(roles[44],"train")
        self.assertEqual(roles[45],"purged")
        self.assertEqual(roles[46],"calibration")
        self.assertEqual(roles[70],"purged")
        self.assertEqual(roles[71],"evaluation")
        self.assertNotIn("reference",output[0])
        self.assertEqual(set(output[0]["features"]),{"peer","yaw","steering","acceleration"})
if __name__=="__main__": unittest.main()
