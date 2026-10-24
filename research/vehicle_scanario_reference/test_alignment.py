import math, unittest
from alignment import causal_sample, planar_speed, wheel_median
class AlignmentTests(unittest.TestCase):
    def test_no_future_receipt(self):
        self.assertIsNone(causal_sample([(101, 90, 2.)], 100, 100, 100))
    def test_no_future_sensor(self):
        self.assertIsNone(causal_sample([(99, 101, 2.)], 100, 100, 100))
    def test_staleness_inclusive(self):
        self.assertEqual(causal_sample([(90, 0, 2.)], 100, 100, 100)[2], 2.)
        self.assertIsNone(causal_sample([(90, 0, 2.)], 101, 101, 100))
    def test_latest_eligible(self):
        self.assertEqual(causal_sample([(80, 80, 1.), (90, 90, 2.), (110, 110, 3.)], 100, 100, 100)[2], 2.)
    def test_planar_norm(self):
        self.assertEqual(planar_speed(3., 4.), 5.)
        self.assertEqual(planar_speed(-3., -4.), 5.)
    def test_wheels(self):
        self.assertAlmostEqual(wheel_median([1., 2., 3., 4.]), 2.5)
    def test_invalid_data(self):
        for values in ([1., 2., 3.], [1., 2., 3., math.nan]):
            with self.assertRaises(ValueError):
                wheel_median(values)
        with self.assertRaises(ValueError):
            planar_speed(math.inf, 0.)
    def test_ordering(self):
        with self.assertRaises(ValueError):
            causal_sample([(90, 90, 1.), (80, 80, 2.)], 100, 100, 100)
if __name__ == "__main__":
    unittest.main()
