import unittest
from geometry import velocity_at_offset
class GeometryTests(unittest.TestCase):
    def test_pure_yaw_rear_axle(self):
        self.assertEqual(velocity_at_offset((3.,0.,0.), (0.,0.,1.), (-2.,0.,0.)), (3.,-2.,0.))
    def test_zero_offset(self):
        self.assertEqual(velocity_at_offset((1.,2.,3.), (4.,5.,6.), (0.,0.,0.)), (1.,2.,3.))
    def test_full_cross_product(self):
        self.assertEqual(velocity_at_offset((0.,0.,0.), (1.,2.,3.), (4.,5.,6.)), (-3.,6.,-3.))
if __name__ == "__main__":
    unittest.main()
