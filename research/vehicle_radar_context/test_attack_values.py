import unittest
try:
    from attack_values import craft
except ImportError:
    craft=None
class AttackTests(unittest.TestCase):
    def test_signed_bias_direction_clipping_and_scale(self):
        self.assertTrue(callable(craft),"attack constructor is missing")
        f=dict(peer=1.,yaw=.2,steering=.1,acceleration=0.)
        for case,want in [("nominal",(2.,1.)),("coordinated_plus_3",(5.,4.)),
            ("coordinated_minus_3_clipped",(0.,0.)),("coordinated_zero",(0.,0.)),("coordinated_times_2",(4.,2.))]:
            c,g=craft(case,2.,f,9.)
            self.assertEqual((c,g["peer"]),want)
            self.assertEqual(g["yaw"],.2);self.assertEqual(f["peer"],1.)
    def test_radar_match_uses_no_reference_and_missing_radar_stays_missing(self):
        self.assertTrue(callable(craft),"attack constructor is missing")
        f=dict(peer=1.,yaw=.2,steering=.1,acceleration=0.)
        c,g=craft("coordinated_radar_match",2.,f,7.125)
        self.assertEqual((c,g["peer"]),(7.125,7.125))
        self.assertEqual(craft("coordinated_radar_match",2.,f,None),(None,None))
if __name__=="__main__": unittest.main()
