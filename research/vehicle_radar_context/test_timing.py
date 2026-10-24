import unittest
from radar_motion import select_frames
class TimingTests(unittest.TestCase):
    def frame(self,raw=0,available=0):
        return dict(sensor=21,raw_header_ns=raw,available_ns=available,corrected_header_ns=raw-106_000_000)
    def test_exact_age_boundary_is_inclusive(self):
        self.assertEqual(len(select_frames([self.frame()],100_000_000,"raw_header_association")),1)
        self.assertEqual(select_frames([self.frame()],100_000_001,"raw_header_association"),[])
    def test_corrected_age_is_not_replaced_by_transport_age(self):
        self.assertEqual(select_frames([self.frame()],0,"processed_corrected_time"),[])
    def test_future_or_incomplete_scan_unavailable(self):
        self.assertEqual(select_frames([self.frame(0,101)],100,"raw_header_association"),[])
        self.assertEqual(select_frames([self.frame(101,101)],100,"raw_header_association"),[])
    def test_only_latest_available_scan_per_sensor(self):
        rows=[self.frame(0,0),self.frame(10,10)]
        self.assertEqual(select_frames(rows,20,"raw_header_association")[0]["raw_header_ns"],10)
if __name__=="__main__": unittest.main()
