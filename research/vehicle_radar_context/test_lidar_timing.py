"""Point timing tests catch epoch precision loss, padding and future/stale admission."""
import struct,unittest
from types import SimpleNamespace as NS
import numpy as np
try:
    from lidar_timing import firing_times,fresh_mask
except ImportError:
    firing_times=fresh_mask=None
MODEL="unibwtas_extended_seconds_from_scan_start"
class TimingTests(unittest.TestCase):
    def cloud(self,offsets,big=False,padding=0):
        endian=">" if big else "<"
        data=b"".join(struct.pack(endian+"ff",2.,x)+b"\0"*padding for x in offsets)
        return NS(header=NS(stamp=NS(sec=1700000000,nanosec=123456789)),
            width=1,height=len(offsets),point_step=8,row_step=8+padding,is_bigendian=big,data=data,
            fields=[NS(name="distance",offset=0,datatype=7,count=1),NS(name="time",offset=4,datatype=7,count=1)])
    def ready(self):
        self.assertTrue(callable(firing_times) and callable(fresh_mask),"point timing parser missing")
    def test_epoch_arithmetic_preserves_nanoseconds(self):
        self.ready();m=self.cloud([0.,.125])
        stamps=firing_times(m,MODEL)
        self.assertEqual(stamps.tolist(),[1700000000123456789,1700000000248456789])
    def test_big_endian_and_row_padding(self):
        self.ready();stamps=firing_times(self.cloud([0.,.125],True,3),MODEL)
        self.assertEqual(int(stamps[1]-stamps[0]),125000000)
    def test_inclusive_age_boundary(self):
        self.ready();stamps=np.array([1000000000,999999999],dtype=np.int64)
        self.assertEqual(fresh_mask(stamps,1090000000,1100000000).tolist(),[True,False])
    def test_future_points_and_unreceived_cloud_rejected(self):
        self.ready();stamps=np.array([1000000001,1000000000],dtype=np.int64)
        self.assertEqual(fresh_mask(stamps,1000000000,1000000000).tolist(),[False,True])
        self.assertFalse(fresh_mask(stamps,1000000001,1000000000).any())
    def test_nonfinite_offsets_and_invalid_ranges_excluded(self):
        self.ready();m=self.cloud([float("nan"),float("inf"),0.])
        self.assertEqual(len(firing_times(m,MODEL)),1)
        m.data=struct.pack("<ff",-1.,0.)*3
        self.assertEqual(len(firing_times(m,MODEL)),0)
    def test_unknown_time_semantics_declines(self):
        self.ready()
        with self.assertRaises(ValueError): firing_times(self.cloud([0.]),"unknown")
    def test_truncated_layout_declines(self):
        self.ready();m=self.cloud([0.,0.]);m.data=m.data[:-1]
        with self.assertRaises(ValueError): firing_times(m,MODEL)
    def test_out_of_model_offsets_decline(self):
        self.ready()
        with self.assertRaises(ValueError): firing_times(self.cloud([1.]),MODEL)
if __name__=="__main__": unittest.main()
