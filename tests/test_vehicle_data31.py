"""Metadata exclusions and exact range bounds protect locked payloads."""
import unittest
import struct
from coact_cps.vehicle.data31 import select_trips,payload_end,channel_report
import numpy as np

CHANNELS=('GNSS/live_gnss_ublox','CAN/wheel_speed','IMU/accelerometer','IMU/gyro')


def records(days=10):
    result=[]
    for day in range(1,days+1):
        route=f'b0c9d2329ad1606b|2018-10-{day:02d}--10-00-00'
        for segment in (3,4):
            for channel in CHANNELS:
                for kind in ('t','value'):result.append({'name':f'Chunk_2/{route}/{segment}/processed_log/{channel}/{kind}'})
    return result


class DataSelectionTests(unittest.TestCase):
    def test_selects_six_development_and_two_later_reservations(self):
        trips=select_trips(records(),[])
        self.assertEqual(len(trips),8)
        self.assertEqual([t['role'] for t in trips],['development_pool']*6+['reserved_validation']*2)
        self.assertEqual(trips[0]['segments'],[3,4])

    def test_excludes_whole_prior_day_including_other_routes(self):
        trips=select_trips(records(),['b0c9d2329ad1606b|2018-10-01--09-00-00'])
        self.assertEqual(trips[0]['day'],'2018-10-02')

    def test_sorting_is_deterministic(self):
        self.assertEqual(select_trips(records(),[]),select_trips(list(reversed(records())),[]))

    def test_incomplete_channel_pair_is_skipped(self):
        r=records();r.pop(0)
        self.assertEqual(select_trips(r,[])[0]['day'],'2018-10-02')

    def test_duplicate_entries_fail(self):
        r=records();r.append(r[0])
        with self.assertRaises(ValueError):select_trips(r,[])

    def test_requires_eight_distinct_dates(self):
        with self.assertRaises(ValueError):select_trips(records(7),[])

    def test_ignores_other_vehicle_and_unsafe_segment(self):
        r=records();r.extend([{'name':'Chunk_2/99c94dc769b5d96e|2018-01-01--00-00-00/1/processed_log/IMU/gyro/t'},
            {'name':'Chunk_2/b0c9d2329ad1606b|2018-01-01--00-00-00/../processed_log/IMU/gyro/t'}])
        self.assertEqual(select_trips(r,[]),select_trips(records(),[]))

    def test_exact_payload_end_has_no_speculative_overread(self):
        header=struct.pack('<4s5H3L2H',b'PK\x03\x04',20,0,8,0,0,1,50,100,12,7)
        self.assertEqual(payload_end(header,1000,50),1000+30+12+7+50-1)

    def test_rejects_invalid_local_header(self):
        with self.assertRaises(ValueError):payload_end(b'x'*30,0,50)
        with self.assertRaises(ValueError):payload_end(b'PK\x03\x04',0,50)

    def test_duplicate_gyro_times_are_quarantined_without_sorting(self):
        time=np.array([0.,0.,1.,1.]);values=np.zeros((4,3));before=time.copy()
        result=channel_report(time,values,3,False)
        self.assertEqual(result['status'],'QUARANTINED')
        self.assertEqual(result['duplicate_timestamp_count'],2)
        np.testing.assert_array_equal(time,before)

    def test_wheel_receipt_ties_remain_permitted(self):
        result=channel_report([0.,0.,1.],np.zeros((3,4)),4,True)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['duplicate_timestamp_count'],1)

    def test_reversed_time_is_quarantined(self):
        self.assertEqual(channel_report([2.,1.],np.zeros((2,3)),3,False)['status'],'QUARANTINED')

    def test_valid_strict_channel_passes(self):
        self.assertEqual(channel_report([0.,1.],np.zeros((2,3)),3,False)['status'],'PASS')


if __name__=='__main__':unittest.main()
