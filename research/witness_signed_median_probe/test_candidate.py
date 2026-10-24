import unittest
from base_loader import baseline
import candidate_core as candidate

def radar(t,speed=2,stamp=None):return {'bag_time_ns':t,'header_time_ns':t if stamp is None else stamp,'speed_raw':speed}
def obd(t,speed=7.2):return {**radar(t,speed),'unit':'kph'}

class CandidateTests(unittest.TestCase):
    def test_symmetric_stationary_noise_does_not_become_speed_magnitude(self):
        data=[radar(i*40000000,2 if i%2 else -2) for i in range(26)]
        result=candidate.signed_features(data,0)
        self.assertEqual(result[25],0)
    def test_constant_signed_motion_retains_magnitude(self):
        for speed in (3,-3):
            data=[radar(i*40000000,speed) for i in range(26)]
            self.assertEqual(candidate.signed_features(data,0)[25],3)
    def test_future_records_cannot_change_an_earlier_feature(self):
        data=[radar(i*40000000,2 if i%2 else -2) for i in range(31)]
        a=candidate.signed_features(data,0)
        b=candidate.signed_features(data+[radar(1240000000,1000)],0)
        self.assertEqual(a,{i:b[i] for i in a})
    def test_stale_or_future_header_never_creates_a_feature(self):
        for delay in (-1,300000000):
            data=[radar(i*40000000,2,stamp=i*40000000-delay) for i in range(31)]
            self.assertEqual(candidate.signed_features(data,0),{})
    def test_invalid_current_measurement_cannot_borrow_good_past_history(self):
        data=[radar(i*40000000) for i in range(31)];data[-1]['speed_raw']=float('nan')
        self.assertNotIn(30,candidate.signed_features(data,0))
    def test_pair_endpoints_and_coverage_match_the_frozen_baseline(self):
        r=[radar(i*40000000,2 if i%2 else -2) for i in range(31)];o=[obd(i*40000000) for i in range(31)]
        a,sa=baseline.causal_pairs(r,o,0);b,sb=candidate.paired_records(r,o,0)
        self.assertEqual(sa,sb);self.assertEqual(len(a),len(b))
        self.assertEqual([{k:v for k,v in p.items() if k!='feature_raw'} for p in a],[{k:v for k,v in p.items() if k!='feature_raw'} for p in b])
    def test_obd_values_do_not_enter_radar_features(self):
        r=[radar(i*40000000,2 if i%2 else -2) for i in range(31)]
        a,_=candidate.paired_records(r,[obd(i*40000000,7.2) for i in range(31)],0)
        b,_=candidate.paired_records(r,[obd(i*40000000,720) for i in range(31)],0)
        self.assertEqual([p['feature_raw'] for p in a],[p['feature_raw'] for p in b])
        self.assertNotEqual(a[0]['label_mps'],b[0]['label_mps'])
    def test_ambiguous_duplicate_event_times_fail_closed(self):
        data=[radar(i*40000000) for i in range(31)];data.append(radar(1200000000,99))
        with self.assertRaises(ValueError):candidate.signed_features(data,0)

if __name__=='__main__':unittest.main()
