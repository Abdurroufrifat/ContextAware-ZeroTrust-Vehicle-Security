"""Adversarial protocol tests; all generated keys are local simulation keys."""
import copy
import json
import hmac
from pathlib import Path
import secrets
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import auth_guard as ag


TEST_TEMP_ROOT=Path(__file__).resolve().parent/'_temporary'
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT);self.root=Path(self.tmp.name)
        self.path=self.root/'replay.sqlite'
        self.context=ag.Context('vehicle-A','session-A','capture-A')
        self.key=secrets.token_bytes(32)
        self.rule=ag.KeyRule('wheel-A','key-A','wheel_speed',self.key)
        ag.create_state(self.path,self.context)
        self.guard=ag.Guard(self.path,self.context,[self.rule])

    def tearDown(self):self.tmp.cleanup()

    def body(self,**changes):
        b=dict(schema=1,source_id='wheel-A',key_id='key-A',vehicle_id='vehicle-A',
               session_id='session-A',capture_id='capture-A',channel='wheel_speed',
               seq=1,measured_us=1_000_000,speed_mmps=5000)
        b.update(changes);return b

    def evaluate(self,body=None,now=1_000_000,receipt=1_000_000,guard=None,key=None):
        return (guard or self.guard).verify(ag.sign(body or self.body(),key or self.key),now,receipt)

    def test_valid_report_not_physical_truth(self):
        r=self.evaluate();self.assertEqual(r['status'],'authenticated')
        self.assertTrue(r['authenticated_report']);self.assertTrue(r['mac_valid'])
        self.assertFalse(r['speed_authorized']);self.assertFalse(r['measurement_truth_established'])

    def test_payload_tampering(self):
        m=ag.sign(self.body(),self.key);m['body']['speed_mmps']+=3000
        self.assertEqual(self.guard.verify(m,1_000_000,1_000_000)['status'],'bad_tag')

    def test_retimed_timestamp_tampering(self):
        m=ag.sign(self.body(),self.key);m['body']['measured_us']+=1
        self.assertEqual(self.guard.verify(m,1_000_001,1_000_001)['status'],'bad_tag')

    def test_replay_rejected_after_restart(self):
        m=ag.sign(self.body(),self.key)
        self.assertTrue(self.guard.verify(m,1_000_000,1_000_000)['authenticated_report'])
        restarted=ag.Guard(self.path,self.context,[self.rule])
        self.assertEqual(restarted.verify(m,1_000_001,1_000_001)['status'],'replay_or_reorder')

    def test_counter_reorder_and_gap(self):
        self.assertTrue(self.evaluate(self.body(seq=3))['authenticated_report'])
        self.assertEqual(self.evaluate(self.body(seq=2))['status'],'replay_or_reorder')
        self.assertTrue(self.evaluate(self.body(seq=9))['authenticated_report'])

    def test_invalid_mac_does_not_poison_counter(self):
        m=ag.sign(self.body(seq=10000),secrets.token_bytes(32))
        self.assertEqual(self.guard.verify(m,1_000_000,1_000_000)['status'],'bad_tag')
        self.assertTrue(self.evaluate()['authenticated_report'])

    def test_freshness_exact_boundary(self):
        self.assertTrue(self.evaluate(now=1_100_000,receipt=1_000_000)['authenticated_report'])

    def test_stale_measurement_does_not_consume_sequence(self):
        r=self.evaluate(now=1_100_001,receipt=1_100_001)
        self.assertEqual(r['status'],'measurement_stale')
        self.assertTrue(self.evaluate(self.body(measured_us=1_100_001),now=1_100_001,receipt=1_100_001)['authenticated_report'])

    def test_stale_receipt(self):
        self.assertEqual(self.evaluate(now=1_100_001,receipt=1_000_000)['status'],'receipt_stale')

    def test_future_measurement(self):
        self.assertEqual(self.evaluate(self.body(measured_us=1_000_001))['status'],'invalid_clock_order')

    def test_measurement_after_receipt(self):
        self.assertEqual(self.evaluate(self.body(measured_us=1_000_001),now=1_000_002)['status'],'invalid_clock_order')

    def test_future_receipt(self):
        self.assertEqual(self.evaluate(receipt=1_000_001)['status'],'invalid_clock_order')

    def test_clock_rollback_after_restart(self):
        self.assertTrue(self.evaluate(now=1_000_002)['authenticated_report'])
        g=ag.Guard(self.path,self.context,[self.rule])
        self.assertEqual(self.evaluate(self.body(seq=2),now=1_000_001,guard=g)['status'],'clock_rollback')

    def test_context_fields_bound(self):
        for field in ['vehicle_id','session_id','capture_id']:
            self.assertEqual(self.evaluate(self.body(**{field:'other'}))['status'],'context_mismatch')
        self.assertTrue(self.evaluate()['authenticated_report'])

    def test_source_and_key_identity_bound(self):
        self.assertEqual(self.evaluate(self.body(source_id='unknown'))['status'],'unknown_key')
        self.assertEqual(self.evaluate(self.body(key_id='unknown'))['status'],'unknown_key')

    def test_channel_authorization(self):
        self.assertEqual(self.evaluate(self.body(channel='pose_speed'))['status'],'channel_denied')

    def test_revoked_key(self):
        rule=ag.KeyRule('wheel-A','key-A','wheel_speed',self.key,enabled=False)
        g=ag.Guard(self.path,self.context,[rule])
        self.assertEqual(self.evaluate(guard=g)['status'],'revoked_key')

    def test_key_rotation_does_not_reset_sequence(self):
        self.evaluate();new=secrets.token_bytes(32)
        g=ag.Guard(self.path,self.context,[self.rule,ag.KeyRule('wheel-A','key-B','wheel_speed',new)])
        self.assertEqual(self.evaluate(self.body(key_id='key-B'),guard=g,key=new)['status'],'replay_or_reorder')
        self.assertTrue(self.evaluate(self.body(key_id='key-B',seq=2),guard=g,key=new)['authenticated_report'])

    def test_distinct_source_counters(self):
        key=secrets.token_bytes(32)
        g=ag.Guard(self.path,self.context,[self.rule,ag.KeyRule('pose-A','pose-key','pose_speed',key)])
        self.assertTrue(self.evaluate(guard=g)['authenticated_report'])
        self.assertTrue(self.evaluate(self.body(source_id='pose-A',key_id='pose-key',channel='pose_speed'),guard=g,key=key)['authenticated_report'])

    def test_shared_keys_between_sources_refused(self):
        with self.assertRaises(ValueError):
            ag.Guard(self.path,self.context,[self.rule,ag.KeyRule('pose-A','pose-key','pose_speed',self.key)])

    def test_missing_state_fails_closed(self):
        self.path.unlink()
        with self.assertRaises((ValueError,OSError)):
            ag.Guard(self.path,self.context,[self.rule])
        with self.assertRaises((ValueError,OSError)):
            self.evaluate()
        self.assertFalse(self.path.exists())

    def test_initialization_never_overwrites_state(self):
        self.evaluate()
        with self.assertRaises((ValueError,OSError)):
            ag.create_state(self.path,self.context)
        self.assertEqual(self.evaluate()['status'],'replay_or_reorder')

    def test_database_bound_to_context(self):
        with self.assertRaises(ValueError):
            ag.Guard(self.path,ag.Context('vehicle-A','other','capture-A'),[self.rule])

    def test_extra_self_asserted_trust_field_rejected(self):
        with self.assertRaises(ValueError):ag.sign(self.body(independent=True),self.key)

    def test_types_and_ranges_strict(self):
        for field,value in [('seq',True),('seq',-1),('seq',1.0),('speed_mmps','5000'),('speed_mmps',float('nan')),('measured_us',-1),('seq',2**63)]:
            with self.assertRaises(ValueError):ag.sign(self.body(**{field:value}),self.key)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):ag.parse_wire('{"body":{},"body":{},"tag":"x"}')

    def test_wire_round_trip(self):
        m=ag.sign(self.body(),self.key)
        self.assertEqual(ag.parse_wire(json.dumps(m)),m)

    def test_oversized_wire_rejected(self):
        with self.assertRaises(ValueError):ag.parse_wire(' '*4097)

    def test_malformed_tag_rejected(self):
        m=ag.sign(self.body(),self.key);m['tag']='bad'
        self.assertEqual(self.guard.verify(m,1_000_000,1_000_000)['status'],'malformed')

    def test_compromised_signer_can_authenticate_false_measurement(self):
        r=self.evaluate(self.body(speed_mmps=8000))
        self.assertTrue(r['authenticated_report']);self.assertFalse(r['speed_authorized'])

    def test_database_snapshot_rollback_is_explicit_limit(self):
        backup=self.root/'old.sqlite';shutil.copyfile(self.path,backup)
        self.evaluate();shutil.copyfile(backup,self.path)
        # No external monotonic anchor exists: restoring the state can re-admit it.
        self.assertTrue(self.evaluate()['authenticated_report'])

    def test_atomic_concurrent_duplicate_rejection(self):
        m=ag.sign(self.body(),self.key)
        def submit(_):return ag.Guard(self.path,self.context,[self.rule]).verify(m,1_000_000,1_000_000)['authenticated_report']
        with ThreadPoolExecutor(max_workers=4) as executor:
            results=list(executor.map(submit,range(4)))
        self.assertEqual(sum(results),1)

    def test_independent_wire_encoding_and_mac(self):
        b=self.body();m=ag.sign(b,self.key)
        # Independently fix the exact bytes; do not call the production canonicalizer.
        ordered=[1,'wheel-A','key-A','vehicle-A','session-A','capture-A','wheel_speed',1,1000000,5000]
        wire=b'COACTCPS-EVIDENCE-HMAC-SHA256-v1'+bytes([0])+json.dumps(ordered,separators=(',',':')).encode('ascii')
        self.assertEqual(m['tag'],hmac.digest(self.key,wire,'sha256').hex())

    def test_unsigned_dataset_record_rejected(self):
        r=self.guard.verify({'utime':1000000,'vel':[5,0,0]},1000000,1000000)
        self.assertEqual(r['status'],'malformed');self.assertFalse(r['speed_authorized'])

    def test_deeply_nested_wire_rejected(self):
        old=sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(500)
            with self.assertRaises(ValueError):ag.parse_wire('['*600+']'*600)
        finally:
            sys.setrecursionlimit(old)

    def test_decoder_recursion_failure_is_rejected(self):
        with patch('auth_guard.json.loads',side_effect=RecursionError('decoder nesting limit')):
            with self.assertRaises(ValueError):ag.parse_wire('{}')

    def test_protected_state_path_refused(self):
        with self.assertRaises(ValueError):ag.create_state(self.root/'vehicle_final_evaluation'/'state.sqlite',self.context)


if __name__=='__main__':unittest.main(verbosity=2)
