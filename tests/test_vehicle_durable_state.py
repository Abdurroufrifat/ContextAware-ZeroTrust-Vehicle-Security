"""Actual durable policy behavior; no data or model imports."""
import json, os, shutil, sqlite3, subprocess, sys, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
from coact_cps.vehicle.durable_policy import DurableStateError, DurableVehiclePolicy
from coact_cps.vehicle.policy import PolicyConfig, VehiclePolicyEngine
from test_vehicle_zero_trust import KEYS, make_frames, make_request

CAP=b'capability-test-key-distinct-32bytes'
ACL={('fleet-service','av-1','vehicle.telemetry','read'),('fleet-service','av-1','vehicle.planner','update_route')}

class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'replay state.sqlite'
        self.e=DurableVehiclePolicy.provision(self.path,KEYS,CAP,ACL)
        self.req=make_request()
    def tearDown(self): self.temp.cleanup()
    def reopen(self,**kw):
        return DurableVehiclePolicy(self.path,kw.get('keys',KEYS),kw.get('cap',CAP),kw.get('acl',ACL),kw.get('config'))
    def issue(self):
        d=self.e.evaluate(self.req,make_frames(),1000)
        return self.e.issue_capability(d)

    def test_context_replay_denied_after_restart(self):
        self.assertEqual(self.e.evaluate(self.req,make_frames(),1000).outcome,'ALLOW')
        self.assertEqual(self.reopen().evaluate(self.req,make_frames(),1000).outcome,'DENY')

    def test_used_token_denied_after_restart(self):
        token=self.issue()
        self.assertTrue(self.e.consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized)
        result=self.reopen().consume_capability(token,self.req,make_frames(seq=3,time=1011),1011)
        self.assertFalse(result.authorized)
        self.assertIn('capability_replay',result.reasons)

    def test_fresh_context_still_works_after_restart(self):
        self.e.evaluate(self.req,make_frames(),1000)
        self.assertEqual(self.reopen().evaluate(self.req,make_frames(seq=2,time=1010),1010).outcome,'ALLOW')

    def test_unconsumed_capability_survives_restart(self):
        token=self.issue()
        self.assertTrue(self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized)

    def test_registered_decision_can_be_issued_by_another_instance(self):
        d=self.e.evaluate(self.req,make_frames(),1000)
        token=self.reopen().issue_capability(d)
        self.assertTrue(self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized)

    def test_clock_rollback_is_retained(self):
        self.e.evaluate(self.req,make_frames(time=1020),1020)
        self.assertIn('clock_rollback',self.reopen().evaluate(self.req,make_frames(seq=2,time=1010),1010).reasons)

    def test_missing_database_never_reinitializes(self):
        self.path.unlink()
        with self.assertRaises(DurableStateError): self.reopen().evaluate(self.req,make_frames(),1000)
        self.assertFalse(self.path.exists())

    def test_explicit_provision_does_not_overwrite_existing_database(self):
        before=self.path.read_bytes()
        with self.assertRaises(DurableStateError): DurableVehiclePolicy.provision(self.path,KEYS,CAP,ACL)
        self.assertEqual(self.path.read_bytes(),before)

    def test_corrupt_database_returns_no_grant(self):
        self.path.write_bytes(b'corrupt database')
        with self.assertRaises(DurableStateError): self.e.evaluate(self.req,make_frames(),1000)

    def test_modified_state_is_rejected(self):
        c=sqlite3.connect(self.path)
        try:
            with c: c.execute("UPDATE policy_state SET payload='{}'")
        finally:
            c.close()
        with self.assertRaises(DurableStateError): self.reopen().evaluate(self.req,make_frames(),1000)

    def test_key_and_policy_changes_require_new_explicit_enrollment(self):
        modified=dict(KEYS);modified['gnss']=b'z'*32
        for kw in ({'keys':modified},{'cap':b'c'*32},{'acl':set()},{'config':PolicyConfig(source_ttl_ms=501)}):
            with self.subTest(kw=list(kw)),self.assertRaises(DurableStateError): self.reopen(**kw).evaluate(self.req,make_frames(),1000)

    def test_base_policy_token_cannot_cross_into_durable_policy(self):
        old=VehiclePolicyEngine(KEYS,CAP,ACL)
        token=old.issue_capability(old.evaluate(self.req,make_frames(),1000))
        result=self.e.consume_capability(token,self.req,make_frames(seq=2,time=1010),1010)
        self.assertFalse(result.authorized)
        self.assertIn('capability_policy_mismatch',result.reasons)

    def test_lease_expiry_stays_exclusive(self):
        token=self.issue()
        self.assertFalse(self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1250),1250).authorized)

    def test_revocation_is_rechecked(self):
        token=self.issue()
        self.assertFalse(self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1010,changes={'identity':{'revoked':True}}),1010).authorized)

    def test_concurrent_instances_authorize_used_token_once(self):
        token=self.issue()
        def consume(_): return self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized
        with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(consume,range(8)))
        self.assertEqual(sum(results),1)

    def test_held_write_lock_returns_no_grant(self):
        c=sqlite3.connect(self.path,isolation_level=None)
        try:
            c.execute('BEGIN IMMEDIATE')
            with self.assertRaises(DurableStateError): self.e.evaluate(self.req,make_frames(),1000)
        finally: c.rollback();c.close()

    def test_failed_commit_returns_no_authorization(self):
        token=self.issue()
        original=sqlite3.connect
        class FailedCommit:
            def __init__(self,*args,**kw): self.conn=original(*args,**kw)
            def execute(self,sql,*args):
                if sql=='COMMIT': raise sqlite3.OperationalError('injected commit failure')
                return self.conn.execute(sql,*args)
            def __getattr__(self,name): return getattr(self.conn,name)
        with patch('coact_cps.vehicle.durable_policy.sqlite3.connect',FailedCommit):
            with self.assertRaises(DurableStateError): self.e.consume_capability(token,self.req,make_frames(seq=2,time=1010),1010)
        self.assertTrue(self.reopen().consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized)

    def test_rollback_failure_does_not_mask_storage_error(self):
        original=sqlite3.connect
        class FailedRollback:
            def __init__(self,*args,**kw): self.conn=original(*args,**kw)
            def execute(self,sql,*args):
                if sql=='COMMIT': raise sqlite3.OperationalError('commit I/O failure')
                return self.conn.execute(sql,*args)
            def rollback(self): raise sqlite3.OperationalError('rollback I/O failure')
            def __getattr__(self,name): return getattr(self.conn,name)
        with patch('coact_cps.vehicle.durable_policy.sqlite3.connect',FailedRollback):
            with self.assertRaises(DurableStateError): self.e.evaluate(self.req,make_frames(),1000)

    def test_close_failure_returns_no_grant(self):
        original=sqlite3.connect
        class FailedClose:
            def __init__(self,*args,**kw): self.conn=original(*args,**kw)
            def close(self):
                self.conn.close()
                raise sqlite3.OperationalError('close I/O failure')
            def __getattr__(self,name): return getattr(self.conn,name)
        with patch('coact_cps.vehicle.durable_policy.sqlite3.connect',FailedClose):
            with self.assertRaises(DurableStateError): self.e.evaluate(self.req,make_frames(),1000)

    def test_provision_close_failure_has_the_storage_error_contract(self):
        original=sqlite3.connect
        class FailedClose:
            def __init__(self,*args,**kw): self.conn=original(*args,**kw)
            def close(self):
                self.conn.close()
                raise sqlite3.OperationalError('close I/O failure')
            def __getattr__(self,name): return getattr(self.conn,name)
        with patch('coact_cps.vehicle.durable_policy.sqlite3.connect',FailedClose):
            with self.assertRaises(DurableStateError): DurableVehiclePolicy.provision(self.path.with_name('failed.sqlite'),KEYS,CAP,ACL)

    def test_valid_old_snapshot_rollback_is_explicitly_not_prevented(self):
        token=self.issue();backup=self.path.with_suffix('.backup');shutil.copyfile(self.path,backup)
        self.assertTrue(self.e.consume_capability(token,self.req,make_frames(seq=2,time=1010),1010).authorized)
        shutil.copyfile(backup,self.path)
        self.assertTrue(self.reopen().consume_capability(token,self.req,make_frames(seq=3,time=1011),1011).authorized)

    def test_unicode_and_spaces_in_database_path(self):
        path=Path(self.temp.name)/'车辆 state.sqlite'
        try: os.fsencode(str(path))
        except UnicodeError:
            # ASCII-only POSIX filesystems cannot represent this name. The
            # storage API must deny cleanly rather than leak a raw codec error.
            with self.assertRaises(DurableStateError): DurableVehiclePolicy.provision(path,KEYS,CAP,ACL)
            return
        e=DurableVehiclePolicy.provision(path,KEYS,CAP,ACL)
        self.assertEqual(e.evaluate(self.req,make_frames(),1000).outcome,'ALLOW')

    def worker_input(self,token,**extra):
        return dict(db=str(self.path),keys={k:v.hex() for k,v in KEYS.items()},cap_key=CAP.hex(),acl=sorted(ACL),request=asdict(self.req),token=token,frames=make_frames(seq=2,time=1010),now=1010,**extra)

    def worker_path(self):
        return Path(__file__).resolve().parents[1]/'research/vehicle_durable_state/process_worker.py'

    def test_separate_python_process_preserves_consumed_nonce(self):
        token=self.issue()
        out=subprocess.run([sys.executable,str(self.worker_path())],input=json.dumps(self.worker_input(token)),capture_output=True,text=True,timeout=15)
        self.assertEqual(out.returncode,0,out.stderr)
        self.assertTrue(json.loads(out.stdout)['authorized'])
        self.assertFalse(self.reopen().consume_capability(token,self.req,make_frames(seq=3,time=1011),1011).authorized)

    def test_abrupt_exit_after_committed_grant_does_not_restore_token(self):
        token=self.issue()
        out=subprocess.run([sys.executable,str(self.worker_path())],input=json.dumps(self.worker_input(token,exit_after_commit=True)),capture_output=True,text=True,timeout=15)
        self.assertEqual(out.returncode,0,out.stderr)
        self.assertFalse(self.reopen().consume_capability(token,self.req,make_frames(seq=3,time=1011),1011).authorized)

    def test_competing_python_processes_authorize_token_once(self):
        token=self.issue();payload=json.dumps(self.worker_input(token))
        processes=[subprocess.Popen([sys.executable,str(self.worker_path())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
        for process in processes:
            process.stdin.write(payload);process.stdin.close();process.stdin=None
        rows=[]
        for process in processes:
            stdout,stderr=process.communicate(timeout=15)
            self.assertEqual(process.returncode,0,stderr);rows.append(json.loads(stdout))
        self.assertEqual(sum(row['authorized'] for row in rows),1)

if __name__=='__main__': unittest.main()
