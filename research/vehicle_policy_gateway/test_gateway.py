import copy,json,secrets,shutil,tempfile,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import policy_gateway as p

TMP=Path(__file__).resolve().parent/'_temporary';TMP.mkdir(exist_ok=True)

class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TMP);self.root=Path(self.tmp.name)
        self.context=p.auth.Context('vehicle-A','session-A','capture-A');self.key_a=secrets.token_bytes(32);self.key_b=secrets.token_bytes(32)
        self.keys=[p.auth.KeyRule('candidate','key-A','wheel_speed',self.key_a),p.auth.KeyRule('anchor','key-B','protected_speed',self.key_b)]
        self.path=self.root/'state.sqlite';p.auth.create_state(self.path,self.context)
        self.guard=p.auth.Guard(self.path,self.context,self.keys)
        self.policies=[p.SourcePolicy('candidate','wheel_speed','candidate','drivetrain',None,0,2000000),p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',250,0,2000000,True)]
        self.gateway=p.Gateway(self.guard,self.policies)
    def tearDown(self):self.tmp.cleanup()
    def packet(self,role,speed=5000,seq=1,measured=1000000,received=1000000,**changes):
        key=self.key_a if role=='candidate' else self.key_b
        body=dict(schema=1,source_id=role,key_id='key-A' if role=='candidate' else 'key-B',vehicle_id='vehicle-A',session_id='session-A',capture_id='capture-A',channel='wheel_speed' if role=='candidate' else 'protected_speed',seq=seq,measured_us=measured,speed_mmps=speed)
        body.update(changes)
        return p.Packet(json.dumps(p.auth.sign(body,key)),received)
    def evaluate(self,candidate=None,anchor=None,**kw):
        return self.gateway.evaluate(candidate or self.packet('candidate',5200),anchor or self.packet('anchor'),now_us=kw.pop('now_us',1000000),**kw)
    def test_authenticated_bounded_candidate(self):
        r=self.evaluate();self.assertEqual(r['selection'],'candidate');self.assertEqual(r['research_speed_mmps'],5200)
        self.assertFalse(r['physical_speed_authorized']);self.assertFalse(r['measurement_truth_established'])
    def test_exact_integer_error_budget_boundary(self):
        self.assertEqual(self.evaluate(self.packet('candidate',5250))['selection'],'candidate')
    def test_outside_budget_falls_back(self):
        r=self.evaluate(self.packet('candidate',5251));self.assertEqual(r['selection'],'anchor');self.assertEqual(r['research_speed_mmps'],5000)
    def test_bias_falls_back(self):self.assertEqual(self.evaluate(self.packet('candidate',8000))['research_speed_mmps'],5000)
    def test_bad_tag_candidate_never_used(self):
        m=json.loads(self.packet('candidate').wire);m['body']['speed_mmps']=8000
        r=self.evaluate(p.Packet(json.dumps(m),1000000));self.assertEqual(r['candidate_status'],'bad_tag');self.assertEqual(r['selection'],'anchor')
    def test_bad_anchor_cannot_authorize(self):
        m=json.loads(self.packet('anchor').wire);m['body']['speed_mmps']=8000
        r=self.evaluate(anchor=p.Packet(json.dumps(m),1000000));self.assertEqual(r['selection'],'abstain')
    def test_missing_anchor_abstains(self):
        r=self.gateway.evaluate(self.packet('candidate'),None,now_us=1000000);self.assertEqual(r['selection'],'abstain')
    def test_missing_candidate_uses_authenticated_anchor(self):
        r=self.gateway.evaluate(None,self.packet('anchor'),now_us=1000000);self.assertEqual(r['selection'],'anchor')
    def test_missing_research_assumption_abstains(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',250,0,2000000)
        r=p.Gateway(self.guard,self.policies).evaluate(self.packet('candidate'),self.packet('anchor'),1000000)
        self.assertEqual(r['reason'],'anchor_assumption_not_supplied')
    def test_elevated_context_uses_anchor(self):self.assertEqual(self.evaluate(risk='elevated')['selection'],'anchor')
    def test_lockdown_does_not_consume_report_sequence(self):
        a=self.packet('candidate');b=self.packet('anchor')
        self.assertEqual(self.evaluate(a,b,risk='lockdown')['selection'],'abstain')
        self.assertEqual(self.evaluate(a,b)['selection'],'candidate')
    def test_unknown_context_denied(self):self.assertEqual(self.evaluate(risk='unknown')['reason'],'unsupported_risk_context')
    def test_physical_operation_unavailable_without_state_consumption(self):
        a=self.packet('candidate');b=self.packet('anchor')
        self.assertEqual(self.evaluate(a,b,operation='drive')['reason'],'physical_operation_unavailable')
        self.assertEqual(self.evaluate(a,b)['selection'],'candidate')
    def test_vehicle_session_capture_mismatch(self):
        for field in ('vehicle_id','session_id','capture_id'):
            r=self.evaluate(self.packet('candidate',seq=2),self.packet('anchor',seq=2,**{field:'different'}))
            self.assertEqual(r['anchor_status'],'context_mismatch');self.assertEqual(r['selection'],'abstain')
    def test_swapped_roles_do_not_authenticate(self):
        r=self.evaluate(self.packet('anchor'),self.packet('candidate'))
        self.assertEqual(r['selection'],'abstain');self.assertEqual(r['anchor_status'],'role_mismatch')
        self.assertEqual(self.evaluate()['selection'],'candidate')
    def test_duplicate_replay_denied_after_restart(self):
        a=self.packet('candidate');b=self.packet('anchor');self.evaluate(a,b)
        self.gateway=p.Gateway(p.auth.Guard(self.path,self.context,self.keys),self.policies)
        r=self.evaluate(a,b);self.assertEqual(r['selection'],'abstain');self.assertEqual(r['anchor_status'],'replay_or_reorder')
    def test_freshness_inclusive(self):self.assertEqual(self.evaluate(now_us=1100000)['selection'],'candidate')
    def test_stale_anchor_abstains(self):self.assertEqual(self.evaluate(now_us=1100001)['selection'],'abstain')
    def test_calibration_expiry_checked_at_query(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',250,0,1000000,True)
        r=p.Gateway(self.guard,self.policies).evaluate(self.packet('candidate'),self.packet('anchor'),1000001)
        self.assertEqual(r['reason'],'anchor_calibration_outside_interval')
    def test_calibration_interval_inclusive(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',250,1000000,1000000,True)
        self.assertEqual(p.Gateway(self.guard,self.policies).evaluate(self.packet('candidate'),self.packet('anchor'),1000000)['selection'],'candidate')
    def test_same_declared_domain_rejected(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','drivetrain',250,0,2000000,True)
        with self.assertRaises(ValueError):p.Gateway(self.guard,self.policies)
    def test_uncertainty_above_budget_rejected(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',501,0,2000000,True)
        with self.assertRaises(ValueError):p.Gateway(self.guard,self.policies)
    def test_numeric_boolean_uncertainty_rejected(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',True,0,2000000,True)
        with self.assertRaises(ValueError):p.Gateway(self.guard,self.policies)
    def test_source_channel_not_in_guard_allowlist_rejected(self):
        self.policies[1]=p.SourcePolicy('anchor','wrong_channel','anchor','other-instrument',250,0,2000000,True)
        with self.assertRaises(ValueError):p.Gateway(self.guard,self.policies)
    def test_mutable_envelope_not_accepted_as_wire(self):
        r=self.evaluate(anchor=p.Packet(json.loads(self.packet('anchor').wire),1000000));self.assertEqual(r['selection'],'abstain')
    def test_extra_self_asserted_trust_field_rejected(self):
        message=json.loads(self.packet('anchor').wire);message['body']['trusted']=True
        r=self.evaluate(anchor=p.Packet(json.dumps(message),1000000));self.assertEqual(r['anchor_status'],'malformed_wire')
    def test_compromised_signer_counterexample_remains(self):
        r=self.evaluate(self.packet('candidate',8000),self.packet('anchor',8000))
        self.assertEqual(r['research_speed_mmps'],8000);self.assertFalse(r['measurement_truth_established']);self.assertFalse(r['physical_speed_authorized'])
    def test_authentication_consumes_sequence_even_if_policy_denies(self):
        a=self.packet('candidate');b=self.packet('anchor');self.evaluate(a,None,operation='drive')
        # A valid candidate is authenticated and committed even when anchor is missing.
        self.gateway.evaluate(a,None,1000000)
        r=self.gateway.evaluate(a,b,1000000);self.assertEqual(r['candidate_status'],'replay_or_reorder');self.assertEqual(r['selection'],'anchor')
    def test_policy_list_is_snapshotted(self):
        self.policies.clear();self.assertEqual(self.evaluate()['selection'],'candidate')
    def test_concurrent_duplicate_pair_has_only_one_research_output(self):
        a=self.packet('candidate');b=self.packet('anchor')
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:self.gateway.evaluate(a,b,1000000),range(2)))
        self.assertEqual(sum(r['research_speed_mmps'] is not None for r in results),1)
    def test_old_database_snapshot_rollback_counterexample_remains(self):
        backup=self.root/'old.sqlite';shutil.copyfile(self.path,backup)
        a=self.packet('candidate');b=self.packet('anchor');self.evaluate(a,b)
        shutil.copyfile(backup,self.path)
        self.gateway=p.Gateway(p.auth.Guard(self.path,self.context,self.keys),self.policies)
        r=self.evaluate(a,b);self.assertEqual(r['selection'],'candidate');self.assertFalse(r['physical_speed_authorized'])
    def test_revoked_anchor_key_abstains(self):
        revoked=[self.keys[0],p.auth.KeyRule('anchor','key-B','protected_speed',self.key_b,False)]
        self.gateway=p.Gateway(p.auth.Guard(self.path,self.context,revoked),self.policies)
        self.assertEqual(self.evaluate()['anchor_status'],'revoked_key')
    def test_candidate_policy_expiry_falls_back(self):
        self.policies[0]=p.SourcePolicy('candidate','wheel_speed','candidate','drivetrain',None,0,999999)
        self.gateway=p.Gateway(self.guard,self.policies)
        r=self.evaluate();self.assertEqual(r['selection'],'anchor');self.assertEqual(r['candidate_status'],'policy_interval_expired')
    def test_invalid_receiver_time_does_not_consume_sequence(self):
        a=self.packet('candidate');b=self.packet('anchor')
        self.assertEqual(self.evaluate(a,b,now_us=True)['reason'],'invalid_receiver_time')
        self.assertEqual(self.evaluate(a,b)['selection'],'candidate')
    def test_zero_and_full_uncertainty_boundaries(self):
        for uncertainty in (0,500):
            self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',uncertainty,0,2000000,True)
            gateway=p.Gateway(self.guard,self.policies)
            r=gateway.evaluate(self.packet('candidate',5000,seq=uncertainty+1),self.packet('anchor',5000,seq=uncertainty+1),1000000)
            self.assertEqual(r['selection'],'candidate')
    def test_different_sources_cannot_share_authentication_key(self):
        with self.assertRaises(ValueError):p.auth.Guard(self.path,self.context,[self.keys[0],p.auth.KeyRule('anchor','key-B','protected_speed',self.key_a)])
    def test_numeric_research_assumption_rejected(self):
        self.policies[1]=p.SourcePolicy('anchor','protected_speed','anchor','other-instrument',250,0,2000000,1)
        with self.assertRaises(ValueError):p.Gateway(self.guard,self.policies)
    def test_missing_state_is_not_automatically_recreated(self):
        self.path.unlink()
        with self.assertRaises(ValueError):self.evaluate()
        self.assertFalse(self.path.exists())

if __name__=='__main__':unittest.main()
