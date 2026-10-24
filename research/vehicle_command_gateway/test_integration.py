import copy,json,secrets,shutil,tempfile,unittest,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_policy_gateway"))
import policy_gateway as p

class IntegrationTests(unittest.TestCase):
 def setUp(self):
  import command_envelope as ce
  import command_gateway as cg
  self.ce=ce;self.cg=cg
  self.tmp=tempfile.TemporaryDirectory(dir=R);self.root=Path(self.tmp.name)
  self.context=p.auth.Context("vehicle-A","session-A","capture-A")
  self.ck=secrets.token_bytes(32);self.ak=secrets.token_bytes(32);self.hk=secrets.token_bytes(32)
  self.keys=[p.auth.KeyRule("candidate","c-key","wheel_speed",self.ck),p.auth.KeyRule("anchor","a-key","protected_speed",self.ak)]
  self.state=self.root/"state.sqlite";p.auth.create_state(self.state,self.context)
  self.guard=p.auth.Guard(self.state,self.context,self.keys)
  self.policies=[p.SourcePolicy("candidate","wheel_speed","candidate","drive",None,0,2000000),p.SourcePolicy("anchor","protected_speed","anchor","instrument",250,0,2000000,True)]
  self.rule=ce.CommandRule("command_logger","h-key",self.hk,"control")
  self.make_gateway()
 def tearDown(self):
  if hasattr(self,"tmp"):self.tmp.cleanup()
 def make_gateway(self,clock=0,execution=True,rule=None,policies=None):
  self.gateway=self.cg.CommandGateway(self.guard,policies or self.policies,rule or self.rule,
   clock_uncertainty_us=clock,execution_error_mmps2=1000,research_execution_assumption=execution)
 def report(self,role,speed=10000,seq=1,stamp=None,**changes):
  body=dict(schema=1,source_id=role,key_id="c-key" if role=="candidate" else "a-key",
   vehicle_id="vehicle-A",session_id="session-A",capture_id="capture-A",channel="wheel_speed" if role=="candidate" else "protected_speed",
   seq=seq,measured_us=(1000000 if role=="candidate" else 900000) if stamp is None else stamp,speed_mmps=speed)
  body.update(changes)
  return p.Packet(json.dumps(p.auth.sign(body,self.ck if role=="candidate" else self.ak)),1000000)
 def history(self,seq=1,commands=None,**changes):
  body=dict(schema=1,source_id="command_logger",key_id="h-key",vehicle_id="vehicle-A",session_id="session-A",capture_id="capture-A",
   channel="command_history",seq=seq,issued_us=1000000,commands=[[900000,1000000,0,1]] if commands is None else commands)
  body.update(changes)
  return p.Packet(json.dumps(self.ce.sign(body,self.hk)),1000000)
 def evaluate(self,c=None,a=None,h=None,**kw):
  return self.gateway.evaluate(self.report("candidate",10100) if c is None else c,self.report("anchor") if a is None else a,
    self.history() if h is None else h,kw.pop("now_us",1000000),**kw)
 def test_complete_pipeline_candidate(self):
  r=self.evaluate();self.assertEqual(r["selection"],"candidate");self.assertEqual(r["research_speed_mmps"],10100)
  self.assertEqual(r["command_status"],"authenticated");self.assertTrue(r["conditional_budget_satisfied"])
  for key in ["physical_speed_authorized","measurement_truth_established","promotion_allowed"]:self.assertFalse(r[key])
 def test_exact_boundary(self):self.assertEqual(self.evaluate(c=self.report("candidate",10150))["selection"],"candidate")
 def test_one_beyond_boundary_falls_back(self):self.assertEqual(self.evaluate(c=self.report("candidate",10151))["research_speed_mmps"],10000)
 def test_elevated_suppresses_candidate(self):
  r=self.evaluate(risk="elevated");self.assertEqual(r["selection"],"propagated_anchor")
 def test_lockdown_does_not_consume(self):
  self.assertEqual(self.evaluate(risk="lockdown")["reason"],"lockdown_context")
  self.assertEqual(self.evaluate()["selection"],"candidate")
 def test_unknown_context_does_not_consume(self):
  self.assertEqual(self.evaluate(risk="other")["reason"],"unsupported_risk_context");self.assertEqual(self.evaluate()["selection"],"candidate")
 def test_physical_operation_does_not_consume(self):
  self.assertEqual(self.evaluate(operation="drive")["reason"],"physical_operation_unavailable");self.assertEqual(self.evaluate()["selection"],"candidate")
 def test_boolean_query_time_does_not_consume(self):
  self.assertEqual(self.evaluate(now_us=True)["reason"],"invalid_receiver_time");self.assertEqual(self.evaluate()["selection"],"candidate")
 def test_unknown_execution_assumption(self):
  self.make_gateway(execution=False);self.assertEqual(self.evaluate()["reason"],"execution_assumption_not_supplied")
 def test_numeric_execution_assumption_rejected(self):
  with self.assertRaises(ValueError):self.make_gateway(execution=1)
 def test_boolean_clock_rejected(self):
  with self.assertRaises(ValueError):self.make_gateway(clock=True)
 def test_uncertain_age_over_limit(self):
  self.make_gateway(clock=1000);self.assertEqual(self.evaluate()["reason"],"uncertain_age_outside_window")
 def test_uncertain_calibration_start(self):
  self.policies[1]=p.SourcePolicy("anchor","protected_speed","anchor","instrument",250,950000,2000000,True)
  self.make_gateway(clock=1000)
  self.assertEqual(self.evaluate(a=self.report("anchor",stamp=950000))["reason"],"anchor_calibration_outside_interval")
 def test_missing_command(self):
  r=self.gateway.evaluate(self.report("candidate"),self.report("anchor"),None,1000000);self.assertEqual(r["reason"],"command_missing")
 def test_missing_anchor(self):
  r=self.gateway.evaluate(self.report("candidate"),None,self.history(),1000000);self.assertEqual(r["selection"],"abstain")
 def test_missing_candidate(self):
  r=self.gateway.evaluate(None,self.report("anchor"),self.history(),1000000);self.assertEqual(r["selection"],"propagated_anchor")
 def test_tampered_command_mac(self):
  h=self.history();m=json.loads(h.wire);m["body"]["commands"][0][2]=2000
  r=self.evaluate(h=p.Packet(json.dumps(m),1000000));self.assertEqual(r["reason"],"command_bad_tag")
 def test_tampered_anchor_denied(self):
  a=self.report("anchor");m=json.loads(a.wire);m["body"]["speed_mmps"]+=3000
  self.assertEqual(self.evaluate(a=p.Packet(json.dumps(m),1000000))["anchor_status"],"bad_tag")
 def test_tampered_candidate_fallback(self):
  c=self.report("candidate");m=json.loads(c.wire);m["body"]["speed_mmps"]+=3000
  r=self.evaluate(c=p.Packet(json.dumps(m),1000000));self.assertEqual(r["selection"],"propagated_anchor");self.assertEqual(r["candidate_status"],"bad_tag")
 def test_unknown_command_key(self):self.assertEqual(self.evaluate(h=self.history(key_id="unknown"))["reason"],"command_unknown_key")
 def test_revoked_command_key(self):
  self.make_gateway(rule=self.ce.CommandRule("command_logger","h-key",self.hk,"control",False))
  self.assertEqual(self.evaluate()["reason"],"command_revoked_key")
 def test_command_context_mismatch(self):self.assertEqual(self.evaluate(h=self.history(session_id="wrong"))["reason"],"command_context_mismatch")
 def test_command_channel_mismatch(self):self.assertEqual(self.evaluate(h=self.history(channel="wrong"))["reason"],"command_channel_denied")
 def test_stale_command_issue(self):self.assertEqual(self.evaluate(h=self.history(issued_us=899999,commands=[[800000,899999,0,1]]))["reason"],"command_measurement_stale")
 def test_invalid_command_receipt(self):
  h=self.history();self.assertEqual(self.evaluate(h=p.Packet(h.wire,999999))["reason"],"command_invalid_clock_order")
 def test_same_domain_rejected(self):
  with self.assertRaises(ValueError):self.make_gateway(rule=self.ce.CommandRule("command_logger","h-key",self.hk,"instrument"))
 def test_shared_key_rejected(self):
  with self.assertRaises(ValueError):self.make_gateway(rule=self.ce.CommandRule("command_logger","h-key",self.ak,"control"))
 def test_same_issuer_source_rejected(self):
  with self.assertRaises(ValueError):self.make_gateway(rule=self.ce.CommandRule("anchor","h-key",self.hk,"control"))
 def test_duplicate_command_json_key(self):
  h=self.history();wire=h.wire.replace('"schema": 1','"schema": 1, "schema": 1')
  self.assertEqual(self.evaluate(h=p.Packet(wire,1000000))["reason"],"command_malformed")
 def test_mutable_command_wire_rejected(self):
  h=self.history();self.assertEqual(self.evaluate(h=p.Packet(json.loads(h.wire),1000000))["reason"],"command_malformed")
 def test_extra_self_asserted_trust_rejected(self):
  h=self.history();m=json.loads(h.wire);m["body"]["trusted"]=True
  self.assertEqual(self.evaluate(h=p.Packet(json.dumps(m),1000000))["reason"],"command_malformed")
 def test_invalid_command_fields(self):
  variants=[[[900000,1000000,0,0]],[[900000,1000000,0,True]],[[900000,1000000,0,1000000001]],[[900000,1000000,2001,1]],[[900000,950000,0,1],[960000,1000000,0,1]],[[900000,1000000,True,1]]]
  for records in variants:
   with self.assertRaises(ValueError):self.history(commands=records)
 def test_incomplete_coverage(self):
  self.assertEqual(self.evaluate(h=self.history(commands=[[950000,1000000,0,1]]))["reason"],"incomplete_history")
 def test_restart_replay_denied(self):
  c,a,h=self.report("candidate"),self.report("anchor"),self.history();self.evaluate(c,a,h)
  self.guard=p.auth.Guard(self.state,self.context,self.keys);self.make_gateway()
  self.assertEqual(self.evaluate(c,a,h)["reason"],"command_replay_or_reorder")
 def test_command_consumed_before_bad_anchor(self):
  h=self.history();self.evaluate(a=self.report("anchor",session_id="wrong"),h=h)
  self.assertEqual(self.evaluate(h=h)["reason"],"command_replay_or_reorder")
 def test_bad_mac_does_not_consume_command(self):
  h=self.history();m=json.loads(h.wire);m["tag"]="0"*64
  self.evaluate(h=p.Packet(json.dumps(m),1000000));self.assertEqual(self.evaluate(h=h)["selection"],"candidate")
 def test_missing_state_fail_closed_without_recreation(self):
  self.state.unlink();self.assertEqual(self.evaluate()["reason"],"state_unavailable");self.assertFalse(self.state.exists())
 def test_corrupt_state_fail_closed(self):
  self.state.write_bytes(b"invalid sqlite");self.assertEqual(self.evaluate()["reason"],"state_unavailable")
 def test_concurrent_duplicate_at_most_one_output(self):
  c,a,h=self.report("candidate"),self.report("anchor"),self.history()
  with ThreadPoolExecutor(max_workers=2) as pool:r=list(pool.map(lambda _:self.evaluate(c,a,h),range(2)))
  self.assertEqual(sum(x["research_speed_mmps"] is not None for x in r),1)
 def test_valid_snapshot_rollback_failure_remains(self):
  backup=self.root/"old.sqlite";shutil.copyfile(self.state,backup)
  c,a,h=self.report("candidate"),self.report("anchor"),self.history();self.evaluate(c,a,h)
  shutil.copyfile(backup,self.state);self.guard=p.auth.Guard(self.state,self.context,self.keys);self.make_gateway()
  self.assertEqual(self.evaluate(c,a,h)["selection"],"candidate")
 def test_joint_valid_signer_bias_failure_remains(self):
  r=self.evaluate(c=self.report("candidate",13000),a=self.report("anchor",13000))
  self.assertEqual(r["research_speed_mmps"],13000);self.assertEqual(abs(r["research_speed_mmps"]-10000),3000);self.assertFalse(r["measurement_truth_established"])
 def test_valid_signed_command_forgery_failure_remains(self):
  r=self.gateway.evaluate(None,self.report("anchor"),self.history(commands=[[900000,1000000,2000,1]]),1000000)
  self.assertEqual(r["research_speed_mmps"],10150);self.assertEqual(abs(r["research_speed_mmps"]-(10000-4000*100000//1000000)),550)
 def test_issue_to_query_delay_abstains_without_complete_history(self):
  r=self.evaluate(a=self.report("anchor",stamp=950000),now_us=1001000)
  self.assertEqual(r["reason"],"incomplete_history")
 def test_malformed_incoming_commands_fail_closed(self):
  source=json.loads(self.history().wire)
  variants=[[[900000,1000000,0,0]],[[900000,1000000,0,True]],[[900000,1000000,0,1000000001]],
   [[900000,1000000,-4001,1]],[[900000,1000000,0.0,1]],[[900000,1000000,True,1]],
   [[900000,900000,0,1]],[[900000,960000,0,1],[950000,1000000,0,1]],
   [[900000+i*1000,901000+i*1000,0,1] for i in range(17)]]
  for records in variants:
   with self.subTest(records=records):
    m=copy.deepcopy(source);m["body"]["commands"]=records
    self.assertEqual(self.evaluate(h=p.Packet(json.dumps(m),1000000))["reason"],"command_malformed")
  self.assertEqual(self.evaluate(h=p.Packet(" "*8193,1000000))["reason"],"command_malformed")
if __name__=="__main__":unittest.main()
