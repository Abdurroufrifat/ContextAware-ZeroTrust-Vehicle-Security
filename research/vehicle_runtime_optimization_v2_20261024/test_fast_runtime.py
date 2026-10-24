import unittest,threading,subprocess,sqlite3,json
import numpy as np
from runtime_common import *
from fast_policy import DirectForest,DurableWALPolicy
from pooled_policy import PooledFastPolicy as FastLifecyclePolicy
class FastRuntimeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  t=next(t for t in load(B/"results.json")["trials"] if t["method"]=="joint" and t["case"]=="nominal")
  cls.rows=trace(t);cls.context=cls.rows[0]["context"]
  fixture=Original(fresh_state("unit-model"),cls.context,KEYS,"joint")
  cls.model=fixture.source.inner.estimator.refined.model;cls.fast=DirectForest(cls.model)
  cls.guard=fixture.source.inner.estimator.base.gateway.commands.report_guard
 def policy(self):return FastLifecyclePolicy(fresh_state("unit"),self.context,KEYS,"joint")
 def test_frozen_forest_random_vectors_exact(self):
  x=np.random.default_rng(227).normal(size=(256,self.model.n_features_in_))
  self.assertTrue(np.array_equal(self.model.predict(x),self.fast.predict(x)))
 def test_frozen_observed_features_exact(self):
  x=[r["output"]["source_output"]["estimator_decision"]["relative_features"] for r in self.rows if r["output"]["source_output"]["estimator_decision"].get("relative_features") is not None]
  self.assertGreater(len(x),0);self.assertTrue(np.array_equal(self.model.predict(x),self.fast.predict(x)))
 def test_float32_split_neighbors_exact(self):
  x=[]
  for t in self.model.estimators_[:10]:
   tr=t.tree_
   for node in range(min(10,tr.node_count)):
    if tr.feature[node]<0:continue
    v=np.float32(tr.threshold[node])
    for z in [np.nextafter(v,np.float32(-np.inf)),v,np.nextafter(v,np.float32(np.inf))]:
     a=np.zeros(self.model.n_features_in_);a[tr.feature[node]]=z;x.append(a)
  self.assertTrue(np.array_equal(self.model.predict(x),self.fast.predict(x)))
 def test_bad_shape_native_fallback(self):
  with self.assertRaises(ValueError):self.fast.predict([[0]])
 def test_nan_native_fallback(self):
  x=np.zeros((1,self.model.n_features_in_));x[0,0]=np.nan
  self.assertTrue(np.array_equal(self.model.predict(x),self.fast.predict(x),equal_nan=True))
 def test_full_sync_and_wal_active(self):
  p=self.policy()
  try:
   self.assertEqual(p._keeper.execute("PRAGMA journal_mode").fetchone()[0],"wal")
   self.assertEqual(p._keeper.execute("PRAGMA synchronous").fetchone()[0],2)
   c=sqlite3.connect(p.database);self.assertEqual(c.execute("PRAGMA synchronous").fetchone()[0],2);c.close()
  finally:p.close()
 def test_saved_policy_commands_and_gate_exact(self):
  p=self.policy()
  try:run_rows(p,self.rows[:20])
  finally:p.close()
 def test_forged_wire_fails_closed(self):
  p=self.policy();x=self.rows[0]
  try:
   out=p.process(x["input"]);packet=p.command_packet(out["requested_mmps2"],x["input"]["now_us"],1)
   bad=dict(packet);obj=json.loads(packet["wire"]);obj["body"]["accel_mmps2"]=2000;bad["wire"]=json.dumps(obj)
   accepted,status=p.actuate(bad,x["input"]["now_us"])
   self.assertEqual(accepted,-4000);self.assertNotEqual(status,"authenticated")
  finally:p.close()
 def test_watchdog_expiry_unchanged(self):
  p=self.policy();x=self.rows[0];now=x["input"]["now_us"]
  try:
   out=p.process(x["input"]);value,status=p.actuate(p.command_packet(out["requested_mmps2"],now,1),now)
   self.assertEqual(p.watchdog(now+99999),(value,"watchdog_active"))
   self.assertEqual(p.watchdog(now+100000),(-4000,"watchdog_expired"))
  finally:p.close()
 def test_single_owner_thread_enforced(self):
  p=self.policy();errors=[]
  def worker():
   try:p.process(self.rows[0]["input"])
   except ValueError as e:errors.append(str(e))
  t=threading.Thread(target=worker);t.start();t.join();p.close()
  self.assertTrue(errors);self.assertIn("owner",errors[0])
 def test_closed_adapter_rejects_calls(self):
  p=self.policy();p.close()
  with self.assertRaises(ValueError):p.process(self.rows[0]["input"])
 def test_replay_counter_survives_reopen(self):
  p=self.policy();run_rows(p,self.rows[:1]);path=p.database;p.close()
  g=type(self.guard)(path,self.guard.context,list(self.guard.rules.values()));x=self.rows[0]
  answer=g.verify(json.loads(x["input"]["packets"]["candidate"]["wire"]),x["input"]["now_us"],x["input"]["now_us"])
  self.assertEqual(answer["status"],"replay_or_reorder")
 def crash(self,mode):
  state=fresh_state("exit-"+mode)
  r=subprocess.run([sys.executable,str(S/"crash_child.py"),str(state),mode],capture_output=True,text=True,timeout=30)
  self.assertEqual(r.returncode,0,r.stderr);self.assertIn("COMMITTED_FIRST_FRAME",r.stdout)
  c=sqlite3.connect(state/"replay.sqlite");counters=c.execute("SELECT seq FROM counters").fetchall()
  self.assertEqual(len(counters),3);self.assertTrue(all(x==(1,) for x in counters))
  self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0],"ok");c.close()
 def test_committed_state_survives_abrupt_process_exit(self):self.crash("committed")
 def test_uncommitted_state_rolled_back_on_process_exit(self):self.crash("uncommitted")
 def test_each_authentication_commit_is_immediately_visible(self):
  p=self.policy()
  try:
   run_rows(p,self.rows[:1]);c=sqlite3.connect(p.database)
   self.assertEqual(c.execute("SELECT COUNT(*) FROM counters WHERE seq=1").fetchone()[0],3);c.close()
   self.assertFalse(p._keeper.in_transaction)
  finally:p.close()
 def test_unregistered_database_uses_native_connection(self):
  p=self.policy()
  try:
   baseline=Original(fresh_state("native-route"),self.context,KEYS,"joint")
   run_rows(baseline,self.rows[:2])
   path=baseline.source.inner.estimator.base.gateway.commands.report_guard.path
   import pooled_policy
   c=pooled_policy._auth.connect(path)
   self.assertIsInstance(c,sqlite3.Connection);self.assertEqual(c.execute("PRAGMA journal_mode").fetchone()[0],"delete");c.close()
  finally:p.close()
 def test_unfinished_connection_loan_rolls_back(self):
  p=self.policy()
  try:
   import pooled_policy
   c=pooled_policy._auth.connect(p.database);c.execute("BEGIN IMMEDIATE")
   c.execute("UPDATE metadata SET value='999' WHERE name='last_now_us'")
   with self.assertRaises(ValueError):c.close()
   self.assertFalse(p._keeper.in_transaction)
   self.assertEqual(p._keeper.execute("SELECT value FROM metadata WHERE name='last_now_us'").fetchone()[0],"-1")
  finally:p.close()
if __name__=="__main__":unittest.main()
