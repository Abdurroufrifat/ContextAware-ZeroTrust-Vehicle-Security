import inspect,json,math,tempfile,unittest
from pathlib import Path
import context_core as core
class ModelTests(unittest.TestCase):
    def model(self):
        rows=[dict(candidate=2.+i*.01,features=dict(peer=2.+i*.01,yaw=.02,steering=.1,acceleration=.2)) for i in range(30)]
        return core.fit_model(rows[:20],rows[20:])
    def test_learns_nominal_without_reference(self):
        model=self.model()
        self.assertAlmostEqual(core.predict(model,dict(peer=2.1,yaw=.02,steering=.1,acceleration=.2)),0.,places=8)
        self.assertGreaterEqual(model["threshold"],.05)
    def test_rejects_reference_as_online_feature(self):
        with self.assertRaises(ValueError):
            core.feature_vector(dict(peer=1.,yaw=0.,steering=0.,acceleration=0.,reference=1.))
    def test_nonfinite_context_rejected(self):
        with self.assertRaises(ValueError):
            core.feature_vector(dict(peer=math.nan,yaw=0.,steering=0.,acceleration=0.))
    def test_ood_is_explicit(self):
        self.assertTrue(core.out_of_support(self.model(),dict(peer=20.,yaw=.02,steering=.1,acceleration=.2)))
    def test_ins_not_a_policy_argument(self):
        self.assertNotIn("reference",inspect.signature(core.assess).parameters)
    def test_candidate_attack_rejected(self):
        model=self.model()
        r=core.assess(model,5.,dict(peer=2.1,yaw=.02,steering=.1,acceleration=.2),True)
        self.assertFalse(r["accept"])
        self.assertEqual(r["reason"],"context_deviation")
    def test_unestablished_model_rejected(self):
        with self.assertRaises(ValueError): core.fit_model([],[])
class BindingTests(unittest.TestCase):
    def fixture(self):
        parent=Path(__file__).resolve().parent/"runtime"
        parent.mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=parent)
        ctx=core.auth.Context("dev_vehicle","dev_session","dev_capture")
        rules=[core.auth.KeyRule("candidate","kc","speed",b"c"*32),core.auth.KeyRule("peer","ka","motion_context",b"a"*32)]
        path=Path(temp.name)/"state.sqlite"
        core.auth.create_state(path,ctx)
        guard=core.auth.Guard(path,ctx,rules)
        def body(source,key,channel,speed):
            return dict(schema=1,source_id=source,key_id=key,vehicle_id=ctx.vehicle_id,
                session_id=ctx.session_id,capture_id=ctx.capture_id,channel=channel,seq=1,measured_us=100,speed_mmps=speed)
        c=core.auth.sign(body("candidate","kc","speed",2000),b"c"*32)
        features=dict(peer=2.,yaw=.02,steering=.1,acceleration=.2)
        a=core.bind_context(body("peer","ka","motion_context",2000),features,b"a"*32)
        return temp,guard,json.dumps(c),json.dumps(a)
    def test_context_fields_bound_by_mac(self):
        temp,g,c,a=self.fixture()
        with temp:
            packet=json.loads(a);packet["features"]["yaw"]+=.1
            result=core.authenticate(g,c,json.dumps(packet),100)
            self.assertFalse(result["context_ok"])
    def test_valid_pair_then_replay_is_denied(self):
        temp,g,c,a=self.fixture()
        with temp:
            first=core.authenticate(g,c,a,100)
            self.assertTrue(first["candidate_ok"] and first["context_ok"])
            second=core.authenticate(g,c,a,101)
            self.assertFalse(second["candidate_ok"])
    def test_durable_replay_state_survives_new_guard(self):
        temp,g,c,a=self.fixture()
        with temp:
            core.authenticate(g,c,a,100)
            restarted=core.auth.Guard(g.path,g.context,list(g.rules.values()))
            self.assertFalse(core.authenticate(restarted,c,a,101)["candidate_ok"])
    def test_context_peer_must_match_envelope_value(self):
        temp,g,c,a=self.fixture()
        with temp:
            packet=json.loads(a)
            packet["features"]["peer"]=3.
            with self.assertRaises(ValueError):
                core.bind_context(packet["report"]["body"],packet["features"],b"a"*32)
    def test_future_and_stale_measurements_denied(self):
        for now in (99,100101):
            temp,g,c,a=self.fixture()
            with temp: self.assertFalse(core.authenticate(g,c,a,now)["candidate_ok"])
if __name__=="__main__": unittest.main()
