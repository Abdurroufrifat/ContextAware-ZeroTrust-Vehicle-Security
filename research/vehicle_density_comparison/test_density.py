import copy,json,math,unittest
try:
    import density_core as d
    import verify_density as v
except ModuleNotFoundError:
    d=v=None


def fixture():
    return [dict(pose_index=i,pose_utime=i*100000,wheel_utime=i*100000,wheel_index=i,
                 wheel_speed_mps=5.+.03*(i%3),pose_speed_mps=5.,receipt_age_us=0,status='valid')
            for i in range(180)]


class DensityTests(unittest.TestCase):
    def test_detector_available(self):
        self.assertIsNotNone(d,'Density component not implemented')

    def test_inclusive_epsilon_and_self_count(self):
        model=d.model([0.,1.,2.],1.,3)
        self.assertEqual(model['core'],[1.])
        self.assertTrue(d.normal(model,0.));self.assertTrue(d.normal(model,2.))
        self.assertFalse(d.normal(model,math.nextafter(2.,math.inf)))

    def test_no_core_points_and_no_clusters(self):
        model=d.model([0.,10.],.2,4)
        self.assertEqual(model['core'],[]);self.assertEqual(model['reward'],-1.)
        self.assertFalse(d.normal(model,0.))

    def test_core_bridges_and_separate_clusters(self):
        model=d.model([0.,.1,.2,5.,5.1,5.2],.21,3)
        self.assertEqual(model['clusters'],2)
        self.assertGreater(model['reward'],.9)
        self.assertFalse(d.normal(model,2.5))

    def test_singleton_silhouette_convention(self):
        model=d.model([0.,10.],.2,1)
        self.assertEqual(model['reward'],0.)

    def test_ucb_explores_every_arm_before_reusing(self):
        chosen,counts=d.select([float(i) for i in range(36)])
        self.assertEqual(sum(counts),72);self.assertTrue(all(x>=1 for x in counts))
        self.assertEqual(chosen,35)

    def test_ucb_tie_is_deterministic(self):
        chosen,counts=d.select([0.]*36)
        self.assertEqual(counts,[2]*36);self.assertEqual(chosen,0)

    def test_nonfinite_detector_inputs_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ArithmeticError):d.model([0.,value],.5,2)
            with self.assertRaises(ArithmeticError):d.normal(d.model([0.,.1],.5,2),value)

    def test_zero_scale_never_gets_an_invented_floor(self):
        fit,_=d.fit_models([0.]*30)
        self.assertFalse(fit['valid']);self.assertEqual(fit['sd'],0.)

    def test_unrepresentable_prefix_scale_is_invalid_and_serializable(self):
        values=[-1.7e308,1.7e308]
        fit,model=d.fit_models(values)
        other,reference=v.reference_fit(tuple(values))
        self.assertFalse(fit['valid']);self.assertIsNone(model)
        self.assertIsNone(fit['sd']);self.assertIsNone(reference)
        json.dumps([fit,other],allow_nan=False)
        v.equal(fit,other,'unrepresentable.scale')

    def test_prefix_cap_is_chronological(self):
        fit,_=d.fit_models([float(i%3) for i in range(1005)])
        self.assertEqual(fit['n'],1000)

    def test_models_freeze_before_attack(self):
        rows=fixture();a,ac=d.events(rows,'nominal');b,bc=d.events(rows,'wheel_bias_plus3')
        self.assertEqual(ac,bc);self.assertEqual(a[:50],b[:50])

    def test_no_post_prefix_training_update(self):
        rows=fixture();changed=copy.deepcopy(rows)
        for r in changed[50:]:r['wheel_speed_mps']=800.
        self.assertEqual(d.events(rows,'nominal')[1],d.events(changed,'nominal')[1])

    def test_protected_track_ignores_localization_target(self):
        rows=fixture();changed=copy.deepcopy(rows)
        for r in changed:r['pose_speed_mps']=900.
        a,_=d.events(rows,'wheel_bias_plus3');b,_=d.events(changed,'wheel_bias_plus3')
        self.assertEqual([r[5][2:] for r in a],[r[5][2:] for r in b])

    def test_missing_inputs_reset_difference(self):
        rows=fixture();rows[60]['wheel_utime']=0
        events,_=d.events(rows,'nominal')
        self.assertEqual(events[60][2][0],'missing')
        self.assertEqual(events[61][2][0],'missing')
        self.assertIsNone(events[61][1][0])

    def test_common_bias_has_no_sustained_unprotected_difference(self):
        rows=fixture();a,_=d.events(rows,'nominal');b,_=d.events(rows,'coordinated_bias_plus3')
        for i in range(51,150):self.assertAlmostEqual(a[i][1][0],b[i][1][0])

    def test_nonfinite_standardization_fails_closed(self):
        rows=fixture()
        for i,r in enumerate(rows):
            r['pose_speed_mps']=0.;r['wheel_speed_mps']=(1+i%2)*1e-320 if i<50 else 5.
        events,_=d.events(rows,'nominal')
        self.assertEqual(events[50][2][0],'numerical_failure')
        self.assertEqual(events[-1][5][:2],[None,None])
        json.dumps(events,allow_nan=False)

    def test_missing_anchor_never_becomes_authorized_fallback(self):
        events,_=d.events(fixture(),'unestablished_real_evidence')
        self.assertTrue(all(e[5][2:]==[None,None] for e in events))

    def test_all_fixed_scenarios_match_separate_reconstruction(self):
        for case in d.CASES:
            a,ac=d.events(fixture(),case);b,bc=v.reconstruct(fixture(),case)
            v.equal(a,b,case);v.equal(ac,bc,case+'.fit')

    def test_primary_and_reference_clusters_match_nontrivial_fixture(self):
        points=[0.,.05,.1,.6,.8,1.,2.,2.05,2.1,8.]
        for eps in d.EPS:
            for size in d.MINPTS:v.equal(d.model(points,eps,size),v.reference_model(points,eps,size),'cluster')

    def test_metrics_include_every_abstention(self):
        rows=fixture();events,_=d.events(rows,'unestablished_real_evidence')
        primary=d.Metrics();reference=v.Metrics()
        for e,r in zip(events,rows):
            for obj in (primary,reference):obj.add(e,r['pose_speed_mps'],r['wheel_speed_mps'],'during')
        v.equal(primary.report(),reference.report(),'metrics')
        for p in d.POLICIES:self.assertEqual(primary.report()[p]['all']['requests'],180)


if d is None:
    for name in dir(DensityTests):
        if name.startswith('test_') and name!='test_detector_available':
            setattr(DensityTests,name,unittest.skip('component not implemented')(getattr(DensityTests,name)))

if __name__=='__main__':unittest.main()
