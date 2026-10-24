import unittest
import numpy as np
import synthesis_core as c
class SynthesisTests(unittest.TestCase):
    def test_exact_intact_route_draws(self):
        groups=np.array(['A','A','B','C','C','C'])
        draws=c.route_draws(groups)
        self.assertEqual(len(draws),27)
        for labels,ids in draws:
            expected=np.concatenate([np.flatnonzero(groups==label) for label in labels])
            np.testing.assert_array_equal(ids,expected)
        self.assertEqual(len({tuple(labels.count(g) for g in ('A','B','C')) for labels,_ in draws}),10)

    def test_quantile_uses_repeated_rows(self):
        y=np.zeros(4);p=np.array([0.,1.,2.,10.])
        self.assertAlmostEqual(c.metrics(y,p)['p95_mps'],8.8)

    def test_label_mismatch_is_rejected(self):
        with self.assertRaises(ValueError): c.align(np.array([1.,2.]),np.array([1.,3.]),np.array([0,1]),np.array([0,1]))

    def test_fold_mismatch_is_rejected(self):
        with self.assertRaises(ValueError): c.align(np.array([1.,2.]),np.array([1.,2.]),np.array([0,1]),np.array([1,0]))

    def test_identical_predictors_have_zero_paired_change(self):
        y=np.arange(6.);p=y+.2;groups=np.array(['A','A','B','B','C','C'])
        for _,ids in c.route_draws(groups):
            self.assertEqual(c.paired_delta(y[ids],p[ids],p[ids]),{'mae_mps':0.,'rmse_mps':0.,'p95_mps':0.})

    def test_paired_change_direction(self):
        values=c.paired_delta(np.zeros(10),np.full(10,.2),np.full(10,.5))
        for value in values.values(): self.assertAlmostEqual(value,-.3)

    def test_failures_are_not_rounded_away(self):
        self.assertEqual(c.metrics(np.zeros(20),np.full(20,.50001))['p95_mps']>.5,True)

    def test_invalid_or_nonfinite_input_rejected(self):
        for values in [np.array([np.nan]),np.array([-1.])]:
            with self.assertRaises(ValueError): c.metrics(np.array([0.]),values)

if __name__=='__main__': unittest.main()
