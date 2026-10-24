import unittest
import numpy as np
from head_core import predict,scaler,head_features

class HeadTests(unittest.TestCase):
    def test_weighted_scaler_uses_only_supplied_rows(self):
        x=np.tile(np.array([[0.],[2.]]),(1,7));m,s=scaler(x,np.array([1.,3.]))
        np.testing.assert_allclose(m,1.5);np.testing.assert_allclose(s,np.sqrt(.75))
    def test_constant_feature_scale_is_one(self):
        m,s=scaler(np.ones((3,7)),np.ones(3));np.testing.assert_array_equal(s,np.ones(7))
    def test_invalid_weights_rejected(self):
        for weights in [np.array([0.,0.]),np.array([1.,-1.]),np.array([1.,np.nan])]:
            with self.assertRaises(ValueError):scaler(np.zeros((2,7)),weights)
    def test_huber_portable_intercept_and_scaling(self):
        m={'method':'huber7','mean':[1.]*7,'scale':[2.]*7,'coefficient':[2.]*7,'intercept':3.}
        np.testing.assert_allclose(predict(m,np.full((2,7),3.)),17.)
    def test_extra_trees_uses_float32_split_semantics(self):
        t=[{'leaf':0,'feature':0,'threshold':1.00000002,'left':1,'right':2,'value':0.},
           {'leaf':1,'value':2.},{'leaf':1,'value':8.}]
        m={'method':'extra_trees7','baseline':0.,'trees':[t]};x=np.zeros((1,7));x[0,0]=1.00000003
        self.assertEqual(float(predict(m,x)[0]),2.)
    def test_ensemble_values_are_accumulated(self):
        m={'method':'extra_trees7','baseline':0.,'trees':[[{'leaf':1,'value':1.}],[{'leaf':1,'value':3.}]]}
        self.assertEqual(float(predict(m,np.zeros((1,7)))[0]),4.)
    def test_nonfinite_and_wrong_shape_rejected(self):
        m={'method':'extra_trees7','baseline':0.,'trees':[[{'leaf':1,'value':1.}]]}
        for x in [np.full((1,7),np.nan),np.zeros((1,6))]:
            with self.assertRaises(ValueError):predict(m,x)
    def test_unknown_or_empty_forest_rejected(self):
        for m in [{'method':'unknown'},{'method':'extra_trees7','baseline':0.,'trees':[]}]:
            with self.assertRaises(ValueError):predict(m,np.zeros((1,7)))
    def test_seven_inputs_preserve_parent_then_axes_order(self):
        x=head_features(np.array([8.]),np.arange(6).reshape(1,6));np.testing.assert_array_equal(x,[[8.,0.,1.,2.,3.,4.,5.]])
    def test_parent_axis_alignment_required(self):
        with self.assertRaises(ValueError):head_features(np.array([1.,2.]),np.zeros((1,6)))
    def test_independent_forest_checker_preserves_double_threshold(self):
        from verify_heads import infer
        tree=[{'leaf':0,'feature':0,'threshold':.99999999,'left':1,'right':2,'value':0.},
              {'leaf':1,'value':2.},{'leaf':1,'value':8.}]
        model={'method':'extra_trees7','baseline':0.,'trees':[tree]};x=np.ones((1,7))
        self.assertEqual(float(infer(model,x)[0]),8.)

if __name__=='__main__':unittest.main()
