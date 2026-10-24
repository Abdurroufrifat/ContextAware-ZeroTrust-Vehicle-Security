import unittest
import numpy as np
import torch
import estimator_core as e
from models_torch import CausalResidualTCN,PaperInspiredCNN
from train_estimator import export
class ExportTests(unittest.TestCase):
 def test_independent_numpy_matches_torch_for_nonzero_residual_weights(self):
  torch.manual_seed(7);model=CausalResidualTCN();torch.nn.init.normal_(model.head.weight,std=.1);model.eval();x=torch.randn(3,4,128);anchor=torch.tensor([2.,4.,6.])
  with torch.no_grad():expected=torch.clamp(model(x,anchor),min=0).numpy()
  actual=e.neural(export(model,'causal_residual_tcn',7),x.numpy(),anchor.numpy());np.testing.assert_allclose(actual,expected,atol=1e-5)
 def test_cnn_export_uses_frozen_batchnorm_statistics(self):
  torch.manual_seed(8);model=PaperInspiredCNN()
  for norm in model.norms:norm.running_mean.copy_(torch.rand_like(norm.running_mean));norm.running_var.copy_(1+torch.rand_like(norm.running_var))
  model.eval();x=torch.randn(3,4,128);anchor=torch.zeros(3)
  with torch.no_grad():expected=torch.clamp(model(x,anchor),min=0).numpy()
  actual=e.neural(export(model,'paper_inspired_cnn',8),x.numpy(),anchor.numpy());np.testing.assert_allclose(actual,expected,atol=1e-5)
if __name__=='__main__':unittest.main()
