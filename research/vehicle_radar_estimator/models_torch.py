"""Fixed models used only for training; JSON weights have a separate NumPy evaluator."""
import math
import torch
from torch import nn
from torch.nn import functional as F
class CausalResidualTCN(nn.Module):
 def __init__(self):
  super().__init__();self.stem=nn.Conv1d(4,16,1);self.layers=nn.ModuleList(nn.Conv1d(16,16,5,dilation=d) for d in (1,2,4,8,16));self.head=nn.Linear(16,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,anchor):
  h=F.relu(self.stem(x))
  for layer,d in zip(self.layers,(1,2,4,8,16)):h=(h+F.relu(layer(F.pad(h,(4*d,0)))))/math.sqrt(2)
  return anchor+self.head(h[:,:,-1]).squeeze(1)
class PaperInspiredCNN(nn.Module):
 def __init__(self):
  super().__init__();channels=(1,8,16,32,16,8,1);self.layers=nn.ModuleList(nn.Conv1d(a,b,7,padding=3) for a,b in zip(channels,channels[1:]));self.norms=nn.ModuleList(nn.BatchNorm1d(c) for c in channels[1:])
 def forward(self,x,anchor):
  h=x[:,:1,:]
  for layer,norm in zip(self.layers,self.norms):h=F.gelu(norm(layer(h)),approximate='tanh')
  return 10*h[:,0,-1]
