"""A single fixed-budget training run, fitting role only; no calibration peeking."""
import json,math,platform,time
import numpy as np
from study_io import ROOT,read,write,digest,freeze
import estimator_core as e
def export(model,kind,seed):return {'kind':kind,'seed':seed,'state':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}}
def train():
 freeze(True)
 if (ROOT/'model_lock.json').exists():raise ValueError('models already frozen; do not retrain after scoring')
 import torch
 from torch.nn import functional as F
 from models_torch import CausalResidualTCN,PaperInspiredCNN
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 d=dict(np.load(ROOT/'prepared/fit/data.npz',allow_pickle=False));x=d['X'];y=d['y'];weights=e.balanced_weights(y);classical={'reference_median_affine':e.fit_affine(d['median_feature'],y,'fit'),'causal_savgol_affine':e.fit_affine(d['sg_feature'],y,'fit')};write('classical_models.json',classical)
 anchor=e.affine(classical['reference_median_affine'],d['median_feature']);tx=torch.tensor(x);ty=torch.tensor(y,dtype=torch.float32);tw=torch.tensor(weights,dtype=torch.float32);ta=torch.tensor(anchor,dtype=torch.float32);(ROOT/'models').mkdir(exist_ok=True);logs=[];model_files={};started=time.monotonic()
 for kind,constructor in (('causal_residual_tcn',CausalResidualTCN),('paper_inspired_cnn',PaperInspiredCNN)):
  for seed in (141,142,143):
   torch.manual_seed(seed);generator=np.random.default_rng(seed);model=constructor();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001) if kind=='causal_residual_tcn' else torch.optim.RAdam(model.parameters(),lr=.001,weight_decay=.00001);history=[]
   for epoch in range(80):
    model.train();order=generator.permutation(len(y));total=mass=0.
    for offset in range(0,len(order),128):
     ids=order[offset:offset+128];optimizer.zero_grad();prediction=model(tx[ids],ta[ids]);losses=F.huber_loss(prediction,ty[ids],reduction='none',delta=.2) if kind=='causal_residual_tcn' else (prediction-ty[ids])**2;loss=(losses*tw[ids]).sum()/tw[ids].sum();loss.backward();optimizer.step();total+=float((losses.detach()*tw[ids]).sum());mass+=float(tw[ids].sum())
    history.append(total/mass)
    if (epoch+1)%20==0:print('TRAIN',kind,'seed',seed,'epoch',epoch+1,'fit_loss',round(history[-1],6),flush=True)
   model.eval();artifact=export(model,kind,seed)
   with torch.no_grad():torch_prediction=torch.clamp(model(tx[:64],ta[:64]),min=0).numpy()
   numpy_prediction=e.neural(artifact,x[:64],anchor[:64]);difference=float(np.max(np.abs(numpy_prediction-torch_prediction)))
   if difference>.0005:raise ValueError('independent NumPy/PyTorch prediction mismatch')
   name=f'models/{kind}_{seed}.json';write(name,artifact);model_files[name]=digest(name);logs.append({'kind':kind,'seed':seed,'epochs':80,'parameters':sum(v.numel() for v in model.parameters()),'fit_loss_by_epoch':history,'numpy_torch_max_abs_difference_mps':difference})
   write('training_trace.json',{'runs':logs,'fit_samples':len(y),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'cpu_threads':4})
 lock={'design_lock_sha256':digest('design_lock.json'),'fit_prepared_sha256':digest('prepared/fit/data.npz'),'classical_sha256':digest('classical_models.json'),'models_sha256':model_files,'training_trace_sha256':digest('training_trace.json'),'primary':'causal_residual_tcn','selection':'Equal average of all three predeclared final-epoch seeds. No calibration labels loaded by training.'};write('model_lock.json',lock);print('MODELS FROZEN',len(model_files),'models;',round(time.monotonic()-started,1),'seconds',flush=True)
if __name__=='__main__':train()
