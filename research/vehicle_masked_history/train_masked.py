"""One fixed warm-start comparison using only the already frozen fit arrays."""
import platform,time
import numpy as np
from masked_core import ROOT,OLD,read,write,digest,freeze,augment,e,neural
def export(model,kind,seed,condition):return {'kind':kind,'seed':seed,'condition':condition,'state':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}}
def train():
 protocol=freeze()
 if (ROOT/'model_lock.json').exists() or (ROOT/'models').exists():raise ValueError('training already started; no unrecorded retries')
 import torch
 from torch.nn import functional as F
 from models_torch import TCN,CNN,initialize
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 with np.load(OLD/'prepared/fit/data.npz',allow_pickle=False) as stored:d={k:stored[k].copy() for k in ['X','y','median_feature']}
 y=d['y'];x=d['X'];weights=e.balanced_weights(y);anchor=e.affine(read(OLD/'classical_models.json')['reference_median_affine'],d['median_feature']);tx=torch.tensor(x);ty=torch.tensor(y,dtype=torch.float32);tw=torch.tensor(weights,dtype=torch.float32);ta=torch.tensor(anchor,dtype=torch.float32);runs=[];files={};started=time.monotonic();(ROOT/'models').mkdir()
 for condition in ['clean_control','masked_training']:
  (ROOT/'models'/condition).mkdir()
  for kind,old_kind,constructor in [('causal_residual_tcn','causal_residual_tcn',TCN),('masked_cnn_4ch','paper_inspired_cnn',CNN)]:
   for seed in [141,142,143]:
    torch.manual_seed(seed);order_rng=np.random.default_rng(seed);mask_rng=np.random.default_rng(seed+100000);model=constructor();original=read(OLD/f'models/{old_kind}_{seed}.json');initialize(model,original);model.eval()
    with torch.no_grad():initial=torch.clamp(model(tx[:64],ta[:64]),min=0).numpy()
    initial_diff=float(np.max(abs(initial-e.neural(original,x[:64],anchor[:64]))))
    if initial_diff>0.0005:raise ValueError('warm start changes original dense predictions')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001) if kind=='causal_residual_tcn' else torch.optim.RAdam(model.parameters(),lr=.001,weight_decay=.00001);history=[]
    for epoch in range(protocol['training']['epochs']):
     model.train();order=order_rng.permutation(len(y));total=mass=0.
     for offset in range(0,len(order),128):
      ids=order[offset:offset+128];batch=torch.tensor(augment(x[ids],mask_rng)) if condition=='masked_training' else tx[ids];optimizer.zero_grad();prediction=model(batch,ta[ids]);losses=F.huber_loss(prediction,ty[ids],reduction='none',delta=.2) if kind=='causal_residual_tcn' else (prediction-ty[ids])**2;loss=(losses*tw[ids]).sum()/tw[ids].sum();loss.backward();optimizer.step();total+=float((losses.detach()*tw[ids]).sum());mass+=float(tw[ids].sum())
     history.append(total/mass)
     if (epoch+1)%5==0:print('MASK TRAIN',condition,kind,seed,'epoch',epoch+1,'loss',round(history[-1],6),flush=True)
    model.eval();artifact=export(model,kind,seed,condition);probe=augment(x[:64],np.random.default_rng(999))
    with torch.no_grad():tp=torch.clamp(model(torch.tensor(probe),ta[:64]),min=0).numpy()
    difference=float(np.max(abs(neural(artifact,probe,anchor[:64])-tp)))
    if difference>.0005:raise ValueError('NumPy export does not reproduce masked Torch output')
    name=f'models/{condition}/{kind}_{seed}.json';write(ROOT/name,artifact);files[name]=digest(ROOT/name);runs.append({'condition':condition,'kind':kind,'seed':seed,'epochs':protocol['training']['epochs'],'parameters':sum(v.numel() for v in model.parameters()),'fit_loss_by_epoch':history,'warm_start_dense_max_difference_mps':initial_diff,'masked_numpy_torch_max_difference_mps':difference});write(ROOT/'training_trace.json',{'fit_samples':len(y),'runs':runs,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4})
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'models_sha256':files,'training_trace_sha256':digest(ROOT/'training_trace.json'),'primary':'masked_training','selection':'All three final-epoch seeds equally averaged; no result-based checkpoint or condition selection'})
 print('MASK MODELS FROZEN',len(files),'models;',round(time.monotonic()-started,1),'seconds',flush=True)
if __name__=='__main__':train()
