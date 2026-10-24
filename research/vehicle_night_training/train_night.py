"""Matched update-count day-only versus expanded night-fit continuation."""
import platform,time
from night_core import *
def train():
 protocol=freeze()
 if (ROOT/'models').exists() or (ROOT/'model_lock.json').exists():raise ValueError('training already started; no retries')
 import torch
 from torch.nn import functional as F
 from models_torch import TCN,CNN
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 with np.load(OLD/'prepared/fit/data.npz',allow_pickle=False) as z:day={k:z[k].copy() for k in ['X','y','median_feature']}
 with np.load(ROOT/'night_fit.npz',allow_pickle=False) as z:night={k:z[k].copy() for k in day}
 expanded={k:np.concatenate([day[k],night[k]]) for k in day};steps=(len(expanded['y'])+127)//128;runs=[];files={};started=time.monotonic();(ROOT/'models').mkdir()
 for condition,data in [('day_only',day),('day_plus_night',expanded)]:
  x=data['X'];y=data['y'];anchor=e.affine(read(OLD/'classical_models.json')['reference_median_affine'],data['median_feature']);tx=torch.tensor(x);ty=torch.tensor(y,dtype=torch.float32);ta=torch.tensor(anchor,dtype=torch.float32);tw=torch.tensor(fixed_weights(day['y'],y),dtype=torch.float32);(ROOT/'models'/condition).mkdir()
  for kind,constructor in zip(KINDS,[TCN,CNN]):
   for seed in SEEDS:
    torch.manual_seed(seed);order_rng=np.random.default_rng(seed);mask_rng=np.random.default_rng(seed+100000);model=constructor();original=read(MASK/f'models/masked_training/{kind}_{seed}.json');model.load_state_dict({k:torch.tensor(v,dtype=model.state_dict()[k].dtype) for k,v in original['state'].items()});model.eval()
    with torch.no_grad():initial=torch.clamp(model(tx[:64],ta[:64]),min=0).numpy()
    initial_difference=float(np.max(abs(initial-m.neural(original,x[:64],anchor[:64]))))
    if initial_difference>.0005:raise ValueError('initial checkpoint differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001) if kind==KINDS[0] else torch.optim.RAdam(model.parameters(),lr=.001,weight_decay=.00001);history=[];order_hash=hashlib.sha256();epochs=protocol['training']['epochs']
    for epoch in range(epochs):
     model.train();total=mass=0.
     for ids in epoch_ids(len(y),steps,order_rng):
      order_hash.update(ids.astype('<i8').tobytes());batch=torch.tensor(m.augment(x[ids],mask_rng));optimizer.zero_grad();p=model(batch,ta[ids]);losses=F.huber_loss(p,ty[ids],reduction='none',delta=.2) if kind==KINDS[0] else (p-ty[ids])**2;loss=(losses*tw[ids]).sum()/tw[ids].sum();loss.backward();optimizer.step();total+=float((losses.detach()*tw[ids]).sum());mass+=float(tw[ids].sum())
     history.append(total/mass)
    model.eval();artifact={'kind':kind,'seed':seed,'condition':condition,'state':{k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}};probe=m.augment(x[:64],np.random.default_rng(999))
    with torch.no_grad():tp=torch.clamp(model(torch.tensor(probe),ta[:64]),min=0).numpy()
    diff=float(np.max(abs(m.neural(artifact,probe,anchor[:64])-tp)))
    if diff>.0005:raise ValueError('portable exported inference differs')
    name=f'models/{condition}/{kind}_{seed}.json';write(ROOT/name,artifact);files[name]=digest(ROOT/name);runs.append({'condition':condition,'kind':kind,'seed':seed,'fit_samples':len(y),'epochs':epochs,'steps_per_epoch':steps,'total_optimizer_updates':epochs*steps,'batch_size':128,'order_sha256':order_hash.hexdigest(),'fit_loss_by_epoch':history,'warm_start_max_difference_mps':initial_difference,'numpy_torch_max_difference_mps':diff});write(ROOT/'training_trace.json',{'day_samples':len(day['y']),'night_samples':len(night['y']),'expanded_samples':len(expanded['y']),'runs':runs,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':4});print('NIGHT TRAINED',condition,kind,seed,epochs*steps,'updates',flush=True)
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'data_lock_sha256':digest(ROOT/'data_lock.json'),'models_sha256':files,'training_trace_sha256':digest(ROOT/'training_trace.json'),'selection':'all three final checkpoints equally averaged; primary day_plus_night'})
 print('TWELVE MODELS FROZEN',round(time.monotonic()-started,1),'seconds',flush=True)
if __name__=='__main__':train()
