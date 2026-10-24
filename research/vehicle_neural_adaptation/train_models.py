from shared import *
import numpy as np,torch,time,random,copy
from neural_core import SpeedSet,pack,selected_epoch
verify_lock()
assert not (R/"training_summary.json").exists(),"Training already completed; preserve outcomes"
allowed=(R/"train_validation.npz").resolve()
def training_guard(event,args):
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0]))
        if path.suffix.lower()==".npz" and path.resolve()!=allowed:raise PermissionError("training may only read its train/validation array")
sys.addaudithook(training_guard)
audit=read("data_audit.json");assert sha(allowed)==audit["data_hashes"][allowed.name]
data=np.load(allowed,allow_pickle=False);fs=frames(data);y=data["reference"];roles=data["roles"]
assert np.all(data["sequence"]==0) and set(roles)<=set(["train","validation"])
train=np.flatnonzero((roles=="train")&np.isfinite(y));val=np.flatnonzero((roles=="validation")&np.isfinite(y))
assert len(train)>=32 and len(val)>=8
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
targets=torch.tensor(y,dtype=torch.float32)
vx,vm=pack([fs[i] for i in val]);vy=targets[val]
summary={"protocol_sha256":sha(R/"PROTOCOL.md"),"training_file_sha256":sha(allowed),
 "train_scored":len(train),"validation_scored":len(val),"train_median_mps":float(np.median(y[train])),
 "epochs":150,"seeds":[17,29,43],"evaluation_labels_opened":False,"runs":[]}
start_all=time.perf_counter()
for seed in summary["seeds"]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=SpeedSet();optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)
    history=[];best=float("inf");state=None;start=time.perf_counter()
    for epoch in range(1,151):
        net.train();order=rng.permutation(train);loss_sum=0.
        for k in range(0,len(order),16):
            ix=order[k:k+16];x,mask=pack([fs[i] for i in ix]);target=targets[ix]
            optimizer.zero_grad(set_to_none=True);output=net(x,mask)
            loss=torch.nn.functional.huber_loss(output,target,delta=.3)
            assert bool(torch.isfinite(loss));loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        net.eval()
        with torch.inference_mode():mae=float(torch.mean(torch.abs(net(vx,vm)-vy)))
        row={"epoch":epoch,"train_huber":loss_sum/len(train),"validation_mae":mae};history.append(row)
        if mae<best:best=mae;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch%50==0:print("seed",seed,"epoch",epoch,"validation_mae",mae,flush=True)
    epoch=selected_epoch(history);assert history[epoch-1]["validation_mae"]==best
    path=R/("seed_"+str(seed)+".pt");assert not path.exists()
    torch.save({"state_dict":state,"seed":seed,"selected_epoch":epoch},path)
    write("history_"+str(seed)+".json",history)
    run={"seed":seed,"selected_epoch":epoch,"validation_mae":best,"checkpoint":path.name,
         "checkpoint_sha256":sha(path),"history_sha256":sha(R/("history_"+str(seed)+".json")),
         "training_seconds":time.perf_counter()-start,"parameters":sum(p.numel() for p in net.parameters())}
    summary["runs"].append(run);print(json.dumps(run),flush=True)
summary["total_training_seconds"]=time.perf_counter()-start_all
assert sha(allowed)==summary["training_file_sha256"]
write("training_summary.json",summary);print("TRAINING_COMPLETE",flush=True)
