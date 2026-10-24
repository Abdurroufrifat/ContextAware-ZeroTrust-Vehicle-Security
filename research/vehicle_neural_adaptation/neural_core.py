"""Available-channel DeepSets speed adaptation; no reference arguments at inference."""
import numpy as np
import torch
from torch import nn
def split_labels(stamps,sequence):
    if sequence not in {"Library01","Library02"}:raise ValueError("unsupported sequence")
    t=np.asarray(stamps,dtype=np.int64)
    if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0):raise ValueError("timestamps must strictly increase")
    if sequence=="Library02":return np.full(len(t),"transfer",dtype="U16")
    relative=t-t[0];span=int(relative[-1]);b1=span*3//5;b2=span*4//5;gap=2_000_000_000
    out=np.full(len(t),"purge",dtype="U16")
    out[relative<b1-gap]="train"
    out[(relative>=b1+gap)&(relative<b2-gap)]="validation"
    out[relative>=b2+gap]="temporal_test"
    return out
def point_features(rays,doppler,rotation):
    a=np.asarray(rays,float);d=np.asarray(doppler,float);r=np.asarray(rotation,float)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)==0 or d.shape!=(len(a),) or r.shape!=(3,3):raise ValueError("invalid feature shape")
    if not all(np.isfinite(v).all() for v in [a,d,r]):raise ValueError("nonfinite feature")
    if not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5):raise ValueError("rays must be unit")
    if not np.allclose(r@r.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5):raise ValueError("invalid rotation")
    return np.column_stack([a@r.T,-d/20.]).astype(np.float32)
def pack(frames):
    if not frames:raise ValueError("empty batch")
    rows=[np.asarray(f,dtype=np.float32) for f in frames]
    if any(f.ndim!=2 or f.shape[1]!=4 or len(f)==0 or not np.isfinite(f).all() for f in rows):raise ValueError("invalid point frame")
    x=torch.zeros(len(rows),max(map(len,rows)),4);mask=torch.zeros(x.shape[:2],dtype=torch.bool)
    for i,f in enumerate(rows):x[i,:len(f)]=torch.from_numpy(f.copy());mask[i,:len(f)]=True
    return x,mask
class SpeedSet(nn.Module):
    def __init__(self):
        super().__init__()
        self.points=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,x,mask):
        if x.ndim!=3 or x.shape[-1]!=4 or mask.shape!=x.shape[:2] or mask.dtype!=torch.bool:raise ValueError("invalid tensor shape")
        if not bool(mask.any(dim=1).all()) or not bool(torch.isfinite(x[mask]).all()):raise ValueError("invalid input points")
        h=self.points(x)
        mean=(h*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1,keepdim=True)
        maximum=h.masked_fill(~mask.unsqueeze(-1),float("-inf")).amax(dim=1)
        return self.head(torch.cat([mean,maximum],dim=1)).squeeze(-1)
def selected_epoch(history):
    if not history or any(not np.isfinite(r["validation_mae"]) for r in history):raise ValueError("invalid validation history")
    return int(min(history,key=lambda r:(r["validation_mae"],r["epoch"]))["epoch"])
