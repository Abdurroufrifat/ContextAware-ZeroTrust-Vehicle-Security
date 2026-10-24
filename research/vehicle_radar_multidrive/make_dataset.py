from pathlib import Path
import base64,gzip,json
import numpy as np
import core_io as c
import estimator_core as e
ROOT=Path(__file__).resolve().parent
def build(role,write=True):
 folder=ROOT/'inputs'/role;labels=json.loads((folder/'labels.json').read_text(encoding='utf-8'));windows=json.loads((folder/'windows.json').read_text(encoding='utf-8'));radar=[]
 with gzip.open(folder/'packets.jsonl.gz','rt',encoding='utf-8') as file:
  for line in file:
   p=json.loads(line)
   if p['file']=='radars.bag':radar.append(c.decode_message('radar',base64.b64decode(p['data'],validate=True),p['time_ns']))
 radar.sort(key=lambda q:q['bag_time_ns']);grid=e.RadarGrid(radar);starts={w['id']:w['start_ns'] for w in windows};index=[];x=[];y=[];median=[];sg=[];old=[]
 for q in labels:
  if q['role']!=role:raise ValueError('mixed role in dataset')
  t=q['time_ns'];start=starts[q['window_id']];features=grid.at(t,start);lo=np.searchsorted(grid.times,t-1000000000,side='left');hi=np.searchsorted(grid.times,t,side='right');raw_feature=abs(float(np.median(grid.values[lo:hi]))) if hi-lo>=10 else None;ready=features is not None and q['reference_horizontal_mps'] is not None and raw_feature is not None
  index.append({'time_ns':t,'window_id':q['window_id'],'role':role,'ready':ready,'warmup':t-start<6350000000,'array_index':len(x) if ready else None})
  if ready:
   x.append(features);y.append(q['reference_horizontal_mps']);median.append(raw_feature);sg.append(abs(float(e.sg_coefficients()@(features[0,-51:].astype(np.float64)*10))));old.append(max(0.,1.6379326297523975*raw_feature-0.16449761978127508))
 arrays={'X':np.asarray(x,dtype=np.float32),'y':np.asarray(y,dtype=np.float64),'median_feature':np.asarray(median,dtype=np.float64),'sg_feature':np.asarray(sg,dtype=np.float64),'frozen_prediction':np.asarray(old,dtype=np.float64)}
 if write:
  dest=ROOT/'prepared'/role;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'data.npz',**arrays);(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
  print('PREPARED',role,len(x),'supported endpoints;',sum(q['warmup'] for q in index),'extra warmup;',sum(not q['ready'] and not q['warmup'] for q in index),'unsupported after warmup',flush=True)
 return arrays,index
if __name__=='__main__':
 for role in ('fit','calibration'):build(role)
