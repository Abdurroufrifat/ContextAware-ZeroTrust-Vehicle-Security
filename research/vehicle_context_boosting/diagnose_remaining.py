"""Reproduce diagnosis of frozen pre-change models; no fitting or target selection."""
from pathlib import Path
import json,numpy as np
def compute():
 P=Path(__file__).resolve().parent.parent;N=P/'vehicle_night_training';M=P/'vehicle_masked_history';O=P/'vehicle_radar_multidrive'
 read=lambda p:json.loads(p.read_text())
 index=read(M/'frame_index.json');targets=read(P/'vehicle_continuous_radar/reference_targets.json');old=read(O/'prepared/calibration/index.json');lookup={v['time_ns']:i for i,v in enumerate(index)}
 with np.load(N/'predictions.npz') as z:pred={k:z[k] for k in z.files if k in ['anchor','day_plus_night__causal_residual_tcn','day_plus_night__masked_cnn_4ch']}
 with np.load(N/'decisions.npz') as z:quality=z['day_plus_night__quality']
 with np.load(M/'prepared.npz') as z:X=z['X'];median=z['median_feature']
 ys=np.array([np.nan if r['reference_mps'] is None else r['reference_mps'] for r in targets]);raw=X[:,0].astype(float)*10
 contexts={'current':raw[:,-1],'median':median,'current_minus_median':raw[:,-1]-median,'range_1s':np.ptp(raw[:,-21:],axis=1),'sd_1s':np.std(raw[:,-21:],axis=1),'range_6s':np.ptp(raw,axis=1),'trend_1s':raw[:,-1]-raw[:,-21],'trend_3s':raw[:,-1]-raw[:,-61]}
 result={}
 for block in sorted({r['window_id'] for r in old}):
  ids=np.array([lookup[r['time_ns']] for r in old if r['ready'] and r['window_id']==block]);ids=ids[quality[ids]];a=np.array([index[i]['array_index'] for i in ids]);y=ys[ids];values={}
  for name,p in pred.items():
   d=p[a]-y;values[name]={'n':len(d),'bias':float(d.mean()),'rmse':float(np.sqrt(np.mean(d*d))),'p95':float(np.quantile(abs(d),.95)),'maximum':float(np.max(abs(d)))}
  d=pred['day_plus_night__causal_residual_tcn'][a]-y
  values['context_correlations']={k:float(np.corrcoef(v[a],d)[0,1]) for k,v in contexts.items()}
  values['truth_speed_quantiles']=np.quantile(y,[0,.25,.5,.75,1]).tolist()
  values['chunks_10s']=[]
  t=np.array([index[i]['time_ns'] for i in ids]);seg=(t-t[0])//10000000000
  for s in np.unique(seg):
   take=seg==s;values['chunks_10s'].append({'block':int(s),'n':int(take.sum()),'bias':float(d[take].mean()),'p95':float(np.quantile(abs(d[take]),.95)),'mean_truth':float(y[take].mean()),'mean_range_1s':float(contexts['range_1s'][a][take].mean())})
  result[block]=values
 return result
if __name__=="__main__":print(json.dumps(compute(),indent=2))
