"""Descriptive post-score paired accounting; never changes models or decisions."""
from context_core import *
def compute():
 index=read(MASK/'frame_index.json');lookup={r['time_ns']:i for i,r in enumerate(index)};targets=read(ROOT.parent/'vehicle_continuous_radar/reference_targets.json');y=np.array([np.nan if r['reference_mps'] is None else r['reference_mps'] for r in targets]);blocks={}
 with np.load(NIGHT/'predictions.npz',allow_pickle=False) as z:old=z['day_plus_night__causal_residual_tcn'].copy()
 with np.load(ROOT/'replay.npz',allow_pickle=False) as z:new=z['context_prediction_mps'].copy();oldkeep=z['frozen_night_tcn__quality'].copy();newkeep=z['context_boosting__quality'].copy()
 for row in read(OLD/'prepared/calibration/index.json'):
  if row['ready']:blocks.setdefault(row['window_id'],[]).append(lookup[row['time_ns']])
 result={}
 for name,ids in blocks.items():
  ids=np.array(ids);source=int(oldkeep[ids].sum());ids=ids[oldkeep[ids]&newkeep[ids]];a=np.array([index[i]['array_index'] for i in ids]);result[name]={'shared_accepted_endpoints':len(ids),'old_accepted_endpoints':source,'additional_endpoint_rejections':source-len(ids),'frozen_night_tcn':c.q.error_metrics(y[ids],old[a]),'context_boosting':c.q.error_metrics(y[ids],new[a])}
 return {'scope':'Descriptive identical accepted endpoint cohorts, added after score; no model or gate selection','blocks':result}
if __name__=='__main__':write(ROOT/'paired_block_check.json',compute());print('PAIRED ORIGINAL BLOCK CHECKS SAVED')
