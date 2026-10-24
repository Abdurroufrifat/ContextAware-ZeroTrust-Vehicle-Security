"""Deterministic fixed-window features and offline labels, training recordings only."""
from night_core import *
def build(save=False):
 freeze();xs=[];ys=[];med=[];indices=[];reports=[];profiles={};allwindows=[]
 for route in ROUTES:
  radar,rp=source_rows(route,'radars.bag');reference,fp=source_rows(route,'reference.bag');lookup=r.ReferenceIndex(reference);grid=m.MaskedGrid(radar)
  profiles[route]={'radar':rp,'reference':fp,'reference_status_counts':dict(collections.Counter(str(x['status']) for x in reference))}
  selected=windows(route,int(radar[0]['bag_time_ns']),int(radar[-1]['bag_time_ns']));allwindows.extend(selected)
  for w in selected:
   counts={'raw':0,'unready':0,'missing_reference':0,'used':0}
   for row in radar:
    t=row['bag_time_ns']
    if not w['start_ns']<=t<=w['end_ns']:continue
    counts['raw']+=1
    # Match the original complete-history fit population; start context never crosses a block.
    pair=grid.at_masked(t,w['start_ns']);ready=pair is not None and bool(np.all(pair[0][3]==1));ref=lookup.latest(t,100000000);use=ready and ref is not None
    counts['unready']+=int(not ready);counts['missing_reference']+=int(ref is None);counts['used']+=int(use)
    indices.append({'route':route,'role':'fit','window_id':w['id'],'time_ns':t,'ready':ready,'reference_mps':ref['horizontal_mps'] if ref else None,'reference_header_ns':ref['header_time_ns'] if ref else None,'array_index':len(xs) if use else None})
    if use:xs.append(pair[0]);ys.append(ref['horizontal_mps']);med.append(pair[1])
   if not counts['used']:raise ValueError('empty fixed fit window; no replacement')
   reports.append({**w,**counts})
 data={'X':np.asarray(xs,dtype=np.float32),'y':np.asarray(ys,dtype=np.float64),'median_feature':np.asarray(med,dtype=np.float64)}
 if not len(ys) or not np.isfinite(data['X']).all() or not np.isfinite(data['y']).all():raise ValueError('empty or nonfinite fitting data')
 report={'scope':'fit only; no development labels','windows':reports,'profiles':profiles,'samples':len(ys),'reference_matching':'latest status3 INSPVA header<=endpoint, age<=100ms; no interpolation','feature_channels':'radar only; reference never used in features'}
 if save:
  if (ROOT/'data_lock.json').exists():raise ValueError('night data already frozen')
  np.savez_compressed(ROOT/'night_fit.npz',**data);write(ROOT/'night_index.json',indices);write(ROOT/'data_report.json',report)
  write(ROOT/'data_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'source_lock_sha256':digest(ROOT/'source_lock.json'),'files_sha256':{n:digest(ROOT/n) for n in ['night_fit.npz','night_index.json','data_report.json']}})
  print('NIGHT FIT FROZEN',len(ys),'samples;6 fixed90s windows; no evaluation labels',flush=True)
 return data,indices,report
if __name__=='__main__':build(True)
