"""Build complete fixed-block timelines; invalid/missing evidence stays visible."""
import bisect,math,statistics
import core_io as c
import final_core as f

def build_stream(radars,obds,gnss,start,end,model):
 if any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(radars,radars[1:])):raise ValueError('ambiguous radar clock')
 original,_=c.causal_pairs(radars,obds,start);lookup={r['time_ns']:r for r in original}
 gs=[r for r in gnss if r['status']>=0 and r['speed_raw']>=0 and c.valid_sample(r,100000000)];gt=[r['bag_time_ns'] for r in gs]
 history=[];head=0;rows=[];prefix=0
 for r in radars:
  t=r['bag_time_ns'];good=c.valid_sample(r,250000000)
  if good:history.append(r)
  while head<len(history) and history[head]['bag_time_ns']<t-1000000000:head+=1
  if not start<=t<=end:continue
  if t<start+1000000000:prefix+=1;continue
  row={'time_ns':t,'valid':False,'label_mps':None,'gnss_mps':None,'prediction_mps':None,'feature_signed':None,'gnss_motion_source':None}
  old=lookup.get(t)
  if old and good and len(history)-head>=10:
   x=abs(statistics.median(q['speed_raw'] for q in history[head:]));row.update(label_mps=old['label_mps'],feature_signed=x,prediction_mps=c.predict(model,x))
   j=bisect.bisect_right(gt,t)-1
   while j>=0 and gt[j]>=t-100000000:
    g=gs[j]
    if 0<=t-g['header_time_ns']<=100000000:row.update(gnss_mps=g['speed_raw'],gnss_motion_source=g['motion_source'],valid=True);break
    j-=1
  rows.append(row)
 return rows,{'scored_endpoints':len(rows),'valid_endpoints':sum(r['valid'] for r in rows),'missing_or_invalid_endpoints':sum(not r['valid'] for r in rows),'warmup_denied_endpoints':prefix,'raw_radar_in_block':len(rows)+prefix}
