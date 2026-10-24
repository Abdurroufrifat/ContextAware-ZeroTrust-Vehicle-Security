"""Offline models and abstract replay policies; never controls a vehicle or issues permits."""
import bisect,collections,math,statistics
import core_io as c
METHODS=('magnitude_median_ols','signed_median_ols','signed_median_balanced')
PRIMARY='signed_median_balanced'

def decode_gnss(b,bag_time_ns):
 q=c.Cursor(b);seq,stamp,frame=q.header();q.header();used=q.unpack('<H')[0];q.array();visible=q.unpack('<H')[0]
 for _ in range(4):q.array()
 status,motion,orientation,position=q.unpack('<hHHH');values=q.unpack('<25d');q.unpack('<9d');q.unpack('<B')
 if q.p!=len(b):raise ValueError('unexpected GPSFix trailing bytes')
 return {'bag_time_ns':bag_time_ns,'header_time_ns':stamp,'sequence':seq,'frame_id':frame,'speed_raw':values[4],'unit':'mps','status':status,'motion_source':motion,'position_source':position,'err_speed':values[19]}

def speed_bin(y):return 0 if y<1 else 1 if y<5 else 2 if y<10 else 3

def fit_model(rows,balanced=False):
 if len(rows)<2 or max(r['label_mps'] for r in rows)-min(r['label_mps'] for r in rows)<3:return None
 if not all(math.isfinite(r[k]) for r in rows for k in ('feature_raw','label_mps')):return None
 counts=collections.Counter(speed_bin(r['label_mps']) for r in rows)
 weights=[1/counts[speed_bin(r['label_mps'])] if balanced else 1 for r in rows]
 total=math.fsum(weights);x=math.fsum(w*r['feature_raw'] for w,r in zip(weights,rows))/total;y=math.fsum(w*r['label_mps'] for w,r in zip(weights,rows))/total
 denominator=math.fsum(w*(r['feature_raw']-x)**2 for w,r in zip(weights,rows))
 if denominator<=1e-12:return None
 a=math.fsum(w*(r['feature_raw']-x)*(r['label_mps']-y) for w,r in zip(weights,rows))/denominator
 if not math.isfinite(a) or a<=0:return None
 return {'slope':a,'intercept':y-a*x}

def pair_window(radars,obds,gnss,start_ns):
 if any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(radars,radars[1:])):raise ValueError('ambiguous radar time')
 original,stats=c.causal_pairs(radars,obds,start_ns);lookup={r['time_ns']:r for r in original};history=[];head=0;rows=[]
 for r in radars:
  t=r['bag_time_ns'];good=c.valid_sample(r,250000000)
  if good:history.append(r)
  while head<len(history) and history[head]['bag_time_ns']<t-1000000000:head+=1
  if t not in lookup:continue
  old=lookup[t];g=latest_gnss(gnss,t)
  rows.append({**old,'feature_magnitude':old['feature_raw'],'feature_signed':abs(statistics.median(v['speed_raw'] for v in history[head:])),
    'gnss_mps':g['speed_raw'] if g else None,'gnss_motion_source':g['motion_source'] if g else None})
 return rows,stats

def latest_gnss(rows,t):
 valid=[r for r in rows if r['bag_time_ns']<=t and r['status']>=0 and r['speed_raw']>=0 and c.valid_sample(r,100000000) and 0<=t-r['header_time_ns']<=100000000]
 return max(valid,key=lambda r:r['bag_time_ns']) if valid else None

def method_feature(row,method):return row['feature_magnitude'] if method=='magnitude_median_ols' else row['feature_signed']

def predict(model,x):return c.predict(model,x) if model else None

def percentile(values,p=.95):
 a=sorted(values)
 if not a:return None
 z=(len(a)-1)*p;i=int(z);j=min(i+1,len(a)-1);return a[i]+(a[j]-a[i])*(z-i)

def metrics(rows,model,method,eligible):
 ys=[r['label_mps'] for r in rows];errors=[predict(model,method_feature(r,method))-r['label_mps'] for r in rows] if model else []
 result={'pairs':len(rows),'eligible_radar':eligible,'pair_coverage':len(rows)/eligible if eligible else 0,'moving_pairs_ge_5mps':sum(y>=5 for y in ys),'label_min_mps':min(ys) if ys else None,'label_max_mps':max(ys) if ys else None,'gnss_pairs':sum(r['gnss_mps'] is not None for r in rows),'speed_bin_counts':dict(collections.Counter(str(speed_bin(y)) for y in ys))}
 if errors:result.update({'mae_mps':statistics.mean(abs(e) for e in errors),'rmse_mps':math.sqrt(statistics.mean(e*e for e in errors)),'p95_disagreement_mps':percentile([abs(e) for e in errors]),'maximum_disagreement_mps':max(abs(e) for e in errors)})
 result['quality_gate_pass']=bool(model and result['pair_coverage']>=.95 and result.get('p95_disagreement_mps',math.inf)<=1 and result['moving_pairs_ge_5mps']>=50)
 return result

class ReplayPolicy:
 def __init__(self,name):self.name=name;self.latched=False;self.admitted=False;self.first=None;self.last=None;self.count=0
 def step(self,t,base_ok,witness_ok):
  if self.name=='gnss_obd_consistency':return base_ok
  if self.name=='radar_stateless':return base_ok and witness_ok
  if self.name=='radar_retained_alert':
   if not base_ok or not witness_ok:self.latched=True
   return base_ok and witness_ok and not self.latched
  if self.name!='radar_recovery_after_1s_fresh_contiguous_clean':raise ValueError('unknown policy')
  if not base_ok or not witness_ok:
   self.admitted=False;self.first=None;self.last=None;self.count=0;return False
  if self.last is None or not 0<t-self.last<=100000000:
   self.admitted=False;self.first=t;self.count=0
  self.last=t;self.count+=1
  if self.count>=5 and t-self.first>=1000000000:self.admitted=True
  return self.admitted
