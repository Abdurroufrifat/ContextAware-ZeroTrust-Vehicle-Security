"""Causal magnitude context from serialized Livox IMU payloads."""
import numpy as np
TTL=100_000_000;PERIOD=50_000_000;WIDTHS=(5,11,21,41,81,128)
FEATURE_NAMES=['imu_current_acceleration_norm','imu_current_gyro_norm','imu_current_available']+[f'imu_w{w}_{s}' for w in WIDTHS for s in ['acc_mean','acc_sd','acc_slope','gyro_mean','gyro_sd','gyro_slope','valid_fraction']]
def decode_payloads(raw):
 bag=np.asarray(raw['bag_ns']);data=np.asarray(raw['payload_bytes']);length=np.asarray(raw['payload_length']);n=len(bag)
 if not n or bag.ndim!=1 or bag.dtype!=np.dtype('int64') or np.any(bag<0) or np.any(bag[1:]<=bag[:-1]) or data.ndim!=2 or data.dtype!=np.uint8 or data.shape[0]!=n or data.shape[1]<16 or length.shape!=bag.shape or length.dtype.kind not in 'iu':raise ValueError('invalid serialized source arrays/clocks')
 head=data[:,:16].copy().view('<u4').reshape(n,4);frame_len=int(head[0,3]);size=16+frame_len+296
 if frame_len==0 or size>4096 or size>data.shape[1] or np.any(head[:,3]!=frame_len) or np.any(length!=size) or np.any(head[:,2]>=1_000_000_000) or np.any(data[:,size:]!=0) or not np.all(data[:,16:16+frame_len]==data[0,16:16+frame_len]):raise ValueError('invalid/mixed serialized frame, length or timestamp')
 try:frame=data[0,16:16+frame_len].tobytes().decode('utf-8')
 except UnicodeDecodeError as exc:raise ValueError('invalid frame encoding') from exc
 v=data[:,16+frame_len:size].copy().view('<f8').reshape(n,37);gyro=v[:,13:16].copy();gc=v[:,16:25];acc0=v[:,25:28];ac=v[:,28:37]
 with np.errstate(over='ignore',invalid='ignore'):
  acc=acc0*9.8;scaled=ac*(9.8**2);scaled=np.where(ac[:,0,None]==-1,ac,scaled);an=np.hypot.reduce(acc,axis=1);gn=np.hypot.reduce(gyro,axis=1)
 def covariance_ok(c):return np.isfinite(c).all(1)&(c[:,0]!=-1)&(c[:,[0,4,8]]>=0).all(1)
 valid=np.isfinite(gyro).all(1)&np.isfinite(acc).all(1)&covariance_ok(gc)&covariance_ok(scaled)&np.isfinite(an)&np.isfinite(gn)
 return {'bag_ns':bag.copy(),'header_ns':head[:,1].astype(np.int64)*1_000_000_000+head[:,2].astype(np.int64),'sequence':head[:,0].copy(),'frame_id':frame,'acceleration':acc,'angular_velocity':gyro,'acceleration_norm':an,'angular_velocity_norm':gn,'usable':valid,'accel_cov_unknown':np.all(ac==0,axis=1),'gyro_cov_unknown':np.all(gc==0,axis=1)}
def lookup(source,query):
 query=np.asarray(query);bag=np.asarray(source['bag_ns']);header=np.asarray(source['header_ns']);usable=np.asarray(source['usable'])
 if query.dtype.kind!='i' or bag.ndim!=1 or bag.dtype.kind!='i' or not len(bag) or np.any(bag<0) or np.any(bag[1:]<=bag[:-1]) or header.dtype.kind!='i' or np.any(header<0) or usable.dtype.kind!='b' or any(np.asarray(source[k]).shape!=bag.shape for k in ['header_ns','usable','acceleration_norm','angular_velocity_norm']):raise ValueError('invalid IMU query/source arrays')
 index=np.searchsorted(bag,query,side='right')-1;present=index>=0;ids=np.maximum(index,0);ha=np.where(present,query-header[ids],-1);ba=np.where(present,query-bag[ids],-1);status=np.ones(query.shape,dtype=np.uint8);future=present&(ha<0);stale=present&~future&((ha>TTL)|(ba>TTL));eligible=present&~future&~stale;finite=np.isfinite(source['acceleration_norm'][ids])&np.isfinite(source['angular_velocity_norm'][ids]);status[future]=3;status[stale]=2;status[eligible]=4;status[eligible&usable[ids]&finite]=0;good=status==0
 return {'target_ns':query.copy(),'latest_message':index,'header_age_ns':ha,'arrival_age_ns':ba,'status':status,'acceleration_norm':np.where(good,source['acceleration_norm'][ids],np.nan),'angular_velocity_norm':np.where(good,source['angular_velocity_norm'][ids],np.nan)}
def features(source,endpoints):
 endpoints=np.asarray(endpoints)
 if endpoints.ndim!=1 or endpoints.dtype.kind!='i' or not len(endpoints) or np.any(endpoints<0) or np.any(endpoints[1:]<=endpoints[:-1]):raise ValueError('invalid feature endpoint clock')
 grid=endpoints[:,None]+np.arange(-127,1,dtype=np.int64)[None,:]*PERIOD;out=lookup(source,grid);mask=out['status']==0;acc=np.where(mask,out['acceleration_norm'],0.);gyro=np.where(mask,out['angular_velocity_norm'],0.);columns=[acc[:,-1],gyro[:,-1],mask[:,-1].astype(float)]
 with np.errstate(over='ignore',invalid='ignore'):
  for width in WIDTHS:
   ok=mask[:,-width:];count=ok.sum(1);safe=np.maximum(count,1);t=np.arange(-(width-1),1)*.05;tm=(ok*t).sum(1)/safe;center=t[None,:]-tm[:,None];den=np.where(ok,center**2,0.).sum(1)
   for v in [acc[:,-width:],gyro[:,-width:]]:
    mean=v.sum(1)/safe;sd=np.sqrt(np.where(ok,(v-mean[:,None])**2,0.).sum(1)/safe);slope=np.divide(np.where(ok,center*(v-mean[:,None]),0.).sum(1),den,out=np.zeros(len(endpoints)),where=den>0);columns.extend([mean,sd,slope])
   columns.append(count/width)
 result=np.column_stack(columns)
 if result.shape!=(len(endpoints),45) or not np.isfinite(result).all():raise ValueError('nonfinite or malformed IMU features')
 q=lookup(source,endpoints);ids=np.maximum(q['latest_message'],0);present=q['latest_message']>=0;current={'endpoint_ns':endpoints.copy(),'status':q['status'],'latest_message':q['latest_message'],'bag_ns':np.where(present,source['bag_ns'][ids],-1),'header_ns':np.where(present,source['header_ns'][ids],-1)}
 return result,current
def prediction_binding(query,latest_endpoint,current):
 query=np.asarray(query);latest=np.asarray(latest_endpoint);n=len(current['endpoint_ns'])
 if query.ndim!=1 or query.dtype.kind!='i' or latest.shape!=query.shape or latest.dtype.kind!='i' or np.any(latest>=n) or np.any(latest< -1) or any(np.asarray(current[k]).shape!=(n,) for k in ['status','bag_ns','header_ns']):raise ValueError('invalid current-source binding arrays')
 if not n:return np.zeros(len(query),dtype=bool)
 ids=np.maximum(latest,0);ba=query-current['bag_ns'][ids];ha=query-current['header_ns'][ids]
 return (latest>=0)&(current['status'][ids]==0)&(query>=current['endpoint_ns'][ids])&(ba>=0)&(ha>=0)&(ba<=TTL)&(ha<=TTL)
