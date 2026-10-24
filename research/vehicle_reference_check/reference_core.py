"""Offline reference analysis only. No production permits or certificates."""
import bisect,collections,math
import core_io as c
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def decode_reference(kind,data,bag_time_ns):
 q=c.Cursor(data);sequence,stamp,frame=q.header();name=q.string();message_id,message_type,nov_sequence,time_status,gps_week,gps_ms=q.unpack('<HBIBHI')
 extra={}
 if kind=='INSPVA':
  values=q.unpack('<9d');status=q.unpack('<I')[0]
 elif kind=='INSPVAX':
  status,pos_type=q.unpack('<II');position=q.unpack('<3d');undulation=q.unpack('<f')[0];velocity_attitude=q.unpack('<6d');values=position+velocity_attitude
  stdev=q.unpack('<9f');ext,updated=q.unpack('<IH');extra=dict(zip(('latitude_stdev','longitude_stdev','height_stdev','north_velocity_stdev','east_velocity_stdev','up_velocity_stdev','roll_stdev','pitch_stdev','azimuth_stdev'),stdev));extra.update(pos_type=pos_type,undulation=undulation,extended_status=ext,time_since_update=updated)
 else:raise ValueError('unknown reference schema')
 if q.p!=len(data):raise ValueError('unexpected reference trailing bytes')
 row={'bag_time_ns':bag_time_ns,'header_time_ns':stamp,'sequence':sequence,'frame_id':frame,'message_name':name,'message_id':message_id,'time_status':time_status,'gps_week':gps_week,'gps_milliseconds':gps_ms,'status':status,**dict(zip(('latitude','longitude','height','north_velocity','east_velocity','up_velocity','roll','pitch','azimuth'),values)),**extra,'uniform_error_bound_mps':None}
 velocity=values[3:6]
 row['horizontal_mps']=math.hypot(*velocity[:2]) if all(finite(v) for v in velocity) else None;row['speed_3d_mps']=math.hypot(*velocity) if all(finite(v) for v in velocity) else None
 return {k:None if type(v) is float and not math.isfinite(v) else v for k,v in row.items()}
class ReferenceIndex:
 def __init__(self,rows):
  self.rows=rows;self.times=[r['header_time_ns'] for r in rows];self.status_counts=dict(collections.Counter(str(r['status']) for r in rows))
  if any(type(t) is not int or t<0 for t in self.times) or any(a>=b for a,b in zip(self.times,self.times[1:])):raise ValueError('nonunique or unordered reference time')
 def latest(self,t,max_age_ns):
  if type(t) is not int or type(max_age_ns) is not int or t<0 or max_age_ns<0:raise ValueError('invalid matching time')
  i=bisect.bisect_right(self.times,t)-1
  if i<0:return None
  r=self.rows[i]
  return r if t-self.times[i]<=max_age_ns and r['status']==3 and all(finite(r[k]) for k in ('horizontal_mps','speed_3d_mps')) else None
def conditional_limit(epsilon,acceleration,age_s):
 if not all(finite(v) and v>=0 for v in (epsilon,acceleration,age_s)):raise ValueError('conditional bounds must be finite and nonnegative')
 return 2*(epsilon+acceleration*age_s)
def qualify(evidence):
 e=evidence;bound=e.get('uniform_error_bound_mps');acceleration=e.get('acceleration_bound_mps2');age=e.get('max_reference_age_s');valid_bound=e.get('bound_kind')=='physical_uniform' and finite(bound) and bound>=0
 numeric=valid_bound and finite(acceleration) and acceleration>=0 and finite(age) and age>=0
 limit=conditional_limit(bound,acceleration,age) if numeric else None
 requirements={k:e.get(k) is True for k in ('raw_units_verified','geometry_verified','threat_independent','authenticated_live_channel','trusted_live_time')}
 requirements.update(non_postprocessed=e.get('postprocessed') is False,uniform_physical_bound=valid_bound,physical_acceleration_and_age_bounds=numeric,one_mps_hypotheses_disjoint=limit is not None and limit<1.)
 return {'qualified':all(requirements.values()),'requirements':requirements,'failed_requirements':[k for k,v in requirements.items() if not v],'conditional_ambiguity_limit_mps':limit,'interpretation':'Consistency of supplied evidence claims only; this function does not authenticate physical evidence, issue a certificate or grant production permission.'}
