import math,struct
import pytest
import probe_core as c

def field(**kw):
 out=b''
 for k,v in kw.items():
  q=k.encode()+b'='+v;out+=struct.pack('<I',len(q))+q
 return out

def record(h,d):return struct.pack('<I',len(h))+h+struct.pack('<I',len(d))+d

def test_records_reject_truncated_or_oversized_data():
 blob=record(field(op=b'\x02'),b'payload')
 assert list(c.records(blob))[0][1]==b'payload'
 for invalid in [blob[:-1],b'\xff\xff\xff\xff',blob+b'\x00']:
  with pytest.raises(ValueError):list(c.records(invalid))

def test_duplicate_header_fields_rejected():
 with pytest.raises(ValueError):c.fields(field(op=b'1')+field(op=b'2'))

def radar(t,speed=2.0,header=None):return {'bag_time_ns':t,'header_time_ns':t if header is None else header,'speed_raw':speed}
def obd(t,speed=7.2,header=None,unit='kph'):return {**radar(t,speed,header),'unit':unit}

def test_future_obd_never_used_and_stale_obd_is_missing():
 r=[radar(t) for t in range(0,1500000001,50000000)]
 pairs,stats=c.causal_pairs(r,[obd(1600000000)],0)
 assert pairs==[] and stats['eligible_radar']==11
 pairs,_=c.causal_pairs(r,[obd(0)],0);assert pairs==[]

def test_future_radar_does_not_change_past_feature():
 r=[radar(t) for t in range(0,1500000001,50000000)];o=[obd(t) for t in range(0,1500000001,50000000)]
 pairs,_=c.causal_pairs(r,o,0);later,_=c.causal_pairs(r+[radar(1600000000,100)],o+[obd(1600000000)],0)
 assert len(pairs)==11 and pairs==later[:11]
 assert all(p['feature_raw']==2 and p['label_mps']==2 for p in pairs)

def test_quality_failures_remain_in_coverage_denominator():
 r=[radar(t) for t in range(0,1500000001,50000000)];r[-1]['speed_raw']=float('nan');o=[obd(t) for t in range(0,1500000001,50000000)]
 pairs,stats=c.causal_pairs(r,o,0)
 assert stats['eligible_radar']==11 and len(pairs)==10
 assert stats['coverage']==10/11

def test_header_time_cannot_be_future_and_obd_units_must_be_known():
 r=[radar(t) for t in range(0,1500000001,50000000)]
 assert c.causal_pairs(r,[obd(t,header=t+1) for t in range(0,1500000001,50000000)],0)[0]==[]
 assert c.causal_pairs(r,[obd(t,unit='mph') for t in range(0,1500000001,50000000)],0)[0]==[]

def test_fit_uses_fit_rows_only_and_prediction_never_uses_labels():
 fit=[{'feature_raw':x,'label_mps':2*x+1} for x in (0,2,4)]
 model=c.fit_affine(fit);assert model=={'slope':2.0,'intercept':1.0}
 assert c.predict(model,3)==7
 assert c.fit_affine([{'feature_raw':1,'label_mps':0}]*3) is None

def test_negative_slope_rejected_instead_of_becoming_witness():
 assert c.fit_affine([{'feature_raw':0,'label_mps':5},{'feature_raw':5,'label_mps':0}]) is None

def test_message_decoder_preserves_nanoseconds_units_and_signed_radar():
 def string(s):return struct.pack('<I',len(s))+s.encode()
 header=struct.pack('<III',9,1700000000,123456789)+string('frame')
 o=c.decode_message('obd',header+struct.pack('<d',36.0)+string('kph'),1700000000123456790)
 assert o['header_time_ns']==1700000000123456789 and o['speed_raw']==36 and o['unit']=='kph'
 data=struct.pack('<f',-3.5)+string('outbound')+struct.pack('<f',1.0)+string('sensor')+struct.pack('<f',1)+header+struct.pack('<IIII',0,0,0,0)+struct.pack('<f',0)
 r=c.decode_message('radar',data,1700000000123456790)
 assert r['header_time_ns']==o['header_time_ns'] and r['speed_raw']==-3.5
 with pytest.raises(ValueError):c.decode_message('radar',data[:-1],0)

def test_http_range_accepts_unknown_total_but_rejects_wrong_offsets_and_lengths():
 c.validate_range_response(206,'bytes 100-199/*','100',100,199,500)
 c.validate_range_response(206,'bytes 100-199/500','100',100,199,500)
 for status,cr,length in [(200,'bytes 100-199/*','100'),(206,'bytes 101-200/*','100'),(206,'bytes 100-199/501','100'),(206,'bytes 100-199/*','99')]:
  with pytest.raises(ValueError):c.validate_range_response(status,cr,length,100,199,500)
