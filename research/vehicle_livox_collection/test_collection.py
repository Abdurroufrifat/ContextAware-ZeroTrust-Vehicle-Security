import io,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
import collection_core as k
from unittest.mock import patch

def fields(v):
 return b''.join(struct.pack('<I',len(key)+1+len(value))+key.encode()+b'='+value for key,value in v.items())
def record(h,b):
 head=fields(h);return struct.pack('<I',len(head))+head+struct.pack('<I',len(b))+b
def payload(acc=(0.,0.,1.)):
 return struct.pack('<IIII',1,10,0,5)+b'frame'+struct.pack('<37d',*([0.,0.,0.,1.]+[0.]*9+[1.,2.,3.]+[0.]*9+list(acc)+[0.]*9))
def chunk():
 b=record({'op':b'\x02','conn':struct.pack('<I',1),'time':struct.pack('<II',10,0)},payload())
 b+=record({'op':b'\x02','conn':struct.pack('<I',2),'time':struct.pack('<II',10,0)},b'EXCLUDED_REFERENCE_BYTES')
 return record({'op':b'\x05','compression':b'none','size':struct.pack('<I',len(b))},b)
class CollectionTests(unittest.TestCase):
 def call(self,name,*args,**kw):
  fn=getattr(k,name,None);self.assertTrue(callable(fn),'collection behavior missing: '+name);return fn(*args,**kw)
 def chunks(self):return [{'position':4096+i*100,'start_ns':100+i*100,'end_ns':200+i*100,'counts':{1:1}} for i in range(3)]
 def meta(self):return {'connections':[{'conn':1,'topic':'/livox/imu','type':'sensor_msgs/Imu','md5sum':'6a62c6daae103f4ff57a132d6f95cec2'},{'conn':2,'topic':'/novatel/imu','type':'sensor_msgs/Imu','md5sum':'6a62c6daae103f4ff57a132d6f95cec2'}]}
 def rows(self):return [{'bag_ns':10_000_000_000,'payload':payload(),'chunk_index':3,'record_ordinal':0}]
 def test_legacy_64k_header_capture_is_supported(self):
  self.assertEqual(self.call('inventory','Urban01_day')[2]['imu_topics']['/livox/imu']['indexed_messages'],429831)
 def test_expected_source_allowed(self):self.assertTrue(self.call('allowed_source','Urban03/2023-11-23-19-12-16-Night'))
 def test_reserved_source_forbidden(self):self.assertFalse(self.call('allowed_source','Urban02/2023-11-23-17-53-52-Night'))
 def test_unknown_variant_forbidden(self):self.assertFalse(self.call('allowed_source','Urban03/other'))
 def test_chunk_overlap_is_inclusive(self):self.assertEqual([v['chunk_index'] for v in self.call('select_chunks',self.chunks(),[[200,300]],4396,5000)],[0,1,2])
 def test_byte_range_stops_before_next_chunk(self):self.assertEqual(self.call('select_chunks',self.chunks(),[[101,199]],4396,5000)[0]['end'],4195)
 def test_final_range_stops_before_index(self):self.assertEqual(self.call('select_chunks',self.chunks(),[[350,351]],4396,5000)[0]['end'],4395)
 def test_duplicate_overlaps_do_not_repeat_chunks(self):self.assertEqual(len(self.call('select_chunks',self.chunks(),[[100,150],[140,180]],4396,5000)),1)
 def test_no_covered_chunk_rejected(self):
  with self.assertRaises(ValueError):self.call('select_chunks',self.chunks(),[[500,600]],4396,5000)
 def test_reversed_interval_rejected(self):
  with self.assertRaises(ValueError):self.call('select_chunks',self.chunks(),[[200,100]],4396,5000)
 def test_duplicate_positions_rejected(self):
  v=self.chunks();v[1]['position']=v[0]['position']
  with self.assertRaises(ValueError):self.call('select_chunks',v,[[100,300]],4396,5000)
 def embedded(self,topic='/livox/imu',kind='sensor_msgs/Imu',md5='6a62c6daae103f4ff57a132d6f95cec2',cid=1):
  body=next(k.c.records(chunk()))[1];declaration=record({'op':b'\x07','conn':struct.pack('<I',cid),'topic':topic.encode()},fields({'type':kind.encode(),'md5sum':md5.encode()}));body=declaration+body;return record({'op':b'\x05','compression':b'none','size':struct.pack('<I',len(body))},body)
 def test_embedded_canonical_connection_accepted(self):self.assertEqual(len(self.call('decode_chunk',self.embedded(),{'chunk_index':3,'counts':{1:1,2:1}},self.meta())),1)
 def test_embedded_wrong_schema_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_chunk',self.embedded(kind='wrong',md5='wrong'),{'chunk_index':3,'counts':{1:1,2:1}},self.meta())
 def test_embedded_wrong_topic_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_chunk',self.embedded(topic='/novatel/imu'),{'chunk_index':3,'counts':{1:1,2:1}},self.meta())
 def test_embedded_unknown_connection_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_chunk',self.embedded(cid=999),{'chunk_index':3,'counts':{1:1,2:1}},self.meta())
 def freeze_fixture(self,root):
  k.write(root/'protocol.json',{'routes':k.ROUTES,'history_prefix_ns':k.PREFIX_NS,'ttl_ns':k.a.TTL,'topic':'/livox/imu','model_training':False,'promotion_allowed':False,'input_sha256':{}});k.write(root/'acquisition_plan.json',{})
  for name in ['collection_core.py','collect_livox.py','verify_collection.py']:(root/name).write_text('fixture')
 def test_measurements_forbidden_without_persisted_lock(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.freeze_fixture(root)
   with patch.object(k,'ROOT',root):
    with self.assertRaises(ValueError):self.call('freeze')
 def test_created_lock_can_be_verified_before_collection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.freeze_fixture(root)
   with patch.object(k,'ROOT',root):
    expected=self.call('freeze',create=True);self.assertTrue((root/'analysis_lock.json').is_file());self.assertEqual(self.call('freeze'),expected)
 def test_only_livox_payload_extracted(self):
  rows=self.call('decode_chunk',chunk(),{'chunk_index':3,'counts':{1:1,2:1}},self.meta());self.assertEqual(rows,self.rows())
 def test_all_connection_counts_required(self):
  with self.assertRaises(ValueError):self.call('decode_chunk',chunk(),{'chunk_index':3,'counts':{1:1}},self.meta())
 def test_selected_schema_must_match(self):
  m=self.meta();m['connections'][0]['md5sum']='other'
  with self.assertRaises(ValueError):self.call('decode_chunk',chunk(),{'chunk_index':3,'counts':{1:1,2:1}},m)
 def test_truncated_chunk_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_chunk',chunk()[:-1],{'chunk_index':3,'counts':{1:1,2:1}},self.meta())
 def test_raw_payload_trailing_zeros_preserved(self):
  arr,report=self.call('pack_raw',self.rows());length=int(arr['payload_length'][0]);self.assertEqual(arr['payload_bytes'][0,:length].tobytes(),payload());self.assertEqual(report['records'],1)
 def test_duplicate_arrivals_rejected(self):
  with self.assertRaises(ValueError):self.call('pack_raw',self.rows()*2)
 def test_invalid_vector_kept_in_raw_export(self):
  rows=self.rows();rows[0]['payload']=payload((float('nan'),0,1));arr,report=self.call('pack_raw',rows);self.assertEqual(report['records'],1);self.assertEqual(report['usable_records'],0);self.assertFalse(arr['usable'][0])
 def test_empty_measurement_export_rejected(self):
  with self.assertRaises(ValueError):self.call('pack_raw',[])
 def test_unknown_covariances_reported(self):
  arr,report=self.call('pack_raw',self.rows());self.assertEqual(report['unknown_acceleration_covariance_records'],1);self.assertFalse(report['physical_unit_calibration_verified'])
 def align_data(self):return {'bag_ns':np.array([10_000_000_000,10_020_000_000],dtype=np.int64),'header_ns':np.array([10_000_000_000,10_040_000_000],dtype=np.int64),'usable':np.array([True,True]),'acceleration_norm':np.array([9.8,9.9]),'angular_velocity_norm':np.array([.1,.2])}
 def test_missing_prior_sample_retained(self):self.assertEqual(self.call('align',self.align_data(),np.array([9_999_999_999],dtype=np.int64))['status'].tolist(),[1])
 def test_future_newest_has_no_fallback(self):
  out=self.call('align',self.align_data(),np.array([10_030_000_000],dtype=np.int64));self.assertEqual(out['status'].tolist(),[3]);self.assertTrue(np.isnan(out['acceleration_norm'][0]));self.assertEqual(out['latest_message'].tolist(),[1])
 def test_invalid_newest_has_no_fallback(self):
  a=self.align_data();a['header_ns'][1]=a['bag_ns'][1];a['usable'][1]=False;self.assertEqual(self.call('align',a,np.array([10_030_000_000],dtype=np.int64))['status'].tolist(),[4])
 def test_both_clock_ttl_enforced(self):
  a=self.align_data();a['header_ns'][1]=10_000_000_000;self.assertEqual(self.call('align',a,np.array([10_100_000_000,10_100_000_001],dtype=np.int64))['status'].tolist(),[0,2])
 def test_unsorted_targets_rejected(self):
  with self.assertRaises(ValueError):self.call('align',self.align_data(),np.array([2,1],dtype=np.int64))
 def test_cache_bound_to_exact_source_and_range(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'chunk.bin';p.write_bytes(b'abc');q={'url':'u','start':1,'end':3,'total':5};receipt={**q,'sha256':__import__('hashlib').sha256(b'abc').hexdigest(),'bytes':3};p.with_suffix('.json').write_text(json.dumps(receipt));self.assertEqual(self.call('cached_range',p,q),b'abc')
 def test_corrupted_cache_is_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'chunk.bin';p.write_bytes(b'bad');q={'url':'u','start':1,'end':3,'total':5};p.with_suffix('.json').write_text(json.dumps({**q,'sha256':'bad','bytes':3}));self.assertIsNone(self.call('cached_range',p,q))
 def test_wrong_source_cache_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'chunk.bin';p.write_bytes(b'abc');q={'url':'u','start':1,'end':3,'total':5};p.with_suffix('.json').write_text(json.dumps({**q,'url':'other'}))
   with self.assertRaises(ValueError):self.call('cached_range',p,q)
 def test_atomic_write_creates_complete_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'out';self.call('atomic',p,b'complete');self.assertEqual(p.read_bytes(),b'complete');self.assertFalse(p.with_name('out.part').exists())
if __name__=='__main__':unittest.main(verbosity=2)
