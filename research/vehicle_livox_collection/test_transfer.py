import io,tempfile,unittest,zipfile
from pathlib import Path
import collection_core as k
import collect_livox as t
from unittest.mock import patch
class Response(io.BytesIO):
 status=206
 def __init__(self,b=b'abc',span='bytes 4096-4098/5000'):
  super().__init__(b);self.headers={'Content-Range':span,'Content-Length':'3'}
class TransferTests(unittest.TestCase):
 def call(self,name,*args,**kw):
  fn=getattr(t,name,None);self.assertTrue(callable(fn),'transfer behavior missing: '+name);return fn(*args,**kw)
 def q(self):return {'url':k.URL_PREFIX+k.ROUTES['Urban01_day']+'/imus.bag','start':4096,'end':4098,'total':5000}
 def test_valid_range_persists_and_resumes_without_network(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'chunk.bin';self.assertEqual(self.call('fetch',self.q(),p,opener=lambda *a,**kw:Response()),b'abc');self.assertEqual(self.call('fetch',self.q(),p,opener=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('network on verified cache'))),b'abc')
 def test_truncated_response_not_committed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'chunk.bin'
   with self.assertRaises(ValueError):self.call('fetch',self.q(),p,opener=lambda *a,**kw:Response(b'ab'))
   self.assertFalse(p.exists())
 def test_wrong_content_range_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.call('fetch',self.q(),Path(d)/'chunk.bin',opener=lambda *a,**kw:Response(span='bytes 4097-4099/5000'))
 def test_reserved_url_never_requested(self):
  q=self.q();q['url']=k.URL_PREFIX+'Urban02/2023-11-23-17-53-52-Night/imus.bag'
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.call('fetch',q,Path(d)/'chunk.bin',opener=lambda *a,**kw:Response())
 def test_real_cached_chunk_export_keeps_lineage_and_fit_identity(self):
  import hashlib,json,shutil
  probe=k.RESEARCH/'witness_calibration_probe';old=k.read(probe/'acquisition_plan.json');q=next(v for v in old['ranges'] if v['file']=='imus.bag');meta=k.read(k.ROOT/'metadata/Urban01_day/source.json');chunks=k.c.chunk_infos((probe/'source_index/imus.bag.index').read_bytes());key='Urban01_day';request={'url':meta['url'],'start':q['start'],'end':q['end'],'total':meta['size']};expected={'chunk_index':q['chunk_index'],'start':q['start'],'end':q['end'],'counts':chunks[q['chunk_index']]['counts']};source={'url':meta['url'],'size':meta['size'],'route':k.ROUTES[key],'ranges':[expected],'indexed_livox_messages_in_selected_chunks':sum(chunks[q['chunk_index']]['counts'].get(v['conn'],0) for v in meta['connections'] if v['topic']=='/livox/imu')}
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'runtime_chunks'/key;folder.mkdir(parents=True);md=root/'metadata'/key;md.mkdir(parents=True);k.write(md/'source.json',meta);path=folder/f"chunk_{q['chunk_index']:04d}.bin";blob=(probe/'evidence/chunks'/q['relative_cache']).read_bytes();path.write_bytes(blob);k.write(path.with_suffix('.json'),{**request,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()});rows=k.decode_chunk(blob,expected,meta);time=rows[0]['bag_ns'];targets={key+'__fit_times_ns':k.np.array([time],dtype=k.np.int64),key+'__fit_ids':k.np.array([123],dtype=k.np.int64)}
   with patch.object(t,'ROOT',root):report,files=t.export_source(key,source,targets)
   self.assertEqual(report['records'],len(rows));self.assertEqual(report['source_only_alignment']['fit']['target_count'],1)
   with k.np.load(root/'runtime_export'/('Urban01_day.npz'),allow_pickle=False) as z:
    self.assertEqual(z['identity__fit_ids'].tolist(),[123]);self.assertEqual(z['alignment__fit__target_ns'].tolist(),[time]);self.assertEqual(z['payload_bytes'][0,:int(z['payload_length'][0])].tobytes(),rows[0]['payload'])
 def test_upload_zip_preserves_exact_payloads(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'raw.npz').write_bytes(b'exact');out=root/'upload.zip';self.call('build_upload',root,[root/'raw.npz'],out)
   with zipfile.ZipFile(out) as z:self.assertIsNone(z.testzip());self.assertEqual(z.read('livox_collection/raw.npz'),b'exact')
 def test_upload_cannot_include_external_file(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'export';root.mkdir();external=Path(d)/'outside';external.write_bytes(b'outside')
   with self.assertRaises(ValueError):self.call('build_upload',root,[external],Path(d)/'upload.zip')
if __name__=='__main__':unittest.main(verbosity=2)
