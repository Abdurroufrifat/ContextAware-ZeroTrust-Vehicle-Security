"""Collect exactly the frozen Livox ranges and export serialized IMU messages."""
import time,urllib.error,urllib.request,zipfile
from collection_core import *

def fetch(q,path,opener=urllib.request.urlopen):
 if q['url'] not in {URL_PREFIX+v+'/imus.bag' for v in ROUTES.values()} or any(type(q[k]) is not int for k in ['start','end','total']) or not 4096<=q['start']<=q['end']<q['total'] or q['end']-q['start']+1>16_777_216:raise ValueError('unapproved measurement source/range')
 blob=cached_range(path,q)
 if blob is not None:return blob
 request=urllib.request.Request(q['url'],headers={'Range':f"bytes={q['start']}-{q['end']}",'Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-Livox-fixed-cohort/1.0'})
 for attempt in range(3):
  try:
   with opener(request,timeout=30) as response:
    c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),q['start'],q['end'],q['total']);blob=response.read(q['end']-q['start']+2);span=response.headers.get('Content-Range')
   if len(blob)!=q['end']-q['start']+1:raise ValueError('incomplete chunk transfer; no bytes committed')
   atomic(path,blob);receipt={**q,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'content_range':span};atomic(path.with_suffix('.json'),(json.dumps(receipt,indent=2)+'\n').encode());return blob
  except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
   if attempt==2:raise
   print('Network retry',attempt+2,'of3:',type(exc).__name__,flush=True);time.sleep(2)
 raise RuntimeError('unreachable transfer state')

def build_upload(folder,payloads,out):
 buffer=io.BytesIO();names=set()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(payloads):
   path=f.resolve()
   if not path.is_relative_to(folder.resolve()) or not path.is_file():raise ValueError('unsafe export payload path')
   name='livox_collection/'+path.relative_to(folder.resolve()).as_posix()
   if name in names:raise ValueError('duplicate export member')
   names.add(name);z.writestr(name,path.read_bytes())
 blob=buffer.getvalue()
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  if z.testzip() is not None:raise ValueError('upload ZIP integrity failed')
 atomic(out,blob)

def export_source(key,source,targets):
 folder=ROOT/'runtime_chunks'/key;meta=read(ROOT/'metadata'/key/'source.json');rows=[];receipts=[]
 for ordinal,q in enumerate(source['ranges']):
  request={'url':source['url'],'start':q['start'],'end':q['end'],'total':source['size']};path=folder/f"chunk_{q['chunk_index']:04d}.bin";blob=fetch(request,path);rows.extend(decode_chunk(blob,q,meta));receipts.append({'chunk_index':q['chunk_index'],**read(path.with_suffix('.json'))})
  if ordinal%20==0 or ordinal+1==len(source['ranges']):print(key,ordinal+1,'/',len(source['ranges']),'chunks checked',flush=True)
 if len(rows)!=source['indexed_livox_messages_in_selected_chunks']:raise ValueError('selected source count differs from fixed index')
 arrays,report=pack_raw(rows);report['route']=source['route'];report['source_only_alignment']={}
 for name in sorted(targets):
  if not name.startswith(key+'__') or not name.endswith('_times_ns'):continue
  scope=name[len(key)+2:-9];aligned=align(arrays,targets[name]);report['source_only_alignment'][scope]={'target_count':len(aligned['status']),'available_count':int(np.sum(aligned['status']==0)),'status_counts':{label:int(np.sum(aligned['status']==code)) for code,label in a.STATUS.items()}}
  arrays.update({'alignment__'+scope+'__'+k:v for k,v in aligned.items()})
 for name,value in targets.items():
  if name.startswith(key+'__') and not name.endswith('_times_ns'):arrays['identity__'+name[len(key)+2:]]=value
 output=ROOT/'runtime_export';output.mkdir(exist_ok=True);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);atomic(output/(key+'.npz'),buffer.getvalue());write(output/(key+'_quality.json'),report);write(output/(key+'_receipts.json'),receipts)
 return report,[output/(key+suffix) for suffix in ['.npz','_quality.json','_receipts.json']]

def main():
 from verify_collection import audit
 audit();lock=freeze();plan=read(ROOT/'acquisition_plan.json');print('Fixed download budget:',round(plan['total_download_bytes']/1_000_000),'MB; verified cached chunks will be reused.',flush=True)
 with np.load(ROOT/'cohort_targets.npz',allow_pickle=False) as z:targets={k:z[k].copy() for k in z.files}
 payloads=[];reports={}
 for key,source in plan['sources'].items():
  print('Collecting',source['route'],flush=True);reports[key],files=export_source(key,source,targets);payloads.extend(files)
 output=ROOT/'runtime_export';write(output/'quality_summary.json',{'scope':'Source quality and causal availability only; no predictive accuracy or security permission','sources':reports,'model_training_started':False,'reserved_final_opened':False,'witness_enabled':False,'production_policy_changed':False});payloads.append(output/'quality_summary.json');write(output/'collection_identity.json',{'analysis_lock':lock,'input_plan_sha256':digest(ROOT/'acquisition_plan.json'),'files_sha256':{f.name:digest(f) for f in payloads},'full_bag_checksums_verified':False,'source_http_receipts_are_not_authenticated_live_time':True});payloads.append(output/'collection_identity.json')
 out=ROOT/'Upload_Livox_Cohort.zip';build_upload(output,payloads,out);print('LIVOX COLLECTION COMPLETE. No model trained or reserved data opened.',flush=True);print('UPLOAD THIS ZIP:',out,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:print('LIVOX COLLECTION INCOMPLETE:',type(exc).__name__,str(exc),flush=True);raise
