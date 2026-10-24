"""Capture only missing fitting IMU bag headers/indexes; create upload ZIP."""
import io,os,urllib.request,zipfile
from aux_core import *

def atomic(path,blob):
 part=path.with_name(path.name+'.part')
 with part.open('wb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
 os.replace(part,path)

def get(url,start,end,size,part):
 allowed_metadata_range(start,end,size,part);request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-IMU-metadata/1.0'})
 with urllib.request.urlopen(request,timeout=30) as response:
  c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),start,end,size);blob=response.read(end-start+2);range_text=response.headers.get('Content-Range')
 if len(blob)!=end-start+1:raise ValueError('metadata transfer byte count differs')
 return blob,{'start':start,'end':end,'bytes':len(blob),'content_range':range_text,'sha256':hashlib.sha256(blob).hexdigest()}

def capture(group,route):
 check_metadata_source(route,'imus.bag');catalog=read(RESEARCH/'vehicle_night_training/metadata'/(group+'.json'));entry=next(v for v in catalog['contents'] if v['name']=='imus.bag');expected='/1/published/publication_1084/submitted_data/'+route+'/imus.bag'
 if entry['path']!=expected or type(entry['size']) is not int or entry['size']<=4096:raise ValueError('pending source catalog differs')
 folder=ROOT/'runtime_metadata'/group;folder.mkdir(parents=True,exist_ok=True);url='https://www.frdr-dfdr.ca/repo/files'+expected;receipt_path=folder/'receipt.json'
 if receipt_path.exists():
  receipt=read(receipt_path)
  if receipt['source_url']!=url or receipt['source_size']!=entry['size']:raise ValueError('cached metadata source identity differs')
  for name,h in receipt['files_sha256'].items():
   if name not in ['imus.bag.header','imus.bag.index','source.json'] or digest(folder/name)!=h:raise ValueError('cached metadata bytes changed')
  if set(receipt['files_sha256'])!={'imus.bag.header','imus.bag.index','source.json'}:raise ValueError('incomplete metadata cache')
  inventory(read(folder/'source.json'),(folder/'imus.bag.index').read_bytes());return folder
 head,head_receipt=get(url,0,4095,entry['size'],'header')
 if not head.startswith(b'#ROSBAG V2.0\n'):raise ValueError('unsupported bag header')
 header=c.fields(head[17:17+c.u32(head,13)]);index_pos=struct.unpack('<Q',header['index_pos'])[0];index,index_receipt=get(url,index_pos,entry['size']-1,entry['size'],'index');connections=[]
 for h,blob in c.records(index):
  if h['op']==b'\x07':
   fields=c.fields(blob);connections.append({'conn':c.u32(h['conn']),'topic':h['topic'].decode(),'type':fields['type'].decode(),'md5sum':fields['md5sum'].decode(),'message_definition':fields['message_definition'].decode()})
 meta={'file':'imus.bag','url':url,'size':entry['size'],'index_pos':index_pos,'connections':connections};inventory(meta,index);atomic(folder/'imus.bag.header',head);atomic(folder/'imus.bag.index',index);write(folder/'source.json',meta)
 write(receipt_path,{'route':route,'source_url':url,'source_size':entry['size'],'header_index_only':True,'measurement_payloads_requested':False,'publisher_full_bag_sha256_verified':False,'ranges':[head_receipt,index_receipt],'files_sha256':{n:digest(folder/n) for n in ['imus.bag.header','imus.bag.index','source.json']}});return folder

def main():
 from verify_auxiliary import audit
 audit();freeze();folders=[]
 for group,route in PENDING.items():
  print('Reading fitting IMU header/index:',route,flush=True);folders.append(capture(group,route))
 summary={'scope':'Only two fitting IMU headers/indexes; no measurement or reference payloads','sources':{},'reserved_final_opened':False,'model_training_started':False}
 for folder in folders:summary['sources'][folder.name]=inventory(read(folder/'source.json'),(folder/'imus.bag.index').read_bytes())
 write(ROOT/'runtime_metadata/summary.json',summary);payloads=[ROOT/'runtime_metadata/summary.json']+[f for folder in folders for f in sorted(folder.iterdir()) if f.name in ['imus.bag.header','imus.bag.index','source.json','receipt.json']];buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for f in payloads:z.writestr('auxiliary_imu_metadata/'+f.relative_to(ROOT/'runtime_metadata').as_posix(),f.read_bytes())
 blob=buffer.getvalue()
 with zipfile.ZipFile(io.BytesIO(blob)) as z:
  if z.testzip() is not None:raise ValueError('metadata ZIP integrity failed')
 out=ROOT/'Upload_Auxiliary_IMU_Metadata.zip';atomic(out,blob);print('METADATA CAPTURE COMPLETE. No model fitted or reserved data opened.',flush=True);print('UPLOAD THIS ZIP:',out,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:print('METADATA CAPTURE INCOMPLETE:',type(exc).__name__,str(exc),flush=True);raise
