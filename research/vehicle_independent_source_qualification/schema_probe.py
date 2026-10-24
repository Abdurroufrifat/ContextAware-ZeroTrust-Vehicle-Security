from pathlib import Path
import urllib.request,re,json,hashlib,html
R=Path(__file__).resolve().parent
def fetch(url,name,cap=4*1024*1024):
 req=urllib.request.Request(url,headers={"User-Agent":"CO-ACT-CPS-source-schema-screen/1.0"})
 with urllib.request.urlopen(req,timeout=25) as r:
  b=r.read(cap+1);status=r.status;final=r.url;ct=r.headers.get("Content-Type")
 if len(b)>cap:raise ValueError("bounded schema byte cap exceeded")
 (R/name).write_bytes(b)
 return b,{"url":url,"final_url":final,"http_status":status,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"content_type":ct}
out={}
try:
 b,out["pcd_listing"]=fetch("https://drive.google.com/drive/folders/1TxcHnrLjfebSJrdda7GQEIVHuPoZ2ptz","msc_pcd_listing.html")
 s=b.decode("utf-8")
 rows=[{"id":m[1],"title":html.unescape(m[2])} for m in re.finditer(r'<[^>]*data-id="([^"]+)"[^>]*data-tooltip="([^"]+)"[^>]*>',s)]
 out["pcd_entries"]=rows
 print("PCD entries",len(rows),json.dumps(rows[:3]),flush=True)
except Exception as e:out["pcd_listing"]={"error":str(e)}
for name,ident in [("radar_times","1U1lNoygxmI_YrwPuQWhxJnwb9TUC_MQb"),("rtk_schema","1Td1TSMRfXE5BitiNFyGpb-aOLSr3EWZZ")]+([("first_pcd_schema",out["pcd_entries"][0]["id"])] if out.get("pcd_entries") else []):
 try:
  b,meta=fetch("https://drive.google.com/uc?export=download&id="+ident,name+".sample")
  if b.lstrip().startswith(b"<"):meta["not_measurement"]="HTML response, no numeric payload decoded"
  else:
   text=b.decode("utf-8",errors="replace")
   meta["first_lines"]=text.splitlines()[:12]
  out[name]=meta
 except Exception as e:out[name]={"error":str(e)}
 print(name,json.dumps(out[name]),flush=True)
(R/"schema_probe.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
