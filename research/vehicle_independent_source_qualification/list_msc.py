from pathlib import Path
import urllib.request,re,json,hashlib,html
R=Path(__file__).resolve().parent
def folder(ident,name):
 url="https://drive.google.com/drive/folders/"+ident
 with urllib.request.urlopen(url,timeout=25) as r:b=r.read(4*1024*1024+1)
 assert len(b)<=4*1024*1024
 (R/(name+".html")).write_bytes(b)
 s=b.decode("utf-8")
 rows=[]
 for m in re.finditer(r'<[^>]*data-id="([^"]+)"[^>]*data-tooltip="([^"]+)"[^>]*>',s):
  rows.append({"id":m[1],"title":html.unescape(m[2])})
 print(name,json.dumps(rows),flush=True)
 return {"url":url,"sha256":hashlib.sha256(b).hexdigest(),"entries":rows}
out={}
out["urban"]=folder("1F6PWPRucCt4JMVAoAzBQIi371MkDHKzh","msc_urban")
for row in out["urban"]["entries"]:
 if any(x in row["title"].lower() for x in ["d0","calib"]):
  out[row["id"]]=folder(row["id"],"msc_"+row["id"])
(R/"msc_listing.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
