from pathlib import Path
import urllib.request,re,json,hashlib,html
R=Path(__file__).resolve().parent
folders={"radar":"1ImyNG88VHowMWWIYBShjGkk7cOvc3Iiq","navigation":"1Imc8crM7Lgyi4qljdnuZauEZz4AphwRo","calibration":"1IigTN2WlVrl1cDB_IXswjUoW21Bk9YbG"}
out={}
for name,ident in folders.items():
 url="https://drive.google.com/drive/folders/"+ident
 try:
  with urllib.request.urlopen(url,timeout=25) as r:b=r.read(4*1024*1024+1)
  assert len(b)<=4*1024*1024
  (R/("msc_"+name+".html")).write_bytes(b)
  s=b.decode("utf-8")
  rows=[{"id":m[1],"title":html.unescape(m[2])} for m in re.finditer(r'<[^>]*data-id="([^"]+)"[^>]*data-tooltip="([^"]+)"[^>]*>',s)]
  out[name]={"url":url,"sha256":hashlib.sha256(b).hexdigest(),"entries":rows}
 except Exception as e:out[name]={"url":url,"error":str(e)}
 print(name,json.dumps(out[name]),flush=True)
(R/"msc_metadata_listing.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
