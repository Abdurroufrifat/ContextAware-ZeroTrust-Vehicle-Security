from pathlib import Path
import urllib.request,urllib.parse,re,json,time
R=Path(__file__).resolve().parent
e=json.loads((R/"source_probe.json").read_text())["Library01"]
with urllib.request.urlopen("https://drive.google.com/uc?export=download&id="+e["archive_id"]+"&probe="+str(time.time_ns()),timeout=45) as r:
 s=r.read(1024*1024).decode();print("FIRST",r.status,r.headers.get("Content-Type"),flush=True)
params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s))
print("FIELDS",list(params),flush=True)
u="https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params)
req=urllib.request.Request(u+"&range_probe="+str(time.time_ns()),headers={"Range":"bytes=402653184-469762047","Cache-Control":"no-cache"})
with urllib.request.urlopen(req,timeout=45) as r:
 print("RESPONSE",r.status,r.headers.get("Content-Type"),r.headers.get("Content-Range"),r.headers.get("Content-Length"),flush=True)
 b=r.read(1024);print(b.decode(errors="replace") if "text/html" in r.headers.get("Content-Type","") else "binary",flush=True)
