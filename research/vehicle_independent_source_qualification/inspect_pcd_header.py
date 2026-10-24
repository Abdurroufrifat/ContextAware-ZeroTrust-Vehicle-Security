from pathlib import Path
import json,re,urllib.request,urllib.parse,hashlib
R=Path(__file__).resolve().parent
s=(R/"first_pcd_schema.sample").read_text(encoding="utf-8")
params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s))
assert params["id"]=="1kWkvZxny1k5nDtsFYFQLiWYWZldzeXib"
url="https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params)
with urllib.request.urlopen(url,timeout=25) as r:
 b=r.read(4*1024*1024+1)
 meta={"url":url,"final_url":r.url,"http_status":r.status,"content_type":r.headers.get("Content-Type")}
assert len(b)<=4*1024*1024
meta.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
if b.lstrip().startswith(b"<"):
 (R/"pcd_download_response.html").write_bytes(b)
 meta["error"]="Still HTML; not a PCD"
else:
 (R/"msc_first_radar.pcd").write_bytes(b)
 end=b.find(b"DATA ")
 assert end>=0
 end=b.find(b"\n",end)
 header=b[:end+1].decode("ascii")
 meta["header"]=header
 fields=re.search(r"(?m)^FIELDS (.*)$",header)[1].split()
 meta["fields"]=fields
 meta["has_named_raw_doppler"]=bool(set(fields)&{"doppler","velocity","v_doppler","radial_velocity"})
(R/"pcd_schema_result.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print(json.dumps(meta,indent=2))
