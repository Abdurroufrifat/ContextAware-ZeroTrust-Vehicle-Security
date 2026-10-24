"""Bounded downloads of exact development-source members, never Library03."""
from pathlib import Path
import urllib.request,urllib.parse,hashlib,json,re,concurrent.futures,time
R=Path(__file__).resolve().parent
inventory=json.loads((R/"structure_inventory.json").read_text())
wanted={"Radar":{"Continental.tar.gz"},"sensor_data":{"continental_stamp.csv","inspva.csv","xsens_imu.csv"},"PR_GT":{"Continental_gt.txt"},"Calibration":{"Continental_LiDAR.txt","IMU_LiDAR.txt"}}
members=[]
for seq in ["Library01","Library02"]:
 for group,names in wanted.items():
  for e in inventory[seq][group]["entries"]:
   name=e["title"].rsplit(" ",1)[0] if not e["title"].endswith(" Compressed archive") else e["title"].removesuffix(" Compressed archive")
   if name in names:members.append({"sequence":seq,"group":group,"name":name,"id":e["id"],"cap":128*1024*1024 if name.endswith(".gz") else 32*1024*1024})
assert len(members)==14
(R/"acquisition_membership.json").write_text(json.dumps(members,indent=2),encoding="utf-8")
def open_download(ident):
 url="https://drive.google.com/uc?export=download&id="+ident
 r=urllib.request.urlopen(url,timeout=45)
 if "text/html" not in r.headers.get("Content-Type",""):return r
 b=r.read(1024*1024);r.close();s=b.decode("utf-8")
 params=dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]+)"',s))
 if params.get("id")!=ident or params.get("confirm")!="t":raise ValueError("HTML response, download not authorized or unavailable")
 return urllib.request.urlopen("https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(params),timeout=45)
def one(e):
 folder=R/"data"/e["sequence"]/e["group"];folder.mkdir(parents=True,exist_ok=True);p=folder/e["name"]
 failures=[]
 for attempt in range(3):
  try:
   if not p.exists():
    part=p.with_name(p.name+".partial")
    with open_download(e["id"]) as response,part.open("wb") as out:
     if "text/html" in response.headers.get("Content-Type",""):raise ValueError("unexpected HTML")
     total=0
     while True:
      b=response.read(1024*1024)
      if not b:break
      total+=len(b)
      if total>e["cap"]:raise ValueError("per-file cap")
      out.write(b)
     final_url=response.url
    part.replace(p)
   else:final_url="cached"
   digest=hashlib.sha256(p.read_bytes()).hexdigest()
   row={**e,"relative_path":str(p.relative_to(R)),"bytes":p.stat().st_size,"sha256":digest,"download_url":final_url,"earlier_errors":failures}
   print(e["sequence"],e["name"],row["bytes"],flush=True);return row
  except Exception as ex:
   failures.append(str(ex));print("retry",e["sequence"],e["name"],attempt+1,str(ex),flush=True)
 return {**e,"errors":failures}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(one,members):rows.append(row)
(R/"download_lock.json").write_text(json.dumps({"files":rows,"total_bytes":sum(x.get("bytes",0) for x in rows),"Library03_payloads_accessed":False},indent=2),encoding="utf-8")
print("ACQUIRED",sum("sha256" in x for x in rows),"of",len(rows),flush=True)
