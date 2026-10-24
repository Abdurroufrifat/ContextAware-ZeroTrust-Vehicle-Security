"""Metadata-frozen bounded acquisition from the author's public S3 bucket."""
from pathlib import Path
import ast,bisect,concurrent.futures,hashlib,json,urllib.request,urllib.parse,xml.etree.ElementTree as ET,time
R=Path(__file__).resolve().parent
SEQS=["boreas-2024-12-03-13-13","boreas-2024-12-05-14-25","boreas-2024-12-04-11-56"]
NS={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
def safe(path):
 path=Path(path);text=(str(path)+" "+str(path.resolve())).lower()
 if any(k in text for k in ("library03","rav4","vehicle_final_evaluation")) or ("urban02" in text and "night" in text):raise ValueError("Reserved path")
 if not path.resolve().is_relative_to(R):raise ValueError("Outside study")
 return path
def get(url,cap):
 with urllib.request.urlopen(url,timeout=25) as f:data=f.read(cap+1)
 if len(data)>cap:raise ValueError("Response exceeds size cap")
 return data
def entries(raw):
 root=ET.fromstring(raw)
 return [{"key":e.find("s:Key",NS).text,"bytes":int(e.find("s:Size",NS).text),"etag":e.find("s:ETag",NS).text.strip('"')} for e in root.findall("s:Contents",NS)]
# Read published split metadata without executing downloaded author code.
tree=ast.parse((R/"author_splits.py").read_text());official=set()
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("odom_test_rt","loc_test_rt") for t in node.targets):
  official.update(row[0] for row in ast.literal_eval(node.value))
assert not set(SEQS)&official
protocol_hash=hashlib.sha256((R/"PROTOCOL.md").read_bytes()).hexdigest()
if (R/"membership.json").exists():membership=json.loads((R/"membership.json").read_text());assert membership["protocol_sha256"]==protocol_hash
else:
 plan=[]
 for seq in SEQS:
  initial=entries((R/(seq+"_aeva_listing.xml")).read_bytes());t0=int(Path(initial[0]["key"]).stem)
  query={"list-type":"2","prefix":seq+"/aeva/","start-after":seq+"/aeva/"+str(t0+120000000-1)+".bin","max-keys":1000}
  raw=get("https://boreas.s3.amazonaws.com/?"+urllib.parse.urlencode(query),1000000)
  (R/(seq+"_selected_listing.xml")).write_bytes(raw)
  scans=entries(raw);times=[int(Path(e["key"]).stem) for e in scans]
  selected=[]
  for k in range(32):
   idx=bisect.bisect_left(times,t0+120000000+k*2000000);assert idx<len(times)
   selected.append({**scans[idx],"kind":"frame","target_us":t0+120000000+k*2000000})
  assert len({e["key"] for e in selected})==32
  reference=entries((R/(seq+"_applanix_listing.xml")).read_bytes())
  reference=[{**e,"kind":"reference"} for e in reference if Path(e["key"]).name in ("aeva_poses.csv","gt_errors.pdf")]
  assert len(reference)==2
  calibration=[{**e,"kind":"calibration"} for e in entries((R/(seq+"_calib_listing.xml")).read_bytes())]
  plan.extend(selected+reference+calibration)
 assert sum(e["kind"]=="frame" for e in plan)==96 and sum(e["bytes"] for e in plan)<=750000000 and max(e["bytes"] for e in plan)<=15000000
 membership={"protocol_sha256":protocol_hash,"sequences":SEQS,"official_test_sequences_excluded":True,"objects":plan,"planned_bytes":sum(e["bytes"] for e in plan)}
 (R/"membership.json").write_text(json.dumps(membership,indent=2)+"\n",encoding="utf-8")
 print("FROZEN",len(plan),"objects;",membership["planned_bytes"],"bytes",flush=True)
def fetch(e):
 path=safe(R/"data"/e["key"]);path.parent.mkdir(parents=True,exist_ok=True)
 error=None
 for attempt in range(2):
  try:
   if path.exists() and path.stat().st_size==e["bytes"]:data=path.read_bytes()
   else:
    data=get("https://boreas.s3.amazonaws.com/"+e["key"],15000000)
    if len(data)!=e["bytes"]:raise ValueError("Size mismatch")
    temp=path.with_suffix(path.suffix+".part");temp.write_bytes(data);temp.replace(path)
   return {**e,"sha256":hashlib.sha256(data).hexdigest(),"status":"downloaded"}
  except Exception as ex:error=str(ex)
 return {**e,"status":"failed","error":error}
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for index,item in enumerate(pool.map(fetch,membership["objects"]),1):
  out.append(item)
  if index%16==0:print("acquired",index,"/",len(membership["objects"]),flush=True)
(R/"data_lock.json").write_text(json.dumps({"protocol_sha256":protocol_hash,"objects":out},indent=2)+"\n",encoding="utf-8")
print(json.dumps({"downloaded":sum(e["status"]=="downloaded" for e in out),"failed":[e for e in out if e["status"]!="downloaded"],"bytes":sum(e["bytes"] for e in out if e["status"]=="downloaded")},indent=2),flush=True)
