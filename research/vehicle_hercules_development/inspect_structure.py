from pathlib import Path
from html.parser import HTMLParser
import urllib.request,json,hashlib,zipfile,io
R=Path(__file__).resolve().parent
class Listing(HTMLParser):
 def __init__(self):super().__init__();self.rows={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if "data-id" in a and "data-tooltip" in a:self.rows[a["data-id"]]={"id":a["data-id"],"title":a["data-tooltip"]}
def fetch(url,name,cap=4*1024*1024):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"CO-ACT-CPS-development/1.0"}),timeout=30) as r:b=r.read(cap+1)
 if len(b)>cap:raise ValueError("size cap")
 (R/name).write_bytes(b)
 return b,{"url":url,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
inventory=json.loads((R/"source_inventory.json").read_text());out={}
for seq in ["Library01","Library02","Library03"]:
 out[seq]={}
 for e in inventory[seq]["entries"]:
  name=e["title"].removesuffix(" Shared folder")
  if name not in ["Radar","sensor_data","Calibration","PR_GT"]:continue
  try:
   b,meta=fetch("https://drive.google.com/drive/folders/"+e["id"],seq+"_"+name+"_listing.html")
   parser=Listing();parser.feed(b.decode("utf-8"));meta["entries"]=list(parser.rows.values())
   out[seq][name]=meta
  except Exception as ex:out[seq][name]={"error":str(ex)}
  print(seq,name,json.dumps(out[seq][name]),flush=True)
(R/"structure_inventory.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
for repo in ["HeRCULES_file_player","HeRCULES_Pointcloud_Toolbox"]:
 try:
  b,meta=fetch("https://codeload.github.com/hanjun815/"+repo+"/zip/refs/heads/main",repo+"_source.zip",8*1024*1024)
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   files=[n for n in z.namelist() if not n.endswith("/")];meta["files"]=files
   dest=R/"official_source"/repo;dest.mkdir(parents=True,exist_ok=True)
   for n in files:
    rel=Path(*Path(n).parts[1:])
    if rel.suffix.lower() not in [".py",".cpp",".h",".hpp",".md",".txt",".xml",""] :continue
    assert ".." not in rel.parts
    p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
  (R/(repo+"_source_lock.json")).write_text(json.dumps(meta,indent=2),encoding="utf-8")
  print(repo,json.dumps(meta),flush=True)
 except Exception as ex:print(repo,"ERROR",str(ex),flush=True)
