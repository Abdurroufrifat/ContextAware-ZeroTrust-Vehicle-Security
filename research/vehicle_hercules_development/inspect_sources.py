"""Metadata-only inventory for the three user-provided HeRCULES folders."""
from pathlib import Path
from html.parser import HTMLParser
import urllib.request,json,hashlib,re
R=Path(__file__).resolve().parent
class Listing(HTMLParser):
 def __init__(self):super().__init__();self.rows={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if "data-id" in a and "data-tooltip" in a:
   self.rows[a["data-id"]]={"id":a["data-id"],"title":a["data-tooltip"]}
def fetch(url,name):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"CO-ACT-CPS-development/1.0"}),timeout=30) as r:
  b=r.read(4*1024*1024+1);final=r.url
 if len(b)>4*1024*1024:raise ValueError("metadata cap")
 (R/name).write_bytes(b)
 return b,{"url":url,"final_url":final,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
roots={"Library01":"1ujW3lgM2Jac6LqL-iG85qAxQ58g5rDPG","Library02":"1b0McYxyl_Iqz68wfaUNXwKZGfvTTfr31","Library03":"1JvlDqtMrDH6wVZFD2SfDTjKf9gNwjS3o"}
out={}
for seq,ident in roots.items():
 try:
  b,meta=fetch("https://drive.google.com/drive/folders/"+ident,seq+"_listing.html")
  parser=Listing();parser.feed(b.decode("utf-8"));meta["entries"]=list(parser.rows.values());out[seq]=meta
 except Exception as e:out[seq]={"error":str(e)}
 print(seq,json.dumps(out[seq]),flush=True)
for repo in ["HeRCULES_file_player","HeRCULES_Pointcloud_Toolbox"]:
 try:
  b,meta=fetch("https://api.github.com/repos/hanjun815/"+repo+"/git/trees/main?recursive=1",repo+"_tree.json")
  tree=json.loads(b);meta["commit"]=tree["sha"];meta["paths"]=[x["path"] for x in tree["tree"] if x["type"]=="blob"]
  out[repo]=meta
 except Exception as e:out[repo]={"error":str(e)}
 print(repo,json.dumps(out[repo]),flush=True)
(R/"source_inventory.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
