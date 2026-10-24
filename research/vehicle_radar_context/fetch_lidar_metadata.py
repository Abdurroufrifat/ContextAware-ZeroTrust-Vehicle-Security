"""Read pinned primary source metadata; no data downloads."""
import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"COACT-development-audit"}),timeout=30) as f: return json.load(f)
report={}
for repo in ("tas_lidar","7V-Scanario","continuous_clustering"):
    try:
        info=get("https://api.github.com/repos/UniBwTAS/"+repo)
        tree=get("https://api.github.com/repos/UniBwTAS/"+repo+"/git/trees/"+info["default_branch"]+"?recursive=1")
        report[repo]=dict(repository=info["html_url"],tree_sha=tree["sha"],license=info.get("license"),tree=tree["tree"])
        print(repo,tree["sha"])
        print(json.dumps([x["path"] for x in tree["tree"] if x["type"]=="blob" and any(k in x["path"].lower() for k in ("lidar","velodyne","point_type","pointtype","readme","clone_repositories"))]))
    except Exception as e: print(repo,type(e).__name__,str(e))
(ROOT/"lidar_primary_source_metadata.json").write_text(json.dumps(report,indent=2)+"\n")
