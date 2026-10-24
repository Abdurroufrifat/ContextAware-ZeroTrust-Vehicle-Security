import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"COACT-development-audit"}),timeout=30) as f: return json.load(f)
info=get("https://api.github.com/repos/UniBwTAS/velodyne")
branches=get("https://api.github.com/repos/UniBwTAS/velodyne/branches?per_page=100")
tree=get("https://api.github.com/repos/UniBwTAS/velodyne/git/trees/"+info["default_branch"]+"?recursive=1")
report=dict(repository=info["html_url"],branch=info["default_branch"],tree_sha=tree["sha"],license=info.get("license"),branches=branches,tree=tree["tree"])
(ROOT/"velodyne_driver_metadata.json").write_text(json.dumps(report,indent=2)+"\n")
print("DEFAULT",info["default_branch"],"SHA",tree["sha"])
print("BRANCHES",json.dumps([x["name"] for x in branches]))
print("SOURCES",json.dumps([x["path"] for x in tree["tree"] if x["type"]=="blob" and (x["path"].endswith((".cpp",".h",".hpp",".launch",".md")) or "LICENSE" in x["path"]) and "calibration" not in x["path"]]))
