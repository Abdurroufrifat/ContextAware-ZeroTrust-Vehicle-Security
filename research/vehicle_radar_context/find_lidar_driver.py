import base64,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"COACT-development-audit"}),timeout=30) as f: return json.load(f)
repos=get("https://api.github.com/orgs/UniBwTAS/repos?per_page=100")
print("REPOS",json.dumps([r["name"] for r in repos]))
metadata=json.loads((ROOT/"lidar_primary_source_metadata.json").read_text())
tree=metadata["continuous_clustering"]["tree"]
for name in ("scripts/clone_repositories_and_install_dependencies.sh","README.md"):
    blob=next(x for x in tree if x["path"]==name)
    content=base64.b64decode(get(blob["url"])["content"]).decode()
    print("SOURCE",name)
    for line in content.splitlines():
        if any(k in line.lower() for k in ("lidar","velodyne","clone","ethernet","timestamp")): print(line)
(ROOT/"lidar_driver_search.json").write_text(json.dumps(dict(organization_repositories=[r["name"] for r in repos]),indent=2)+"\n")
