import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    url="https://api.github.com/repos/UniBwTAS/tas_radar/git/trees/master?recursive=1"
    with urllib.request.urlopen(url,timeout=60) as inp: obj=json.load(inp)
    (ROOT/"driver_tree.json").write_text(json.dumps(obj,indent=2)+"\n")
    print("PINNED DRIVER TREE",obj["sha"])
    print("\n".join(x["path"] for x in obj["tree"] if x["type"]=="blob" and any(k in x["path"].lower() for k in ["parser","decoder","umrr","transport","timestamp","latency","protocol"])))
if __name__=="__main__": main()
