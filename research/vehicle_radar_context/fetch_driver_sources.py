"""Retrieve exact public driver blobs and their license, with provenance."""
import base64,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    tree=json.loads((ROOT/"driver_tree.json").read_text())
    wanted=["LICENSE","driver/smartmicro/smartmicro_driver/src/ethernet/core/sms.cpp",
        "driver/smartmicro/smartmicro_driver/src/ethernet/core/sms.h",
        "driver/smartmicro/smartmicro_driver/src/ethernet/ros1/node.cpp",
        "driver/smartmicro/smartmicro_driver/src/common/detections.h",
        "launch/radar_launch/launch/ros1/smartmicro_ethernet.launch"]
    blobs=[]
    for name in wanted:
        item=next(x for x in tree["tree"] if x["path"]==name)
        with urllib.request.urlopen(item["url"],timeout=60) as inp: obj=json.load(inp)
        content=base64.b64decode(obj["content"])
        target=ROOT/"vendor"/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        blobs.append(dict(path=name,git_blob_sha=item["sha"]))
        print(name,flush=True)
        lines=content.decode(errors="replace").splitlines()
        indexes=set()
        for i,line in enumerate(lines):
            if any(k in line.lower() for k in ["latency","offset","timestamp","time_stamp","100000","106000"]):
                indexes.update(range(max(0,i-2),min(len(lines),i+3)))
        for i in sorted(indexes): print(f"{i+1}: {lines[i]}",flush=True)
    (ROOT/"driver_source_provenance.json").write_text(json.dumps(dict(tree_sha=tree["sha"],files=blobs),indent=2)+"\n")
if __name__=="__main__": main()
