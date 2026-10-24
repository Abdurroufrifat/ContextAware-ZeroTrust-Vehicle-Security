import base64,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
metadata=json.loads((ROOT/"velodyne_driver_metadata.json").read_text())
selected=[x for x in metadata["tree"] if x["type"]=="blob" and (
    "LICENSE" in x["path"].upper() or x["path"]=="README.md" or
    ("velodyne_pointcloud" in x["path"] and any(k in x["path"] for k in ("rawdata.","datacontainerbase.","pointcloud_extended.","transform."))))]
provenance=[]
for x in selected:
    with urllib.request.urlopen(urllib.request.Request(x["url"],headers={"User-Agent":"COACT-development-audit"}),timeout=30) as f: blob=json.load(f)
    content=base64.b64decode(blob["content"]);p=ROOT/"vendor_velodyne"/x["path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    provenance.append(dict(path=x["path"],blob_sha=x["sha"],sha256=hashlib.sha256(content).hexdigest()))
    print("FILE",x["path"],len(content))
    for i,line in enumerate(content.decode(errors="replace").splitlines()):
        if any(k in line for k in ("scan_start_time","sub_segment","rotation_segment","firing_bin","time_diff","point.time","timestamp_correction")):
            print(i+1,line[:200])
(ROOT/"velodyne_source_provenance.json").write_text(json.dumps(dict(repository=metadata["repository"],tree_sha=metadata["tree_sha"],files=provenance),indent=2)+"\n")
