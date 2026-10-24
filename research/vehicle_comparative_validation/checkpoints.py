"""Atomic complete manifests; incomplete attempts remain available for diagnosis."""
from pathlib import Path
import hashlib,json,os
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,obj):
    temp=path.with_name(path.name+".tmp");temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8");os.replace(temp,path)
def write_checkpoint(root,summary,lock_digest,files):
    root=Path(root);root.mkdir(exist_ok=True,parents=True)
    path=root/"summary.json";atomic(path,summary)
    bindings={}
    for p in [*map(Path,files),path]:
        if not p.resolve().is_relative_to(root.resolve()):raise ValueError("checkpoint scope")
        bindings[p.relative_to(root).as_posix()]=sha(p)
    atomic(root/"complete.json",{"trial_id":summary["trial_id"],"lock_digest":lock_digest,"files":bindings})
def completed_trial(root,trial_id,lock_digest):
    root=Path(root);marker=root/"complete.json"
    if not marker.exists():return None
    m=json.loads(marker.read_text())
    if m["trial_id"]!=trial_id or m["lock_digest"]!=lock_digest:raise ValueError("checkpoint experiment identity")
    for n,h in m["files"].items():
        p=root/n
        if not p.resolve().is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=h:raise ValueError("checkpoint content changed")
    summary=json.loads((root/"summary.json").read_text())
    if summary["trial_id"]!=trial_id:raise ValueError("summary identity")
    return summary
