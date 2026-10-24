"""Pinned allowlisted development data acquisition and existing-reader reuse."""
import hashlib, json, shutil, tarfile, time, urllib.request
from pathlib import Path
import extract_streams, run_pilot
ROOT=Path(__file__).resolve().parent
def digest(path):
    with path.open("rb") as inp:
        return hashlib.file_digest(inp,"sha256").hexdigest()
def download(item, protocol):
    raw=ROOT/"raw_data"
    dest=raw/item["name"]
    partial=raw/(item["name"]+".part")
    if not dest.exists():
        start=partial.stat().st_size if partial.exists() else 0
        if start>item["bytes"]:
            raise RuntimeError("Unexpected partial length")
        if start<item["bytes"]:
            url=f'https://huggingface.co/datasets/UniBw-AD/7V-Scanario/resolve/{protocol["dataset_revision"]}/recordings/{item["name"]}'
            req=urllib.request.Request(url)
            if start: req.add_header("Range",f"bytes={start}-")
            with urllib.request.urlopen(req,timeout=90) as response:
                if start and (response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {start}-")):
                    raise RuntimeError("Server did not honor resume; partial preserved")
                count=start
                last=time.monotonic()
                with partial.open("ab" if start else "wb") as out:
                    while chunk:=response.read(4*1024*1024):
                        count+=len(chunk)
                        if count>item["bytes"]: raise RuntimeError("Excess response")
                        out.write(chunk)
                        if time.monotonic()-last>20:
                            print(f'{item["id"]}: {count/item["bytes"]:.1%} downloaded',flush=True)
                            last=time.monotonic()
        target=partial
    else:
        target=dest
    print(f'{item["id"]}: checking SHA256',flush=True)
    if target.stat().st_size!=item["bytes"] or digest(target)!=item["sha256"]:
        raise RuntimeError("Publisher archive verification failed; payload not opened")
    if target==partial: partial.rename(dest)
    return dest
def unpack(archive, folder):
    bag=folder/"development.bag"
    if bag.exists(): return bag
    temp=folder/"development.bag.part"
    found=0
    with tarfile.open(archive,"r|xz") as inp:
        for member in inp:
            if not member.isfile() or not member.name.endswith(".bag"): continue
            if found or not 0<member.size<30_000_000_000:
                raise RuntimeError("Unexpected member count or size")
            with inp.extractfile(member) as data,temp.open("wb") as out:
                shutil.copyfileobj(data,out,4*1024*1024)
            if temp.stat().st_size!=member.size: raise RuntimeError("Incomplete bag")
            found+=1
        if found!=1: raise RuntimeError("Expected one bag")
    temp.rename(bag)
    return bag
def main():
    protocol_path=ROOT/"broader_development_protocol_v1.json"
    protocol_bytes=protocol_path.read_bytes()
    protocol=json.loads(protocol_bytes)
    protocol_sha=hashlib.sha256(protocol_bytes).hexdigest()
    allowed={"04_crossTraffic","14_countryRoad"}
    if {r["id"] for r in protocol["recordings"]}!=allowed or protocol["role"]!="development_only":
        raise RuntimeError("Allowlist mismatch")
    freeze=ROOT/"broader_protocol_freeze.json"
    if freeze.exists() and json.loads(freeze.read_text())["protocol_sha256"]!=protocol_sha:
        raise RuntimeError("Frozen protocol changed")
    freeze.write_text(json.dumps(dict(protocol_sha256=protocol_sha,
        frozen_before_new_payload_access=True),indent=2)+"\n")
    for item in protocol["recordings"]:
        folder=ROOT/"raw_data"/item["id"]
        folder.mkdir(exist_ok=True)
        archive=download(item,protocol)
        print(f'{item["id"]}: unpacking verified archive',flush=True)
        bag=unpack(archive,folder)
        extract_streams.ROOT=folder
        extract_streams.BAG=bag
        extract_streams.run()
        (folder/"acquisition_manifest.json").write_text(json.dumps(dict(item,
            protocol_sha256=protocol_sha, revision=protocol["dataset_revision"]),indent=2)+"\n")
        run_pilot.ROOT=folder
        run_pilot.run()
        print(f'{item["id"]}: numeric extraction and fixed pilot complete',flush=True)
    print("BROADER DEVELOPMENT EXTRACTION COMPLETE",flush=True)
if __name__=="__main__":
    main()
