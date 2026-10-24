"""Acquire only the smallest public development recording; verify publisher SHA256."""
import hashlib, json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "c131d445f3ea51c9db6ed3e63c64eac762f01d6e"
NAME = "multi-vehicle__07_turning__2024-12-05-12-43-24.bag.tar.xz"
SIZE = 1645420460
SHA256 = "f820d7a36e56f7c1b62270e856b0aacb5189d75c5de75678a0f4a4efb3854cd0"
BASE = f"https://huggingface.co/datasets/UniBw-AD/7V-Scanario/resolve/{REVISION}/"
RAW = ROOT / "raw_data"
RAW.mkdir(exist_ok=True)
def fetch_small(name):
    data = urllib.request.urlopen(BASE + name, timeout=60).read()
    (ROOT / ("SOURCE_" + name)).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
license_sha = fetch_small("LICENSE")
readme_sha = fetch_small("README.md")
dest = RAW / NAME
partial = RAW / (NAME + ".part")
if not dest.exists():
    start = partial.stat().st_size if partial.exists() else 0
    if start > SIZE:
        raise RuntimeError("Partial file exceeds expected size")
    if start < SIZE:
        request = urllib.request.Request(BASE + "recordings/" + NAME)
        if start:
            request.add_header("Range", f"bytes={start}-")
        with urllib.request.urlopen(request, timeout=90) as response:
            if start and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                raise RuntimeError("Resume range not honored; existing partial preserved")
            expected_length = SIZE - start
            if int(response.headers.get("Content-Length", expected_length)) != expected_length:
                raise RuntimeError("Unexpected response size")
            print(f"Downloading development recording: {start}/{SIZE} bytes", flush=True)
            with partial.open("ab" if start else "wb") as out:
                last = time.monotonic()
                total = start
                while chunk := response.read(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > SIZE:
                        raise RuntimeError("Response exceeds expected size")
                    out.write(chunk)
                    if time.monotonic() - last >= 15:
                        print(f"Downloaded {total}/{SIZE} bytes ({100*total/SIZE:.1f}%)", flush=True)
                        last = time.monotonic()
    if partial.stat().st_size != SIZE:
        raise RuntimeError("Incomplete download; rerun to resume")
    target = partial
else:
    target = dest
print("Verifying archive SHA256", flush=True)
digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
if target.stat().st_size != SIZE or digest != SHA256:
    raise RuntimeError("Archive verification failed; payload not opened")
if target == partial:
    partial.rename(dest)
manifest = dict(dataset="7V-Scanario", publisher="UniBwTAS / UniBw-AD",
    revision=REVISION, recording=NAME, archive_bytes=SIZE, archive_sha256=digest,
    license_sha256=license_sha, readme_sha256=readme_sha,
    purpose="development parser/provenance pilot only", final_evaluation=False,
    acquisition_authentication=False, physical_error_bound_established=False)
(ROOT / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print("ACQUISITION VERIFIED: development-only archive, no final evaluation", flush=True)
