"""Download pinned official CATCH files separately, then verify exact hashes."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path, PurePosixPath


def verify_upstream(root, download=False):
    root = Path(root)
    inventory = json.loads((root / "VEHICLE_STAGE13_UPSTREAM.json").read_bytes())
    for item in inventory["files"]:
        p = PurePosixPath(item["local_path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe_upstream_path")
        dest = root / p
        if not dest.exists() and download:
            if not item["url"].startswith(
                "https://raw.githubusercontent.com/decisionintelligence/CATCH/"
                + inventory["catch_commit"]
                + "/"
            ):
                raise ValueError("unexpected_upstream_url")
            with urllib.request.urlopen(item["url"], timeout=60) as r:
                data = r.read(1024 * 1024)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("upstream_checksum_mismatch")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("upstream_checksum_mismatch")
    base = root / "third_party/vehicle_ai_stage13/catch"
    for n in [
        "ts_benchmark",
        "ts_benchmark/baselines",
        "ts_benchmark/baselines/catch",
        "ts_benchmark/baselines/catch/layers",
        "ts_benchmark/baselines/catch/models",
        "ts_benchmark/baselines/catch/utils",
    ]:
        p = base / n / "__init__.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("")
        if p.read_bytes() != b"":
            raise ValueError("unexpected_package_initializer")
    return dict(commit=inventory["catch_commit"], files=len(inventory["files"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    print(verify_upstream(a.root, a.download))
