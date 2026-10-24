"""Verify delivery bytes without evaluating policies or using the network."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "STAGE10B_DELIVERY_MANIFEST.json").read_bytes())
for name, expected in manifest["files"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Delivery file missing or changed: " + name)
print("Delivery verified:", len(manifest["files"]), "files; no policy run")
