"""Check overlay and inherited dependency hashes before running the experiment."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve()
    m = json.loads((root / "STAGE13_DELIVERY_MANIFEST.json").read_bytes())
    result = {}
    for section in ["files", "base_files"]:
        for name, h in m[section].items():
            p = PurePosixPath(name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in name
                or not (root / name).resolve().is_relative_to(root)
            ):
                raise ValueError("unsafe_delivery_path")
            path = root / name
            b = path.read_bytes()
            if path.suffix in {".py", ".json", ".txt"}:
                b = b.replace(b"\r\n", b"\n")
            if hashlib.sha256(b).hexdigest() != h:
                raise ValueError("delivery_file_changed:" + name)
        result[section] = len(m[section])
    return dict(result, status="PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(verify(p.parse_args().root))
