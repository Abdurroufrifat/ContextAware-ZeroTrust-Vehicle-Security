from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze(root: str | Path, manifest: dict[str, Any]) -> list[str]:
    project_root = Path(root).resolve()
    failures: list[str] = []
    if manifest.get("protocol_version") != "cps-truthframe-surface-utility-v8.0":
        failures.append("manifest: unsupported protocol_version")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return failures + ["manifest: files must be a non-empty list"]
    for item in files:
        if not isinstance(item, dict):
            failures.append("manifest: file entry is not an object")
            continue
        relative = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        candidate = (project_root / relative).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError:
            failures.append(f"{relative}: path escapes project root")
            continue
        if not candidate.is_file():
            failures.append(f"{relative}: missing")
            continue
        actual = file_sha256(candidate)
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, found {actual}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed if a frozen V8 file drifted")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    failures = verify_freeze(args.root, manifest)
    if failures:
        print("V8 freeze verification FAILED:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"V8 freeze verification passed for {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
