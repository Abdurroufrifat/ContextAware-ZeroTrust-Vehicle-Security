from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
m=json.loads((P/"package_manifest.json").read_text())
for n,d in m["files"].items():
 f=P/n
 assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==d,n
assert hashlib.sha256((P/"PROTOCOL.json").read_bytes()).hexdigest()==(P/"protocol_sha256.txt").read_text().strip()
print("PACKAGE_INTEGRITY_PASS",len(m["files"]))
