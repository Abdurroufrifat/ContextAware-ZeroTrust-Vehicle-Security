"""Verify delivered additive files plus frozen synthetic reference fingerprints."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'STAGE12_DELIVERY_MANIFEST.json').read_bytes())
    for name,record in manifest['files'].items():
        p=root/name
        assert p.stat().st_size==record['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],name
    reference=root/'data/reference/vehicle_witness_stage12/stage12_linux_review.zip'
    with zipfile.ZipFile(reference) as z:
        assert len(z.namelist())==len(set(z.namelist()))==6
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in z.namelist())
        assert z.testzip() is None
        f=json.loads(z.read('frozen_protocol.json'))
        for name,h in f['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,name
        hashes=json.loads(z.read('file_hashes.json'))
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    print(f"PASS: {len(manifest['files'])} delivered files; {len(f['source_sha256'])} frozen source/input files; reference ZIP integrity.")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    verify(p.parse_args().root)
