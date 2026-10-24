"""Acquire frozen development entries only; verification never uses the network."""

import argparse
import hashlib
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vehicle_native_archive import RangeArchive, parse_central, validate_npy  # noqa: E402

from coact_cps.vehicle.data24 import CHANNELS, check_channel, select_trips  # noqa: E402

REFERENCE = Path('data/reference/vehicle_data_stage24')
RAW = Path('data/raw/vehicle_data_stage24')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_protocol(root):
    ref = root / REFERENCE
    raw = (ref / 'protocol.json').read_bytes()
    if sha(raw) != (ref / 'protocol.sha256').read_text().strip():
        raise ValueError('frozen protocol hash mismatch')
    p = json.loads(raw)
    central = (ref / 'chunk2_central.bin').read_bytes()
    if sha(central) != p['central_sha256']:
        raise ValueError('central directory hash mismatch')
    records = parse_central(central, p['central_count'])
    prior_raw = (ref / 'prior_demo_routes.json').read_bytes()
    if sha(prior_raw) != p['prior_inventory_sha256']:
        raise ValueError('prior exclusion inventory hash mismatch')
    if select_trips(records, json.loads(prior_raw)['routes']) != p['trips']:
        raise ValueError('metadata-only trip selection mismatch')
    indexed = {r['name']: r for r in records}
    if any(indexed.get(r['name']) != r['archive_entry'] for r in p['development_entries']):
        raise ValueError('selected original entry metadata mismatch')
    expected = {}
    for trip in p['trips']:
        if trip['role'] == 'development':
            for segment in trip['segments']:
                for channel in CHANNELS:
                    for kind in ('t', 'value'):
                        name = f"Chunk_2/{trip['route_id']}/{segment}/processed_log/{channel}/{kind}"
                        expected[name] = (RAW / trip['id'] / f'segment_{segment}' / channel /
                                          f'{kind}.npy').as_posix()
    if (len(p['development_entries']) != 48 or
            {r['name']: r['local_path'] for r in p['development_entries']} != expected):
        raise ValueError('only exact48 development entries may be accessed')
    return p


def inspect(root, p):
    rows, file_hashes = [], {}
    for trip in p['trips']:
        if trip['role'] != 'development':
            if (root / RAW / trip['id']).exists():
                raise ValueError('reserved validation payload must remain absent')
            continue
        for segment in trip['segments']:
            channels = {}
            for channel, columns in CHANNELS.items():
                arrays = []
                for kind in ('t', 'value'):
                    relative = RAW / trip['id'] / f'segment_{segment}' / channel / f'{kind}.npy'
                    raw = (root / relative).read_bytes()
                    name = f"Chunk_2/{trip['route_id']}/{segment}/processed_log/{channel}/{kind}"
                    record = next(r['archive_entry'] for r in p['development_entries'] if r['name'] == name)
                    if len(raw) != record['size'] or zlib.crc32(raw) & 0xFFFFFFFF != record['crc32']:
                        raise ValueError('native entry CRC/size mismatch')
                    arrays.append(validate_npy(raw, None if kind == 't' else columns))
                    file_hashes[relative.as_posix()] = sha(raw)
                channels[channel] = check_channel(*arrays, columns, channel == 'CAN/wheel_speed')
            overlap = min(c['last_timestamp_seconds'] for c in channels.values()) - max(
                c['first_timestamp_seconds'] for c in channels.values())
            if overlap <= 20:
                raise ValueError('insufficient common native sensor overlap')
            rows.append(dict(trip=trip['id'], segment=segment, channels=channels,
                             common_overlap_seconds=overlap))
    return dict(status='DATA_VERIFIED_NOT_SCORED', protocol_sha256=sha(
        (root / REFERENCE / 'protocol.json').read_bytes()), raw_file_sha256=file_hashes,
        development_trips=3, segments=6, raw_arrays=len(file_hashes),
        reserved_validation_payloads_accessed=False, stage16_final_routes_accessed=False,
        model_training_or_inference=False, policy_changed=False, quality=rows)


def run(root, download=False):
    root = Path(root).resolve()
    p = load_protocol(root)
    if download:
        archive = RangeArchive(p['archive_url'], p['archive_size'], root / RAW / 'range_cache')

        def fetch(row):
            dest = root / row['local_path']
            record = row['archive_entry']
            if dest.is_file():
                raw = dest.read_bytes()
                if len(raw) == record['size'] and zlib.crc32(raw) & 0xFFFFFFFF == record['crc32']:
                    return
                raise ValueError('existing native array modified; refusing overwrite')
            raw = archive.entry(record)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, p['development_entries']))
    result = inspect(root, p)
    print(json.dumps({k: v for k, v in result.items() if k not in ('quality', 'raw_file_sha256')}, indent=2))
    if download:
        out = root / 'artifacts/vehicle_data_stage24'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    else:
        saved = json.loads((root / 'artifacts/vehicle_data_stage24/summary.json').read_text())
        if result != saved:
            raise ValueError('saved acquisition summary mismatch')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    run(args.root, args.download)
