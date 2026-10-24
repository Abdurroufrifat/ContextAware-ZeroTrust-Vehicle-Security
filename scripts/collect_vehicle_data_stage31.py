"""Frozen development-only acquisition; default offline validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vehicle_native_archive import RangeArchive,parse_central,extract_entry,validate_npy
from coact_cps.vehicle.data31 import CHANNELS,select_trips,payload_end,channel_report

BASE=Path('data/reference/vehicle_data_stage24')
REF=Path('data/reference/vehicle_data_stage31')
RAW=Path('data/raw/vehicle_data_stage31')
OUT=Path('artifacts/vehicle_data_stage31')
BASE_PIN='704acd41fd0168215b1e3ab4339b754ea16eb870566aa098234a7be8d2fe9fee'


def sha(data):return hashlib.sha256(data).hexdigest()


def build_protocol(root):
    root=Path(root);base=root/BASE;raw=(base/'protocol.json').read_bytes()
    if sha(raw)!=BASE_PIN:raise ValueError('fixed Stage24 protocol mismatch')
    p=json.loads(raw);central=(base/'chunk2_central.bin').read_bytes()
    prior=(base/'prior_demo_routes.json').read_bytes()
    if sha(central)!=p['central_sha256'] or sha(prior)!=p['prior_inventory_sha256']:
        raise ValueError('fixed selection metadata mismatch')
    records=parse_central(central,p['central_count'])
    excluded=sorted(set(json.loads(prior)['routes'])|{t['route_id'] for t in p['trips']})
    trips=select_trips(records,excluded);indexed={r['name']:r for r in records};entries=[]
    for trip in trips[:6]:
        for segment in trip['segments']:
            for channel in CHANNELS:
                for kind in ('t','value'):
                    name=f"Chunk_2/{trip['route_id']}/{segment}/processed_log/{channel}/{kind}"
                    path=RAW/trip['id']/f'segment_{segment}'/channel/f'{kind}.npy'
                    entries.append(dict(name=name,local_path=path.as_posix(),archive_entry=indexed[name]))
    code=('src/coact_cps/vehicle/data31.py','scripts/collect_vehicle_data_stage31.py',
          'src/coact_cps/vehicle/data24.py','scripts/vehicle_native_archive.py')
    return dict(schema='vehicle-development-expansion-stage31-v1',stage24_protocol_sha256=BASE_PIN,
        archive_url=p['archive_url'],archive_size=p['archive_size'],central_count=p['central_count'],
        central_sha256=p['central_sha256'],excluded_routes=excluded,trips=trips,development_entries=entries,
        archive_lfs_sha256_recorded_not_fully_verified=p['archive_lfs_sha256_recorded_not_fully_verified'],
        code_sha256={name:sha((root/name).read_bytes()) for name in code},
        design_sha256=sha((root/'STAGE31_FROZEN_DESIGN.txt').read_bytes()),
        quality_reporting_addendum_sha256=sha((root/'STAGE31_QUALITY_ADDENDUM.txt').read_bytes()),
        existing_reserved_trips=[t for t in p['trips'] if t['role']=='reserved_validation'],
        original_stage16_final_routes=p['existing_stage16_final_routes'],
        source_documentation=p['source_documentation'],normal_data_qualification='Presumed benign, not verified attack-free.',
        prior_dates_are_excluded=True,new_validation_dates_are_later_than_development=True)


def protocol(root):
    raw=(root/REF/'protocol.json').read_bytes()
    if sha(raw)!=(root/REF/'protocol.sha256').read_text().strip():raise ValueError('Stage31 protocol hash mismatch')
    p=json.loads(raw)
    if p!=build_protocol(root) or len(p['development_entries'])!=96:
        raise ValueError('fixed metadata-only selection or code mismatch')
    return p


def inspect(root,p):
    quality=[];hashes={};eligible=[];quarantined=[]
    for trip in p['existing_reserved_trips']:
        if (root/'data/raw/vehicle_data_stage24'/trip['id']).exists():raise ValueError('old reserved payload present')
    for trip in p['trips']:
        if trip['role']=='reserved_validation':
            if (root/RAW/trip['id']).exists():raise ValueError('new reserved payload present')
            continue
        for segment in trip['segments']:
            channels={}
            for channel,columns in CHANNELS.items():
                arrays=[]
                for kind in ('t','value'):
                    relative=RAW/trip['id']/f'segment_{segment}'/channel/f'{kind}.npy';raw=(root/relative).read_bytes()
                    record=next(e['archive_entry'] for e in p['development_entries'] if e['local_path']==relative.as_posix())
                    if len(raw)!=record['size'] or zlib.crc32(raw)&0xffffffff!=record['crc32']:
                        raise ValueError('native array CRC/size mismatch')
                    arrays.append(validate_npy(raw,None if kind=='t' else columns));hashes[relative.as_posix()]=sha(raw)
                channels[channel]=channel_report(*arrays,columns,channel=='CAN/wheel_speed')
            overlap=(min(c['last_timestamp_seconds'] for c in channels.values())-max(c['first_timestamp_seconds'] for c in channels.values())
                if all('last_timestamp_seconds' in c for c in channels.values()) else None)
            failed=[name for name,c in channels.items() if c['status']!='PASS']
            passed=not failed and overlap is not None and overlap>20
            item=dict(trip=trip['id'],segment=segment)
            (eligible if passed else quarantined).append(item)
            quality.append(dict(**item,status='ELIGIBLE' if passed else 'QUARANTINED',failed_channels=failed,
                channels=channels,common_overlap_seconds=overlap))
    return dict(status='DATA_AUDITED_WITH_QUARANTINE' if quarantined else 'DATA_VERIFIED_NOT_SCORED',
        protocol_sha256=sha((root/REF/'protocol.json').read_bytes()),
        development_trips=6,development_segments=12,raw_arrays=len(hashes),raw_file_sha256=hashes,quality=quality,
        eligible_segments=eligible,quarantined_segments=quarantined,
        fully_eligible_trips=sum(all(q['status']=='ELIGIBLE' for q in quality if q['trip']==trip['id']) for trip in p['trips'][:6]),
        original_reserved_payloads_accessed=False,new_reserved_payloads_accessed=False,
        stage16_final_routes_accessed=False,model_training_or_inference=False,policy_changed=False)


def run(root,download=False):
    root=Path(root).resolve();p=protocol(root)
    if download:
        archive=RangeArchive(p['archive_url'],p['archive_size'],root/RAW/'range_cache')
        def fetch(row):
            dest=root/row['local_path'];record=row['archive_entry']
            if dest.is_file():
                raw=dest.read_bytes()
                if len(raw)==record['size'] and zlib.crc32(raw)&0xffffffff==record['crc32']:return
                raise ValueError('modified existing array; refusing overwrite')
            start=record['header_offset'];header=archive.fetch(start,start+29)
            end=payload_end(header,start,record['compressed_size'])
            rest=archive.fetch(start+30,end);raw=extract_entry(header+rest,record)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for begin in range(0,len(p['development_entries']),16):
                list(pool.map(fetch,p['development_entries'][begin:begin+16]))
    result=inspect(root,p);saved=root/OUT/'summary.json'
    if saved.exists():
        if json.loads(saved.read_text())!=result:raise ValueError('saved acquisition summary mismatch')
    elif download:
        saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(json.dumps(result,indent=2)+'\n')
    else:raise ValueError('missing frozen acquisition summary')
    print(json.dumps({k:v for k,v in result.items() if k not in ('quality','raw_file_sha256')},indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args();run(args.root,args.download)
