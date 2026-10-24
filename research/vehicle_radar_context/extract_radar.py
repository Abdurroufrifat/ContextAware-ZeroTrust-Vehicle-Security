"""Decode the three allowlisted bags and match measurements exactly to processed records."""
import dataclasses,hashlib,json
from collections import Counter,defaultdict,deque
from pathlib import Path
from rosbags.highlevel import AnyReader
from radar_decoder import Decoder
ROOT=Path(__file__).resolve().parent
SCAN=ROOT.parent/"vehicle_scanario_reference"
FIELDS=("range","radial_speed","azimuth","elevation","rcs","power","noise")
def plain(x):
    if dataclasses.is_dataclass(x): return {f.name:plain(getattr(x,f.name)) for f in dataclasses.fields(x) if not f.name.startswith("__")}
    if hasattr(x,"tolist"): return x.tolist()
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x
def signature(sensor,targets):
    return hashlib.sha256(json.dumps([sensor,targets],separators=(",",":"),allow_nan=False).encode()).hexdigest()
def extract(name,bag):
    decoder=Decoder();frames=[];processed=defaultdict(deque);transforms=[];failures=Counter();processed_count=0
    with AnyReader([bag]) as reader:
        selected=[c for c in reader.connections if c.topic in {"/bus/umrr/eth_measurements/bus_to_host","/sensor/radar/umrr/detections","/tf_static"}]
        for c,t,raw in reader.messages(connections=selected):
            m=reader.deserialize(raw,c.msgtype)
            if c.topic=="/tf_static":
                transforms.extend(plain(m)["transforms"]);continue
            stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec
            if c.topic=="/sensor/radar/umrr/detections":
                sensor=int(m.header.frame_id.rsplit("/",1)[1])
                targets=[[getattr(d,f).value for f in FIELDS] for d in m.detections]
                processed[signature(sensor,targets)].append(dict(stamp_ns=stamp,receipt_ns=t))
                processed_count+=1
            else:
                try: frame=decoder.feed(m.payload,int(m.sender_ip[-1]),stamp,t)
                except ValueError as exc:
                    failures[str(exc)]+=1;continue
                if frame is not None: frames.append(frame)
    matched=0;offsets=Counter()
    for frame in frames:
        key=signature(frame["sensor"],frame["targets"])
        queue=processed[key]
        while queue and queue[0]["receipt_ns"]<frame["available_ns"]-2_000_000_000: queue.popleft()
        if queue:
            original=queue.popleft()
            frame["corrected_header_ns"]=original["stamp_ns"]
            frame["processed_receipt_ns"]=original["receipt_ns"]
            offsets[frame["raw_header_ns"]-original["stamp_ns"]]+=1
            matched+=1
        else: frame["corrected_header_ns"]=None
    folder=ROOT/"runtime"/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/"frames.jsonl").open("w") as out:
        for frame in frames: out.write(json.dumps(frame,allow_nan=False)+"\n")
    (folder/"transforms.json").write_text(json.dumps(transforms))
    summary=dict(recording=name,raw_frames=len(frames),processed_frames=processed_count,
        exact_value_matches=matched,unmatched_raw_frames=len(frames)-matched,
        raw_minus_processed_stamp_ns=dict(offsets),decoder_counts=dict(decoder.counts),decoder_errors=dict(failures))
    (folder/"extraction_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
def main():
    bags={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
        "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
        "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
    for name,bag in bags.items(): extract(name,bag)
if __name__=="__main__": main()
