"""Inspect raw radar packet schema and timing; no velocity/reference outcome tuning."""
import dataclasses,json,statistics
from collections import defaultdict
from pathlib import Path
from rosbags.highlevel import AnyReader
ROOT=Path(__file__).resolve().parent
BAG=ROOT.parent/"vehicle_scanario_reference"/"raw_data"/"development_07_turning.bag"
def plain(x):
    if dataclasses.is_dataclass(x): return {f.name:plain(getattr(x,f.name)) for f in dataclasses.fields(x) if not f.name.startswith("__")}
    if hasattr(x,"tolist"): return x.tolist()
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x
def main():
    samples=[];ages=[];defs=[]
    with AnyReader([BAG]) as reader:
        selected=[c for c in reader.connections if c.topic=="/bus/umrr/eth_measurements/bus_to_host"]
        defs=[dict(topic=c.topic,msgtype=c.msgtype,count=c.msgcount,definition=str(c.msgdef)) for c in selected]
        for c,t,raw in reader.messages(connections=selected):
            m=reader.deserialize(raw,c.msgtype)
            ts=m.header.stamp.sec*10**9+m.header.stamp.nanosec
            ages.append((t-ts)/1e6)
            if len(samples)<3:
                value=plain(m)
                for key,v in value.items():
                    if isinstance(v,list): value[key]=dict(length=len(v),first_64=v[:64])
                samples.append(dict(receipt_ns=t,message=value))
    result=dict(definitions=defs,examples=samples,raw_receipt_minus_header_ms=dict(count=len(ages),
        min=min(ages),median=statistics.median(ages),max=max(ages),over_100ms=sum(x>100 for x in ages)))
    (ROOT/"raw_packet_probe.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
