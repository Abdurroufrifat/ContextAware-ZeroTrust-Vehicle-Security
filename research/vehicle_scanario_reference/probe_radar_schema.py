"""Exploratory next-stage schema check only; separate from frozen baseline protocol."""
import json
from pathlib import Path
from rosbags.highlevel import AnyReader
from extract_streams import plain
ROOT=Path(__file__).resolve().parent
def run():
    bag=ROOT/"raw_data"/"development_07_turning.bag"
    with AnyReader([bag]) as reader:
        connections=[c for c in reader.connections if c.topic=="/sensor/radar/umrr/detections"]
        examples=[]
        for c,receipt,raw in reader.messages(connections=connections):
            message=plain(reader.deserialize(raw,c.msgtype))
            shape={k:(dict(kind="list",length=len(v),example=v[:2]) if isinstance(v,list) else v) for k,v in message.items()}
            examples.append(dict(receipt_ns=receipt,message=shape))
            if len(examples)==2: break
        metadata=[dict(topic=c.topic,msgtype=c.msgtype,count=c.msgcount,definition=str(c.msgdef)) for c in connections]
    report=dict(exploratory_schema_only=True,baseline_protocol_changed=False,
        no_radar_velocity_estimated=True,connections=metadata,examples=examples)
    (ROOT/"radar_schema_probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(examples=examples),indent=2))
if __name__=="__main__": run()
