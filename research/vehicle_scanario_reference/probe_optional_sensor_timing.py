"""Exploratory timing feasibility; no policy inputs or baseline outcomes are changed."""
import json, statistics
from collections import defaultdict
from pathlib import Path
from rosbags.highlevel import AnyReader
ROOT=Path(__file__).resolve().parent
def run():
    timing=defaultdict(list)
    with AnyReader([ROOT/"raw_data"/"development_07_turning.bag"]) as reader:
        selected=[c for c in reader.connections if c.topic in {
            "/sensor/radar/umrr/detections","/sensor/lidar/vls128_roof/velodyne_points"}]
        for c,receipt,raw in reader.messages(connections=selected):
            msg=reader.deserialize(raw,c.msgtype)
            sensor=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            timing[c.topic+" | "+msg.header.frame_id].append((receipt-sensor)/1e6)
    report=dict(exploratory_only=True,baseline_protocol_changed=False,topics={
        k:dict(count=len(v),min_ms=min(v),median_ms=statistics.median(v),max_ms=max(v),
               receipt_delay_above_100ms=sum(x>100 for x in v),negative_receipt_delay=sum(x<0 for x in v))
        for k,v in timing.items()})
    (ROOT/"optional_sensor_timing_probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": run()
