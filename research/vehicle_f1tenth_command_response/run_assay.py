"""Read-only extraction and command/motion comparison of three public physical runs."""
from pathlib import Path
import csv, hashlib, json, math, sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/"dependencies"))
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from assay_core import pair_samples, longitudinal_commands
COMMAND="/cmd_vel"
REFERENCE="/vrpn_client_node/Car_2_Tracking/twist"

def quantile(values,p=.95):
    v=sorted(values)
    if not v: return None
    h=(len(v)-1)*p; lo=int(h); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(h-lo)

def metrics(rows):
    err=[r["error_mps"] for r in rows if r["error_mps"] is not None]
    return {"count":len(err),"mae_mps":sum(err)/len(err) if err else None,
            "p95_mps":quantile(err),"maximum_mps":max(err) if err else None,
            "over_0_5":sum(x>.5 for x in err),"over_2":sum(x>2 for x in err)}

def write_csv(path,rows,fields=None):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    protocol=json.loads((P/"PROTOCOL.json").read_text())
    expected=(P/"protocol_sha256.txt").read_text().strip()
    assert hashlib.sha256((P/"PROTOCOL.json").read_bytes()).hexdigest()==expected
    ts=get_typestore(Stores.ROS1_NOETIC)
    results=[]
    for entry in protocol["files"]:
        if not entry["key"].endswith(".bag"): continue
        raw=P/"raw"/entry["key"]
        assert raw.stat().st_size==entry["size"]
        assert hashlib.md5(raw.read_bytes()).hexdigest()==entry["checksum"].split(":")[1]
        commands=[]; refs=[]; topic_counts={}
        with Reader(raw) as reader:
            topics=[{"topic":c.topic,"type":c.msgtype,"count":c.msgcount} for c in reader.connections]
            connections=[c for c in reader.connections if c.topic in (COMMAND,REFERENCE)]
            assert {c.topic for c in connections}=={COMMAND,REFERENCE}
            for c,t,data in reader.messages(connections=connections):
                msg=ts.deserialize_ros1(data,c.msgtype)
                topic_counts[c.topic]=topic_counts.get(c.topic,0)+1
                if c.topic==COMMAND:
                    commands.append({"time_ns":int(t),"linear_x":float(msg.linear.x),
                                     "linear_y":float(msg.linear.y),"linear_z":float(msg.linear.z),
                                     "angular_z":float(msg.angular.z)})
                else:
                    stamp=msg.header.stamp
                    refs.append({"time_ns":int(t),"header_ns":int(stamp.sec)*1000000000+int(stamp.nanosec),
                                 "x":float(msg.twist.linear.x),"y":float(msg.twist.linear.y),
                                 "z":float(msg.twist.linear.z)})
        folder=P/"results"/raw.stem; folder.mkdir(parents=True,exist_ok=True)
        write_csv(folder/"commands.csv",commands)
        write_csv(folder/"reference.csv",refs)
        paired=pair_samples(longitudinal_commands(commands),
                            [(r["time_ns"],r["x"],r["y"],r["z"]) for r in refs],protocol["threshold_ns"])
        write_csv(folder/"paired.csv",paired)
        fresh=[r for r in paired if r["status"]=="fresh"]
        matched=[r for r in paired if r["status"]!="unmatched"]
        offsets=[r["header_ns"]-r["time_ns"] for r in refs]
        ages=[r["age_ns"]/1e6 for r in matched]
        summary={"run":raw.stem,"topics":topics,"command_count":len(commands),"reference_count":len(refs),
                 "status_counts":{s:sum(r["status"]==s for r in paired) for s in ["unmatched","stale","fresh"]},
                 "all_matched":metrics(matched),"fresh_receipt_age_le_100ms":metrics(fresh),
                 "duration_s":(refs[-1]["time_ns"]-refs[0]["time_ns"])/1e9,
                 "negative_commands":sum(r["linear_x"]<0 for r in commands),
                 "nonlongitudinal_linear_commands":sum(r["linear_y"]!=0 or r["linear_z"]!=0 for r in commands),
                 "command_changes":sum(a["linear_x"]!=b["linear_x"] for a,b in zip(commands,commands[1:])),
                 "command_age_ms":{"min":min(ages),"p95":quantile(ages),"max":max(ages)},
                 "header_minus_receipt_ms":{"min":min(offsets)/1e6,"max":max(offsets)/1e6,
                                           "median":quantile(offsets,.5)/1e6},
                 "zero_header_stamps":sum(r["header_ns"]==0 for r in refs),
                 "backward_header_steps":sum(b["header_ns"]<a["header_ns"] for a,b in zip(refs,refs[1:])),
                 "duplicate_reference_receipts":sum(b["time_ns"]==a["time_ns"] for a,b in zip(refs,refs[1:]))}
        (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        results.append(summary)
        print(raw.stem,summary["all_matched"],flush=True)
    assessment={"study":"public physical recorded desired-command response",
                "runs":results,"claim":"command and external motion measured on real F1TENTH hardware",
                "is_policy_closed_loop_physical_validation":False,
                "is_executed_acceleration_measurement":False,
                "is_calibrated_worst_case_reference":False,
                "is_full_scene_command_interface":False,
                "time_alignment":"causal bag receipt timestamps, no fitted lag",
                "reference":"VICON velocity magnitude in world frame; command magnitude in body frame; direction not validated",
                "source_license":"CC BY 4.0; Badakis, Galanis, Zhang; DOI 10.5281/zenodo.12536536"}
    (P/"assessment.json").write_text(json.dumps(assessment,indent=2),encoding="utf-8")

if __name__=="__main__": main()
