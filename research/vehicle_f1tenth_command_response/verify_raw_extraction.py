"""ROS bag container reader + independent standard-message struct decoding."""
from pathlib import Path
import csv,json,struct,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/"dependencies"))
from rosbags.rosbag1 import Reader
def main():
    counts={}
    for file in (P/"raw").glob("*.bag"):
        folder=P/"results"/file.stem
        with (folder/"commands.csv").open(newline="",encoding="utf-8") as f: commands=list(csv.DictReader(f))
        with (folder/"reference.csv").open(newline="",encoding="utf-8") as f: refs=list(csv.DictReader(f))
        i=j=0
        with Reader(file) as reader:
            conn=[c for c in reader.connections if c.topic in ["/cmd_vel","/vrpn_client_node/Car_2_Tracking/twist"]]
            for c,t,data in reader.messages(connections=conn):
                b=bytes(data)
                if c.topic=="/cmd_vel":
                    assert c.msgtype=="geometry_msgs/msg/Twist"
                    assert len(b)==48
                    values=struct.unpack("<6d",b)
                    row=commands[i]; i+=1
                    assert int(row["time_ns"])==t
                    assert [float(row[k]) for k in ["linear_x","linear_y","linear_z","angular_z"]]==[values[0],values[1],values[2],values[5]]
                else:
                    assert c.msgtype=="geometry_msgs/msg/TwistStamped"
                    seq,sec,nsec,n=struct.unpack_from("<4I",b,0)
                    assert nsec<1000000000 and len(b)==16+n+48
                    values=struct.unpack_from("<6d",b,16+n)
                    row=refs[j]; j+=1
                    assert int(row["time_ns"])==t
                    assert int(row["header_ns"])==sec*1000000000+nsec
                    assert [float(row[k]) for k in ["x","y","z"]]==list(values[:3])
        assert i==len(commands) and j==len(refs)
        counts[file.stem]={"commands":i,"reference":j}
    out={"passed":True,"method":"stdlib struct independent of production rosbags type deserializer; shared ROS bag container reader","counts":counts}
    assert len(counts)==3
    (P/"raw_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out))
if __name__=="__main__":main()
