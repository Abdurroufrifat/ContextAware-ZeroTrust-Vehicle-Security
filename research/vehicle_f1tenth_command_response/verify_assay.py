"""Independent metrics and linear causal scan; does not import production pairing."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent
def read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def main():
    report=json.loads((P/"assessment.json").read_text())
    checked=0
    for summary in report["runs"]:
        folder=P/"results"/summary["run"]
        commands=read(folder/"commands.csv"); refs=read(folder/"reference.csv"); rows=read(folder/"paired.csv")
        assert len(refs)==len(rows)==summary["reference_count"]
        assert len(commands)==summary["command_count"]
        j=-1; errors=[]; fresh=[]; status={"unmatched":0,"stale":0,"fresh":0}
        for i,(ref,row) in enumerate(zip(refs,rows)):
            t=int(ref["time_ns"])
            while j+1<len(commands) and int(commands[j+1]["time_ns"])<=t: j+=1
            assert int(row["reference_index"])==i
            norm=float(np.linalg.norm([float(ref[k]) for k in ("x","y","z")]))
            assert abs(float(row["reference_mps"])-norm)<1e-10
            assert int(row["reference_time_ns"])==t
            assert int(row["command_index"])==j
            if j<0:
                expected_status="unmatched"
                assert row["error_mps"]=="" and row["age_ns"]==""
            else:
                age=t-int(commands[j]["time_ns"])
                assert age>=0 and int(row["age_ns"])==age
                err=abs(abs(float(commands[j]["linear_x"]))-norm)
                assert abs(float(row["error_mps"])-err)<1e-10
                assert int(row["command_time_ns"])==int(commands[j]["time_ns"])
                assert float(row["command_signed_mps"])==float(commands[j]["linear_x"])
                assert float(row["command_mps"])==abs(float(commands[j]["linear_x"]))
                expected_status="fresh" if age<=100000000 else "stale"
                errors.append(err)
                if expected_status=="fresh": fresh.append(err)
            assert row["status"]==expected_status
            status[expected_status]+=1; checked+=1
        assert status==summary["status_counts"]
        for key,values in [("all_matched",errors),("fresh_receipt_age_le_100ms",fresh)]:
            m=summary[key]; a=np.array(values)
            assert len(a)==m["count"]
            if len(a):
                assert abs(float(a.mean())-m["mae_mps"])<1e-10
                assert abs(float(np.quantile(a,.95))-m["p95_mps"])<1e-10
                assert abs(float(a.max())-m["maximum_mps"])<1e-10
                assert int((a>.5).sum())==m["over_0_5"]
                assert int((a>2).sum())==m["over_2"]
    for item in json.loads((P/"download_manifest.json").read_text()):
        raw=P/"raw"/item["file"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==item["sha256"]
    out={"passed":True,"runs":len(report["runs"]),"reference_rows_checked":checked,
         "scope":"independent alignment/metric recomputation on extracted CSV; raw extraction separately checks ROS message structure and publisher MD5"}
    (P/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out))
if __name__=="__main__":main()
