"""Receipt-time availability and motion-innovation proxies; not sensor calibration."""
from pathlib import Path
import bisect,csv,hashlib,json,math
B=Path(__file__).resolve().parent;P=B.parent.parent
SOURCE=P/"research"/"vehicle_f1tenth_command_response"
def quantile(values,p):
    if not values or not 0<p<=1:raise ValueError("quantile")
    a=sorted(values);return a[max(0,math.ceil(len(a)*p)-1)]
def extract_series(commands,reference):
    if not reference or any(b<a for a,b in zip(commands,commands[1:])):raise ValueError("nonmonotone command receipt clock")
    times=[int(r["time_ns"]) for r in reference]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError("reference receipt clock must increase")
    ages=[];speed=[];grid=[]
    for stamp in range(times[0],times[-1]+1,100_000_000):
        i=bisect.bisect_right(times,stamp)-1;r=reference[i]
        speed.append(math.sqrt(sum(float(r[k])**2 for k in ("x","y","z"))))
        j=bisect.bisect_right(commands,stamp)-1
        ages.append(None if j<0 else (stamp-commands[j])//1000);grid.append(stamp)
    innovations=[None if i<2 else speed[i]-2*speed[i-1]+speed[i-2] for i in range(len(speed))]
    return {"grid_start_receipt_ns":times[0],"grid_us":100000,"command_rows":len(commands),"reference_rows":len(reference),
            "duplicate_command_stamps":sum(a==b for a,b in zip(commands,commands[1:])),
            "age_us":ages,"innovation_mps":innovations,"reference_speed_mps":speed}
def main():
    groups={}
    for label,prefix in [("calibration_easy","ex-easy-perception-r2"),("heldout_medium","ex-medium-perception-r1"),("heldout_hard","ex-hard-r2")]:
        matches=list((SOURCE/"results").glob(prefix+"*"));assert len(matches)==1
        folder=matches[0];cp=folder/"commands.csv";rp=folder/"reference.csv"
        with cp.open(newline="",encoding="utf-8") as f:commands=[int(r["time_ns"]) for r in csv.DictReader(f)]
        with rp.open(newline="",encoding="utf-8") as f:reference=list(csv.DictReader(f))
        groups[label]=extract_series(commands,reference)
        groups[label]["source_files"]={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,rp)}
        groups[label]["run"]=folder.name
    easy=groups["calibration_easy"];ages=[x for x in easy["age_us"] if x is not None]
    innovations=[abs(x) for x in easy["innovation_mps"] if x is not None]
    calibration={"source_doi":"10.5281/zenodo.12536536","calibration_run":easy["run"],"heldout_runs":[groups[k]["run"] for k in ("heldout_medium","heldout_hard")],
       "age_us_nearest_rank":{"p50":quantile(ages,.5),"p95":quantile(ages,.95),"p99":quantile(ages,.99),"maximum":max(ages)},
       "absolute_motion_innovation_mps_nearest_rank":{"p50":quantile(innovations,.5),"p95":quantile(innovations,.95),"p99":quantile(innovations,.99),"maximum":max(innovations)},
       "interpretation":"Receipt-time command age is an observation-availability proxy, not network delay/clock calibration. Velocity second differences combine real motion and reference noise; transferred stress perturbations are not sensor-error calibration or hard bounds.",
       "source_manifest_sha256":hashlib.sha256((SOURCE/"package_manifest.json").read_bytes()).hexdigest()}
    for name,obj in [("public_sequences.json",groups),("calibration.json",calibration)]:
        (B/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("PUBLIC_PROFILE_FROZEN",json.dumps(calibration),flush=True)
if __name__=="__main__":main()
