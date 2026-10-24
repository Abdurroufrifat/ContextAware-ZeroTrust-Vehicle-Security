"""Resumable isolated lifecycle study. Frozen predecessors are read-only inputs."""
from lc_common import *
import json,os,argparse,itertools,concurrent.futures,importlib.util
def run_one(task):
    job,phase,digest=task
    spec=importlib.util.spec_from_file_location("lifecycle_simulation_engine",R/"harness_engine.py")
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    cfg=json.loads((R/("DEVELOPMENT_PROTOCOL.json" if phase=="development" else "PROTOCOL.json")).read_text())
    root=R/phase;root.mkdir(exist_ok=True)
    engine.B=root;engine.CONFIG=cfg
    return engine.run_trial((job,None,digest))
def main():
    a=argparse.ArgumentParser();a.add_argument("--phase",choices=("development","confirmation"),required=True);args=a.parse_args()
    phase=args.phase;cfg=json.loads((R/("DEVELOPMENT_PROTOCOL.json" if phase=="development" else "PROTOCOL.json")).read_text())
    lockfile=R/(phase+"_lock.json");lock=json.loads(lockfile.read_text());assert all(sha(P/n)==h for n,h in lock.items())
    digest=sha(lockfile);jobs=[]
    for profile,case,seed,method in itertools.product(cfg["profiles"],cfg["primary_cases"],cfg["seeds"],cfg["methods"]):
        jobs.append({"trial_id":len(jobs),"group":phase,"profile":profile,"case":case,"seed":seed,"method":method})
    assert len(jobs)==cfg["expected_trials"]
    root=R/phase;root.mkdir(exist_ok=True);outfile=root/"results.json"
    if outfile.exists():raise RuntimeError("Saved complete results exist; use verification, not a new experiment")
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["parallel_workers"]) as pool:
        for t in pool.map(run_one,[(j,phase,digest) for j in jobs]):
            rows.append(t)
            if len(rows)%5==0:print(phase,"COMPLETE",len(rows),"/",len(jobs),flush=True)
    assert all(sha(P/n)==h for n,h in lock.items())
    dump(outfile,{"phase":phase,"lock_digest":digest,"trials":rows,"study_id":cfg["study_id"],"scope":"Numerical longitudinal simulation with explicit bootstrap and execution premises"})
    print("STUDY_SAVED",phase,len(rows),flush=True)
if __name__=="__main__":main()
