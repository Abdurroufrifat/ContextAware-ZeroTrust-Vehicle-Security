"""Fresh DEVELOPMENT closed-loop pilot; never calls the frozen runner main/grid."""
from dev_common import *
from dev_policy import CorroboratedPolicy
from policy import ComparisonPolicy
import importlib.util,json,os,time,concurrent.futures
CFG=json.loads((R/"PILOT_PROTOCOL.json").read_text())
def factory(state,context,keys,method):
    return CorroboratedPolicy(state,context,keys) if method=="corroborated" else ComparisonPolicy(state,context,keys,method)
def trial(task):
    spec=importlib.util.spec_from_file_location("isolated_frozen_pilot_engine",B/"runner.py")
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    engine.B=R;engine.CONFIG=CFG;engine.ComparisonPolicy=factory
    return engine.run_trial(task)
def main():
    lock=json.loads((R/"development_lock.json").read_text());assert all(sha(P/n)==h for n,h in lock.items())
    digest=sha(R/"development_lock.json")
    jobs=[]
    for case in CFG["primary_cases"]+CFG["stress_cases"]:
        for method in CFG["methods"]:
            jobs.append({"trial_id":len(jobs),"group":"development","profile":"fresh_merge","case":case,"seed":2069,"method":method})
    assert len(jobs)==52
    output=R/"pilot_results.json"
    if output.exists():raise RuntimeError("Saved development pilot exists; do not rerun")
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trial,[(j,"corroborated-development-v1",digest) for j in jobs]):
            rows.append(result);print("DEVELOPMENT_TRIALS",len(rows),"/52",flush=True)
    assert all(sha(P/n)==h for n,h in lock.items())
    dump(output,{"pilot":True,"scope":"Fresh development profile and seed; not independent confirmation","trials":rows,"lock_digest":digest})
    print("DEVELOPMENT_PILOT_COMPLETE_52",flush=True)
if __name__=="__main__":main()
