"""Recompute strict packet-only decisions in fresh state; not an independent algorithm."""
from cv_common import B,P,KEYS,sha,dump
import os,json,gzip,time,argparse
from policy import ComparisonPolicy
from verification_contracts import verify_checkpoint

def engine_digest():
    return {n:sha(B/n) for n in ("replay_policy.py","verification_contracts.py")}

def replay_one(task):
    trial,digest,tag=task
    verify_checkpoint(B,trial,digest)
    root=B/"runtime"/tag/f"trial_{trial['trial_id']:04d}"
    marker=root/"replay_complete.json"
    if marker.exists():
        value=json.loads(marker.read_text())
        assert value["trace_sha256"]==sha(B/trial["trace"]) and value["lock_digest"]==digest
        assert value["evidence_engine"]==engine_digest() and value["trial_id"]==trial["trial_id"] and value["passed"] is True and value["queries"]==200
        return value
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(B/trial["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
    state=P/"research"/"project_release"/"runtime"/(tag+"-"+str(trial["trial_id"])+"-"+str(time.time_ns()))
    policy=ComparisonPolicy(state,rows[0]["context"],KEYS,trial["method"])
    for row in rows:
        out=policy.process(row["input"])
        out=json.loads(json.dumps(out))
        if tag.startswith("cv-replay-pilot") and "refinement_ablation_source_decision" not in row["output"]:
            out.pop("refinement_ablation_source_decision",None)
        assert out==row["output"],("packet replay output",trial["trial_id"],row["step"])
        packet=policy.command_packet(out["requested_mmps2"],row["input"]["now_us"],row["step"]+1)
        assert packet==row["issued_command"]
        accepted,status=policy.actuate(row["delivered_command"],row["input"]["now_us"])
        assert (accepted,status)==(row["accepted_mmps2"],row["actuator_status"])
        assert policy.card(out,accepted,status)==row["operator_card"]
    value={"trial_id":trial["trial_id"],"evidence_engine":engine_digest(),"passed":True,"queries":len(rows),"trace_sha256":sha(B/trial["trace"]),"lock_digest":digest}
    dump(marker,value);return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pilot");args=parser.parse_args()
    result=json.loads((__import__("pathlib").Path(args.pilot) if args.pilot else B/"results.json").read_text())
    if not args.pilot:
        assert result["lock_digest"]==sha(B/"experiment_lock.json")
        assert all(sha(P/n)==h for n,h in json.loads((B/"experiment_lock.json").read_text()).items())
    tag="cv-replay-pilot-v2" if args.pilot else "cv-replay-confirmation"
    from concurrent.futures import ProcessPoolExecutor
    os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
    values=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for value in pool.map(replay_one,[(t,result["lock_digest"],tag) for t in result["trials"]]):
            values.append(value)
            if len(values)%32==0:print("STRICT_REPLAY",len(values),"/",len(result["trials"]),flush=True)
    target=B/("runtime/pilot_replay.json" if args.pilot else "policy_replay.json")
    dump(target,{"passed":True,"evidence_engine":engine_digest(),"trials":len(values),"decisions":sum(v["queries"] for v in values),"lock_digest":result["lock_digest"],"input_results_sha256":sha(__import__("pathlib").Path(args.pilot) if args.pilot else B/"results.json"),"per_trial":values,"scope":"Same frozen production algorithm replayed from strict input packets in fresh state; independent arithmetic verification is separate."})
    print("STRICT_PACKET_REPLAY_PASS",len(values),flush=True)
if __name__=="__main__":main()
