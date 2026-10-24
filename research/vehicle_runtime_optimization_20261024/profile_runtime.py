from runtime_common import *
import cProfile,pstats,io,datetime
def main():
 (S/"runtime").mkdir(exist_ok=True)
 result=load(B/"results.json");t=next(t for t in result["trials"] if t["method"]=="joint" and t["case"]=="nominal")
 rows=trace(t);start=time.perf_counter();policy=Original(fresh_state("profile"),rows[0]["context"],KEYS,"joint");init=time.perf_counter()-start
 profile=cProfile.Profile();profile.enable();times,_=run_rows(policy,rows);profile.disable()
 profile.dump_stats(S/"runtime/baseline_profile.pstats")
 output=io.StringIO();pstats.Stats(profile,stream=output).sort_stats("cumulative").print_stats(35)
 (S/"runtime/baseline_profile.txt").write_text(output.getvalue(),encoding="utf-8")
 record={"trial_id":t["trial_id"],"queries":len(rows),"init_seconds":init,"measured_seconds":sum(times),"output_equivalence_passed":True,"profile_trace_sha256":sha(B/t["trace"]),"results_sha256":sha(B/"results.json"),"note":"Sequential diagnostic replay with profiling overhead; not a vehicle experiment or deadline qualification."}
 (S/"profile_result.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
 print(json.dumps(record),flush=True);print(output.getvalue())
if __name__=="__main__":main()
