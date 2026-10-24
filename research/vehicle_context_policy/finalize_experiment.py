"""Save reviewable findings and versioned provenance after independent verification."""
import hashlib,json,platform,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    tests=subprocess.run([sys.executable,"-m","unittest","discover","-p","test_*.py","-v"],cwd=ROOT,capture_output=True,text=True)
    (ROOT/"tests.log").write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0
    count=int(re.search(r"Ran (\d+) tests",tests.stderr).group(1))
    report=json.loads((ROOT/"experiment_summary.json").read_text())
    check=json.loads((ROOT/"verification_summary.json").read_text())
    assert check["all_passed"] and check["independent_policy_decisions"]==report["policy_decisions"]
    original=json.loads((ROOT/"initial_v1_snapshot"/"experiment_summary.json").read_text())
    assert original["results"]==report["results"] and original["per_record"]==report["per_record"]
    lines=["# Context-aware policy development findings","",
        f'Compared {len(report["policies"])} policies in 12 cases; {report["packet_events"]} packet events and {report["policy_decisions"]} decisions.',
        f'New module tests: {count} passed. Independent model, packet and decision checks passed.',
        "","Training: 473 earlier-block samples. Calibration: 232 separate samples. Later-block evaluation: 283 queries per case.",
        "","| Policy | Clean coverage (%) | Clean P95 discrepancy (m/s) | Coordinated-case accepted discrepancies >2 m/s |",
        "|---|---:|---:|---:|"]
    for policy in report["policies"]:
        n=report["results"]["nominal"][policy];c=report["results"]["coordinated_bias"][policy]
        p=f'{n["p95_discrepancy_mps"]:.6f}' if n["p95_discrepancy_mps"] is not None else "unavailable"
        lines.append(f'| {policy} | {n["coverage_pct"]:.3f} | {p} | {c["accepted_discrepancy_above_2_mps"]} |')
    full=report["results"]["nominal"]["context_learned"]
    bad=report["results"]["coordinated_bias"]["context_learned"]
    lines+=["","## Decision",
        f'The complete context model fails the clean availability target: {full["coverage_pct"]:.3f}% versus the unchanged 95% requirement.',
        f'It also accepts {bad["accepted_discrepancy_above_2_mps"]} coordinated-case outputs with discrepancy above 2 m/s.',
        "The version is not promoted. No thresholds were retuned after observing these results.",
        "","Context regression without the support check has higher availability, but removing that check does not establish protection against common CAN compromise.",
        "","Authentication, stale/future checks and replay protection constrain message handling. They do not certify physical measurements or create independent sources.",
        "","Some other policies accept one nominal output without an INS comparison near a recording boundary. Such outputs remain explicitly unscorable.",
        "","## Next work",
        "Address training-context coverage and add evidence from a different physical source before claiming coordinated-compromise protection. Processed radar/LiDAR timing must retain its true source age. Freeze a new development protocol before changing the method, and keep this failed version as the baseline.",
        "","These are previously inspected development records from one day. AI novelty, independent confirmation, full published-method comparison and journal readiness remain unresolved."]
    (ROOT/"FINDINGS.md").write_text("\n".join(lines)+"\n")
    runtime=ROOT/report["runtime_relative_path"]
    (ROOT/"environment.json").write_text(json.dumps(dict(python=platform.python_version(),
        executable=sys.executable,numpy=__import__("numpy").__version__),indent=2)+"\n")
    provenance={}
    for name in ("protocol_v1.json","protocol_freeze.json","fitted_model.json","experiment_summary.json","verification_summary.json"):
        provenance[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ("online_inputs.json","offline_references.json","packet_events.jsonl","decisions.jsonl"):
        provenance[str(Path(report["runtime_relative_path"])/name)]=hashlib.sha256((runtime/name).read_bytes()).hexdigest()
    for module in ("vehicle_evidence_auth/auth_guard.py","vehicle_monitor_comparison/monitor_core.py"):
        provenance["dependency:"+module]=hashlib.sha256((ROOT.parent/module).read_bytes()).hexdigest()
    (ROOT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    completion=dict(batch_complete=True,tests_passed=count,independent_decisions=report["policy_decisions"],
        initial_results_unchanged_after_hardening=True,method_passes_requirements=False,
        full_project_complete=False,production_promotion_allowed=False)
    (ROOT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
    files=[p for p in ROOT.iterdir() if p.is_file() and p.name!="file_manifest.json"]
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(completion,indent=2))
if __name__=="__main__": main()
