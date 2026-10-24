"""Finalize the completed data batch with independent checks and an evidence manifest."""
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run():
    scripts=["report_broader_development.py","verify_broader_outputs.py"]
    for name in scripts:
        result=subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,capture_output=True,text=True)
        (ROOT/(name.removesuffix(".py")+".log")).write_text(result.stdout+result.stderr)
        if result.returncode:
            print(result.stdout+result.stderr,flush=True)
            raise RuntimeError(f"{name} failed; no completion claim")
        print(name+": passed",flush=True)
    tests=subprocess.run([sys.executable,"-m","unittest","discover","-p","test_*.py","-v"],
        cwd=ROOT,capture_output=True,text=True)
    (ROOT/"broader_tests.log").write_text(tests.stdout+tests.stderr)
    if tests.returncode:
        print(tests.stdout+tests.stderr,flush=True)
        raise RuntimeError("Tests failed")
    count=int(re.search(r"Ran (\d+) tests",tests.stderr).group(1))
    summary=json.loads((ROOT/"broader_development_summary.json").read_text())
    verification=json.loads((ROOT/"broader_independent_verification.json").read_text())
    status=dict(batch_complete=True,tests_passed=count,recordings=summary["recording_count"],
        queries=summary["total_queries"],independent_estimator_checks=verification["total_estimator_checks"],
        development_only=True,full_project_complete=False,physical_gate_pass=None)
    (ROOT/"broader_completion.json").write_text(json.dumps(status,indent=2)+"\n")
    lines=["# Broader development results","",
        f'{status["recordings"]} development recordings; {status["queries"]} INS queries; {count} tests passed.',
        "","| Recording | Estimator | Coverage (%) | P95 discrepancy (m/s) | Discrepancies >2 m/s |",
        "|---|---|---:|---:|---:|"]
    for record in summary["recordings"]:
        for method,metrics in record["overall"].items():
            p95=metrics["p95_discrepancy_mps"]
            p95_text=f"{p95:.6f}" if p95 is not None else "unavailable"
            lines.append(f'| {record["recording"]} | {method} | {metrics["coverage_pct"]:.3f} | {p95_text} | {metrics["discrepancy_above_2_mps"]} |')
    lines+=["","All records are development data from one acquisition day. The cross-traffic ego vehicle was practically stationary; it is a stationary control.",
        "","The independent verifier checks availability, pair selection and numerical discrepancies against raw extracted messages.",
        "","These are engineering reference comparisons. No final confirmation, attack robustness, learned-policy novelty or certified physical safety is claimed.",
        "","Full settings, strata and timing diagnostics: broader_development_summary.json. Checks: broader_independent_verification.json and broader_tests.log."]
    (ROOT/"BROADER_RESULTS.md").write_text("\n".join(lines)+"\n")
    initial=ROOT/"file_manifest_initial_pilot.json"
    current=ROOT/"file_manifest.json"
    if current.exists() and not initial.exists(): shutil.copyfile(current,initial)
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py",".json",".md",".txt",".csv",".log",".msg")]
    files+=[ROOT/"SOURCE_LICENSE",ROOT/".gitignore"]
    manifests={}
    for path in sorted(set(files)):
        if path.name.startswith("file_manifest"): continue
        with path.open("rb") as inp: manifests[path.name]=hashlib.file_digest(inp,"sha256").hexdigest()
    current.write_text(json.dumps(manifests,indent=2)+"\n")
    print(json.dumps(status,indent=2),flush=True)
if __name__=="__main__": run()
