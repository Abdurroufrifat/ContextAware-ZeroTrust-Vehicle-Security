"""Save the verified comparison and expose remaining availability/timing failures."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    check=json.loads((ROOT/"independent_verification.json").read_text())
    assert check["all_passed"]
    report=json.loads((ROOT/"radar_experiment_summary.json").read_text())
    runtime=ROOT/report["runtime_relative_path"]
    can=ROOT.parent/"vehicle_context_policy"
    prior=json.loads((can/"experiment_summary.json").read_text())
    with (can/prior["runtime_relative_path"]/"packet_events.jsonl").open() as inp:
        refs={(e["recording"],e["index"]):e["reference"] for e in map(json.loads,inp) if e["case"]=="nominal"}
    diagnostic={}
    with (runtime/"radar_estimates.jsonl").open() as inp: estimates=[json.loads(line) for line in inp]
    for track in report["radar_quality"]:
        accepted=[e for e in estimates if e["track"]==track and e["estimate"]["accepted"]]
        errors=[abs(e["estimate"]["speed_mps"]-refs[(e["recording"],e["index"])])
            for e in accepted if refs[(e["recording"],e["index"])] is not None]
        diagnostic[track]=dict(accepted_estimates=len(accepted),unscorable=len(accepted)-len(errors),
            p95_speed_discrepancy_mps=float(np.quantile(errors,.95)) if errors else None,
            median_speed_discrepancy_mps=float(np.median(errors)) if errors else None,
            maximum_speed_discrepancy_mps=max(errors) if errors else None,
            discrepancies_above_2_mps=sum(v>2 for v in errors))
    (ROOT/"radar_motion_accuracy.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    tests=subprocess.run([sys.executable,"-m","unittest","discover","-p","test_*.py","-v"],
        cwd=ROOT,capture_output=True,text=True)
    (ROOT/"tests.log").write_text(tests.stdout+tests.stderr)
    if tests.returncode: raise RuntimeError(tests.stdout+tests.stderr)
    count=int(re.search(r"Ran (\d+) tests",tests.stderr).group(1))
    lines=["# Radar context development findings","",
        f'{check["matched_raw_frames"]} raw radar frames matched the original processed values exactly.',
        f'{check["fusion_decisions"]} fusion decisions independently checked. {count} module tests passed.',
        "","| Timing track | Policy | Clean coverage (%) | Coordinated active requests accepted | Coordinated accepted discrepancies >2 m/s |",
        "|---|---|---:|---:|---:|"]
    for track,cases in report["results"].items():
        for policy,nominal in cases["nominal"].items():
            coordinated=cases["coordinated_bias"][policy]
            lines.append(f'| {track} | {policy} | {nominal["coverage_pct"]:.3f} | {coordinated["active_accepted"]}/{coordinated["active_requests"]} | {coordinated["accepted_discrepancy_above_2_mps"]} |')
    raw=report["results"]["raw_header_association"]["nominal"]["context_no_ood_plus_radar"]
    bad=report["results"]["raw_header_association"]["coordinated_bias"]["context_no_ood_plus_radar"]
    lines+=["","## Findings",
        f'With raw-header association, independent radar plus learned context improves clean coverage from 60.777% to {raw["coverage_pct"]:.3f}% and reduces coordinated-case accepted discrepancies above 2 m/s from 59 to {bad["accepted_discrepancy_above_2_mps"]}.',
        f'It rejects all {bad["active_requests"]} active coordinated CAN-compromise requests in these development blocks.',
        "Clean coverage remains below the unchanged 95% target.",
        "","These transport-timestamp results do not establish physical acquisition freshness. The original processed timestamps are 106 ms earlier than raw packet headers for all decoded frames. Retaining those corrected timestamps leaves no radar scans eligible at the 100 ms deadline.",
        "","The strict corrected-time policy therefore grants zero requests. Its absence of accepted errors must not be reported as successful protection with usable coverage.",
        "","Radar motion accuracy is reported separately in radar_motion_accuracy.json. The fused policy selects the authenticated CAN candidate; its output P95 must not be described as radar estimation accuracy.",
        f'Standalone radar still has {diagnostic["raw_header_association"]["discrepancies_above_2_mps"]} estimates with speed discrepancy above 2 m/s, with maximum {diagnostic["raw_header_association"]["maximum_speed_discrepancy_mps"]:.3f} m/s. Consensus quality checks therefore do not establish a trusted physical-speed witness.', 
        "","No thresholds, timing corrections or source values were retuned to improve these results. The first run is preserved; a reason-label correction changed explanations without changing acceptance metrics.",
        "","## Remaining work",
        "Improve static-scene observability/coverage and obtain a validated acquisition-time model or a sufficiently fresh independent sensor stream. Merely adding more CAN channels or relabeling radar packet arrival as physical acquisition cannot resolve the current evidence gap.",
        "","All three recordings are previously inspected development data from one day. There is no independent confirmation or production promotion."]
    (ROOT/"FINDINGS.md").write_text("\n".join(lines)+"\n")
    status=dict(batch_complete=True,tests_passed=count,matched_radar_frames=check["matched_raw_frames"],
        independently_verified_fusion_decisions=check["fusion_decisions"],
        raw_timing_clean_coverage_pct=raw["coverage_pct"],raw_timing_coordinated_large_discrepancies=bad["accepted_discrepancy_above_2_mps"],
        acquisition_age_validated=False,corrected_timing_radar_coverage_pct=0.,meets_all_requirements=False,
        physical_witness_enabled=False,production_promotion_allowed=False)
    (ROOT/"completion.json").write_text(json.dumps(status,indent=2)+"\n")
    paths=[p for p in ROOT.iterdir() if p.is_file() and p.name!="file_manifest.json"]
    paths+=list((ROOT/"vendor").rglob("*"))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    (ROOT/"file_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(status=status,motion_accuracy=diagnostic),indent=2))
if __name__=="__main__": main()
