"""One-time evidence-preserving numerical and rerun hardening; no threshold tuning."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SNAPSHOT=ROOT/"initial_v1_snapshot"
def main():
    if SNAPSHOT.exists(): raise RuntimeError("Initial snapshot already exists; do not overwrite")
    SNAPSHOT.mkdir()
    for name in ("run_experiment.py","data_adapter.py","verify_experiment.py","experiment_summary.json",
                 "fitted_model.json","verification_summary.json","protocol_v1.json","protocol_freeze.json"):
        shutil.copyfile(ROOT/name,SNAPSHOT/name)
    def replace(name,old,new):
        path=ROOT/name
        text=path.read_text()
        if text.count(old)!=1: raise ValueError("Unexpected replacement count in "+name)
        path.write_text(text.replace(old,new))
    replace("data_adapter.py","def build(spec):","def build(spec,runtime=None):")
    replace("data_adapter.py",'    runtime=ROOT/"runtime"\n    runtime.mkdir(exist_ok=True)',
        '    runtime=Path(runtime) if runtime is not None else ROOT/"runtime"\n    runtime.mkdir(parents=True,exist_ok=True)')
    replace("run_experiment.py","import numpy as np","import numpy as np\nfrom fractions import Fraction")
    replace("run_experiment.py",'    online,references,input_hashes=data.build(spec)',
        '    runtime=ROOT/"runtime"/("run_"+os.urandom(8).hex())\n    online,references,input_hashes=data.build(spec,runtime)')
    replace("run_experiment.py",'    runtime=ROOT/"runtime"\n    keys=',"    keys=")
    replace("run_experiment.py",'    (ROOT/"fitted_model.json").write_text(json.dumps(model,indent=2)+"\\n")',
        '    (ROOT/"fitted_model.json").write_text(json.dumps(model,indent=2)+"\\n")\n    (runtime/"fitted_model.json").write_text(json.dumps(model,indent=2)+"\\n")')
    replace("run_experiment.py",'    report=dict(protocol_sha256=protocol_hash,source_extracted_sha256=input_hashes,split_counts=split_counts,',
        '    report=dict(protocol_sha256=protocol_hash,source_extracted_sha256=input_hashes,split_counts=split_counts,\n        runtime_relative_path=str(runtime.relative_to(ROOT)),')
    replace("run_experiment.py",'    (ROOT/"experiment_summary.json").write_text(json.dumps(report,indent=2)+"\\n")',
        '    (ROOT/"experiment_summary.json").write_text(json.dumps(report,indent=2)+"\\n")\n    (runtime/"experiment_summary.json").write_text(json.dumps(report,indent=2)+"\\n")')
    old='        empirical=bool(errors and np.quantile(errors,.95)<=.5 and len(accepted)*100>=95*len(rows) and not any(e>2 for e in errors))'
    new='''        if errors:
            ordered=sorted(errors)
            index=Fraction(19,20)*(len(ordered)-1)
            lo=index.numerator//index.denominator
            weight=index-lo
            exact_p95=Fraction(ordered[lo])*(1-weight)+Fraction(ordered[min(lo+1,len(ordered)-1)])*weight
            empirical=bool(exact_p95<=Fraction(1,2) and len(accepted)*100>=95*len(rows) and not any(e>2 for e in errors))
        else: empirical=False'''
    replace("run_experiment.py",old,new)
    notes=dict(initial_experiment_preserved=True,thresholds_changed=False,source_data_changed=False,
        repairs=["Exact rational interpolation for empirical P95 gate prevents floating-point admission above 0.5",
                 "Fresh runs get unique runtime directories, preserving prior replay databases and signing keys"],
        reason="Regression tests and evidence-preserving reproducibility")
    (ROOT/"hardening_notes.json").write_text(json.dumps(notes,indent=2)+"\n")
    print("Initial results preserved; numerical/rerun hardening applied without tuning")
if __name__=="__main__": main()
