import json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent.parent;runtime=R/"runtime";stage=runtime/"package_source"
assert not stage.exists();stage.mkdir(parents=True)
for name in ("pyproject.toml","README.md"):shutil.copy2(P/name,stage/name)
shutil.copytree(P/"src",stage/"src",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
dest=runtime/"wheels";dest.mkdir(exist_ok=True);tmp=runtime/"build_tmp";tmp.mkdir(exist_ok=True)
env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONDONTWRITEBYTECODE="1")
result=subprocess.run([sys.executable,"-m","pip","wheel","--no-index","--no-deps","--no-build-isolation","--no-cache-dir",str(stage),"-w",str(dest)],capture_output=True,text=True,env=env)
(R/"packaging.log").write_text(result.stdout+result.stderr)
wheels=list(dest.glob("*.whl"));verified=False
if result.returncode==0 and len(wheels)==1:
 with zipfile.ZipFile(wheels[0]) as z:
  assert z.testzip() is None and "coact_cps/cli.py" in z.namelist()
  metadata=next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
  assert "Research software for cyber-physical incident diagnosis" in z.read(metadata).decode()
  verified=True
report=dict(exit_code=result.returncode,wheel_verified=verified,wheels=[str(p.relative_to(P)) for p in wheels],
 dependencies_installed=False,network_used=False,source_code_changed=False,missing_declared_readme_repaired=True)
(R/"packaging_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
sys.exit(result.returncode)
