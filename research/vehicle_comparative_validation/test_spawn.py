import subprocess,sys,json,os
from pathlib import Path
def test_windows_spawn_resolves_this_studys_policy_dependencies(tmp_path):
    b=Path(__file__).resolve().parent
    script=tmp_path/"spawn_import_check.py"
    source="import sys\nimport runner\nfrom concurrent.futures import ProcessPoolExecutor\ndef child(value):\n from policy import ComparisonPolicy\n return ComparisonPolicy.__name__\nif __name__=='__main__':\n with ProcessPoolExecutor(max_workers=1) as pool:\n  assert list(pool.map(child,[0]))==['ComparisonPolicy']\n"
    script.write_text(source,encoding="utf-8")
    r=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
