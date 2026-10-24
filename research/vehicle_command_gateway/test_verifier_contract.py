"""Mutate parsed summary in a subprocess; frozen source files stay untouched."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent
@pytest.mark.parametrize("mutation",["empty","omitted","duplicate","wrong_label","wrong_count","wrong_decisions","wrong_queries"])
def test_verifier_rejects_invalid_summary_inventory(mutation):
 code="""
import json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);mutation=sys.argv[2];sys.path.insert(0,str(root))
import common
common.write=lambda *args,**kwargs:None
target=(root/"results.json").read_text();original=json.loads
def changed(text,*args,**kwargs):
 value=original(text,*args,**kwargs)
 if text==target:
  if mutation=="empty":value["conditions"]=[]
  if mutation=="omitted":value["conditions"]=value["conditions"][:-1]
  if mutation=="duplicate":value["conditions"][-1]=value["conditions"][0]
  if mutation=="wrong_label":value["conditions"][0]["case"]="wrong"
  if mutation=="wrong_count":value["condition_count"]=0
  if mutation=="wrong_decisions":value["decisions"]=0
  if mutation=="wrong_queries":value["conditions"][0]["queries"]=0
 return value
json.loads=changed
runpy.run_path(str(root/"verify_matrix.py"),run_name="__main__")
"""
 env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}
 out=subprocess.run([sys.executable,"-c",code,str(ROOT),mutation],capture_output=True,text=True,env=env)
 assert out.returncode!=0,("Verifier incorrectly accepted",mutation,out.stdout)
