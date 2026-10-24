from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
CODE=('core_io.py','final_core.py','estimator_core.py','make_dataset.py','models_torch.py','study_io.py','train_estimator.py','score_estimator.py')
def freeze(write_lock=False):
 protocol=read('protocol.json')
 for name,expected in protocol['input_sha256'].items():
  if digest(name)!=expected:raise ValueError('input source changed: '+name)
 lock={'protocol_sha256':digest('protocol.json'),'code_sha256':{name:digest(name) for name in CODE}}
 path=ROOT/'design_lock.json'
 if path.exists() and read('design_lock.json')!=lock:raise ValueError('frozen estimator implementation changed')
 if write_lock and not path.exists():write('design_lock.json',lock)
 return lock
