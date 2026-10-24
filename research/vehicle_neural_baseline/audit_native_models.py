"""Audit pinned native architectures/weights on synthetic tensors, no dataset scoring."""
from pathlib import Path
import os,sys,json,hashlib,ast,zipfile,importlib.util,time,types
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,torch
from input_contract import check
R=Path(__file__).resolve().parent;S=R/"author_source";P=R.parent/"vehicle_matched_evidence"
torch.set_num_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inventory=json.loads((R/"source_inventory.json").read_text());assert sha(R/"author_repository.zip")==inventory["zip_sha256"]
for e in inventory["extracted_source"]:assert sha(S/e["path"])==e["sha256"],e["path"]
names=["PROTOCOL.md","input_contract.py","test_input_contract.py","audit_native_models.py","source_inventory.json"]
frozen={n:sha(R/n) for n in names};lock=R/"experiment_lock.json"
if lock.exists():assert json.loads(lock.read_text())==frozen
else:lock.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
unchanged={str(P/n):sha(P/n) for n in ["baseline_summary.json","ablation_summary.json","reference_triangulation.json","RESEARCH_DECISION.json"]}
spec=importlib.util.spec_from_file_location("pinned_native_network",S/"src/utils/network.py");native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def config(folder):
 tree=ast.parse((S/("src/"+folder+"/main_3D.py") if folder!="radarscenes" else S/"src/radarscenes/main.py").read_text())
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="test")
 values={}
 for node in function.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   name=node.targets[0].id
   if name in ["convnet_params","linearnet_params"]:values[name]=ast.literal_eval(node.value)
   if name=="map_params":
    allowed=(ast.Dict,ast.Constant,ast.Name,ast.Load,ast.Subscript,ast.BinOp,ast.Mult)
    assert all(isinstance(x,allowed) for x in ast.walk(node.value))
    values[name]=eval(compile(ast.Expression(node.value),"<reviewed-map-constants>","eval"),{"__builtins__":{}},values)
 return values
def state_hash(model):
 h=hashlib.sha256()
 for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()
fields={"x":{"quantity":"position","unit":"m","frame":"sensor","semantics_verified":True,"evidence":"official Continental coordinate fields and fixed radar model"},"y":{"quantity":"position","unit":"m","frame":"sensor","semantics_verified":True,"evidence":"official Continental coordinate fields and fixed radar model"},"z":{"quantity":"position","unit":"m","frame":"sensor","semantics_verified":True,"evidence":"official Continental coordinate fields and fixed radar model"},"doppler":{"quantity":"raw_radial_velocity","unit":"m/s","frame":"sensor","semantics_verified":True,"evidence":"official velocity field and source Doppler convention"},"rcs":{"quantity":"radar_cross_section","unit":None,"frame":None,"semantics_verified":False,"evidence":"official reader decodes unsigned byte; cross-dataset dBsm mapping not justified"}}
cases=[("msc3d","mscrad4r","3D_22_Feb_2024_20_52_42"),("nav3d","navinst","3D_FR_24_Feb_2024_20_59_36"),("radarscenes2d","radarscenes","A_S2_24_Feb_2024_21_00_17")]
rows=[]
with zipfile.ZipFile(R/"author_repository.zip") as archive:
 prefix=Path(archive.infolist()[0].filename).parts[0]+"/"
 for name,folder,tag in cases:
  row={"id":name,"dataset_family":folder,"tag":tag,"target_admission":check(name,fields),"dataset_inference_executed":False}
  cfg=config(folder);row["config"]=cfg
  rel=f"checkpoints/{folder}/{tag}/weights.pt";info=archive.getinfo(prefix+rel);assert info.file_size<=16*1024**2
  data=archive.read(info);target=R/"weights"/(name+".pt");target.parent.mkdir(exist_ok=True);target.write_bytes(data)
  row.update(weight_file=str(target.relative_to(R)),weight_sha256=hashlib.sha256(data).hexdigest(),weight_bytes=len(data))
  try:
   checkpoint=torch.load(target,map_location="cpu",weights_only=True)
   model=native.RadarNet(cfg["convnet_params"],cfg["linearnet_params"])
   strict=model.load_state_dict(checkpoint["model_state_dict"],strict=True);assert not strict.missing_keys and not strict.unexpected_keys
   model.eval();before=state_hash(model)
   # Execute only the pure source mapping method; never dataset loaders/eval(line).
   tree=ast.parse((S/("src/"+folder+"/dataset.py")).read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SequenceRadarDataset")
   fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=="_map");scope={"np":np}
   exec(compile(ast.Module(body=[fn],type_ignores=[]),"<pure-author-map>","exec"),scope)
   holder=types.SimpleNamespace(map_params=cfg["map_params"],other_params={"in_channels":2})
   # Artificial, finite synthetic records only. No reference or HeRCULES input.
   raw=np.array([[1.,0.,2.,3.],[2.,1.,4.,5.]],dtype=float) if folder=="radarscenes" else np.array([[1.,0.,0.,0.,0.,2.,3.],[2.,1.,0.,0.,0.,4.,5.]],dtype=float)
   image=scope["_map"](holder,raw);shape=cfg["convnet_params"]["input_size"];assert image.shape==tuple(shape[1:])
   tensor=torch.from_numpy(image).unsqueeze(0)
   start=time.perf_counter_ns()
   with torch.inference_mode():first=model(tensor);second=model(tensor)
   assert tuple(first.shape)==(1,1) and torch.isfinite(first).all() and torch.equal(first,second) and state_hash(model)==before
   row.update(strict_state_dict_loaded=True,synthetic_forward_passed=True,deterministic_repeat=True,parameters_and_buffers_unchanged=True,parameters=sum(p.numel() for p in model.parameters()),input_shape=list(tensor.shape),output_shape=list(first.shape),two_forward_elapsed_ms=(time.perf_counter_ns()-start)*1e-6)
  except Exception as ex:row.update(strict_state_dict_loaded=False,synthetic_forward_passed=False,error=type(ex).__name__+": "+str(ex))
  rows.append(row);print(name,json.dumps({k:row[k] for k in ["target_admission","strict_state_dict_loaded","synthetic_forward_passed"]}),flush=True)
result={"source_commit":inventory["commit"],"torch_version":torch.__version__,"safe_weights_only_loader":True,"models":rows,"available_target_schema":fields,"unchanged_prior_hashes":unchanged,"synthetic_only":True,"target_neural_comparison_executed":False,"reference_gap_resolved":False,"Library03_payloads_accessed":False}
(R/"model_admission_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print("NATIVE_MODEL_ADMISSION_FINISHED",flush=True)
