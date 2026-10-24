from pathlib import Path
import json,hashlib,importlib.util,ast,warnings,torch
R=Path(__file__).resolve().parent;S=R/"author_source"
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((R/"model_admission_audit.json").read_text());inventory=json.loads((R/"source_inventory.json").read_text())
for n,h in json.loads((R/"experiment_lock.json").read_text()).items():assert sha(R/n)==h,n
assert sha(R/"author_repository.zip")==inventory["zip_sha256"]
for e in inventory["extracted_source"]:assert sha(S/e["path"])==e["sha256"]
for p,h in audit["unchanged_prior_hashes"].items():assert sha(Path(p))==h
spec=importlib.util.spec_from_file_location("verified_native_network",S/"src/utils/network.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
count=wrong=0;notes=[]
for row in audit["models"]:
 assert row["strict_state_dict_loaded"] and row["synthetic_forward_passed"] and row["deterministic_repeat"] and row["parameters_and_buffers_unchanged"]
 assert row["dataset_inference_executed"] is False and row["target_admission"]["admitted"] is False
 path=R/row["weight_file"];assert sha(path)==row["weight_sha256"] and path.stat().st_size==row["weight_bytes"]
 main=S/("src/"+row["dataset_family"]+("/main.py" if row["dataset_family"]=="radarscenes" else "/main_3D.py"))
 with warnings.catch_warnings(record=True) as warning_list:
  warnings.simplefilter("always");tree=ast.parse(main.read_text())
 notes.extend(str(w.message) for w in warning_list)
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="test")
 for name in ["convnet_params","linearnet_params"]:
  node=next(n for n in function.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
  # Stored JSON uses lists instead of literal tuples; normalize both sides.
  assert json.loads(json.dumps(ast.literal_eval(node.value)))==row["config"][name]
 cfg=row["config"];model=module.RadarNet(cfg["convnet_params"],cfg["linearnet_params"])
 state=torch.load(path,map_location="cpu",weights_only=True)["model_state_dict"];result=model.load_state_dict(state,strict=True);assert not result.missing_keys and not result.unexpected_keys
 model.eval();shape=cfg["convnet_params"]["input_size"];x=torch.ones(shape)
 with torch.inference_mode():y=model(x);again=model(x)
 assert y.shape==(1,1) and torch.isfinite(y).all() and torch.equal(y,again)
 assert all(torch.equal(value,state[key]) for key,value in model.state_dict().items())
 try:
  with torch.inference_mode():model(torch.zeros((1,shape[1]-1,*shape[2:])))
 except RuntimeError:wrong+=1
 else:raise AssertionError("wrong feature channel count accepted")
 assert sum(p.numel() for p in model.parameters())==row["parameters"];count+=1
fields=audit["available_target_schema"]
assert "power" not in fields and "snr" not in fields and fields["rcs"]["unit"] is None and not fields["rcs"]["semantics_verified"]
assert audit["safe_weights_only_loader"] and audit["synthetic_only"] and not audit["target_neural_comparison_executed"]
scope=json.loads((R/"EVALUATION_SCOPE.json").read_text());assert not scope["reference_policy"]["label_replacement"] and not scope["final_policy"]["physical_witness_admitted"]
tests=json.loads((R/"test_validation.json").read_text());assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
out={"passed":True,"native_architecture_weight_pairs_reverified":count,"independent_wrong_channel_rejections":wrong,"feature_contract_blocks":len(audit["models"]),"safe_weights_only_loads":count,"native_source_parser_warnings":sorted(set(notes)),"unchanged_prior_results":True,"real_dataset_neural_forward_count":0,"valid_neural_performance_comparison_executed":False,"reference_gap_resolved":False,"Library03_payloads_accessed":False,"tests":tests}
assert count==wrong==3
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="tests"},indent=2))
