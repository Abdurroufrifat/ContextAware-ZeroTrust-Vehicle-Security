from cs_common import *
install_guard()
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import gzip,collections,time,warnings
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from command_scene_core import refine,relative_features
from learned_core import project_point
from fractions import Fraction as F
assert not (R/"model_selection.json").exists()
cfg=json.loads((R/"PROTOCOL.json").read_text());sets={"train":[],"validation":[]};inputs={};audit=[];ids={"train":[],"validation":[]}
for trial in json.loads((KW/"results.json").read_text())["trials"]:
 if trial["policy"]!="kinematic":continue
 split="train" if trial["seed"] in cfg["training_seeds"] else "validation"
 assert trial["seed"] in cfg["training_seeds"]+cfg["validation_seeds"]
 path=checked(KW/trial["trace"]);inputs[str(path.relative_to(P))]=sha(path);ids[split].append(trial["trial_id"]);previous=None
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line in f:
   row=json.loads(line);scene=row["decision"];base=scene["base_decision"]
   body=json.loads(row["scene_packet"]["wire"])["body"] if scene.get("scene_status")=="authenticated" else None
   before=previous
   if body is not None:previous=body
   why=None;refined=None
   if trial["case"] in cfg["exclude_training_cases"]:why="excluded_broken_trust_domain"
   elif body is None or before is None or scene["research_speed_mmps"] is None or not scene["scene_budget_satisfied"]:why="no_valid_scene_admission"
   elif base["command_status"]!="authenticated":why="command_not_authenticated"
   else:
    broad=(F(scene["witness_low_n"],scene["witness_low_d"]),F(scene["witness_high_n"],scene["witness_high_d"]))
    commands=json.loads(row["packets"]["command"]["wire"])["body"]["commands"]
    refined=refine(before["gap_um"],body["gap_um"],before["lead_mmps"],body["lead_mmps"],before["measured_us"],row["now_us"],commands,broad)
    if refined["status"]!="available":why="refinement_"+refined["status"]
   if why is None:
    x=relative_features(base,row["packets"],body,row["now_us"],refined)
    target=row["ego_speed_mps"]-float(refined["center"]/1000)
    sets[split].append((x,target,refined["low"],refined["high"],row["ego_speed_mps"]))
   audit.append(dict(trial_id=trial["trial_id"],step=row["step"],split=split,included=why is None,reason=why))
assert set(ids["train"]).isdisjoint(ids["validation"])
models={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.)),
 "forest":RandomForestRegressor(n_estimators=60,max_depth=6,min_samples_leaf=16,random_state=7,n_jobs=1),
 **{f"mlp_{s}":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,16),activation="tanh",solver="lbfgs",alpha=.1,random_state=s,max_iter=500,max_fun=20000)) for s in (7,19,31)}}
X=np.array([r[0] for r in sets["train"]]);y=np.array([r[1] for r in sets["train"]]);V=np.array([r[0] for r in sets["validation"]])
(R/"models").mkdir(exist_ok=True);records=[]
for name,model in models.items():
 begin=time.perf_counter()
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter("always");model.fit(X,y)
 pred=model.predict(V);errors=[]
 for residual,row in zip(pred,sets["validation"]):
  _,_,lo,hi,truth=row
  output=project_point(float((lo+hi)/2)+float(residual)*1000,lo,hi);assert output is not None
  errors.append(abs(output/1000-truth))
 path=R/"models"/(name+".joblib");joblib.dump(model,path)
 records.append(dict(name=name,validation_mae_mps=float(np.mean(errors)),validation_p95_mps=float(np.quantile(errors,.95)),sha256=sha(path),
  seconds=time.perf_counter()-begin,warnings=[str(w.message) for w in ws]))
 print(name,records[-1]["validation_mae_mps"],flush=True)
winner=min(records,key=lambda r:(r["validation_mae_mps"],r["name"]))["name"]
miderrors=[abs(float((r[2]+r[3])/2000)-r[4]) for r in sets["validation"]]
write("training_row_audit.json",audit)
write("model_selection.json",dict(selected=winner,candidates=records,training_rows=len(X),validation_rows=len(V),
 midpoint_validation_mae_mps=float(np.mean(miderrors)),input_bindings=inputs,trial_ids=ids,
 inclusion_counts=dict(collections.Counter(a["reason"] or "included" for a in audit)),
 test_results_accessed=False,feature_count=8,absolute_speed_gap_profile_seed_attack_label_features=False,
 scope="Development-only refit; independent testing required."))
write("model_lock.json",{str(q.relative_to(P)):sha(q) for q in [R/"PROTOCOL.json",R/"command_scene_core.py",R/"command_scene_guard.py",R/"train_models.py",R/"model_selection.json",*sorted((R/"models").glob("*.joblib"))]})
print("MODELS_FROZEN",winner,len(X),len(V),flush=True)
