"""Fixed development-only training. No fresh test traces read."""
from learned_common import *
install_guard()
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import gzip,math,time,warnings
import numpy as np,joblib,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from learned_core import features,project_point,FEATURE_NAMES
from fractions import Fraction as F
cfg=json.loads((R/"PROTOCOL.json").read_text())
assert not (R/"model_selection.json").exists(),"Models already frozen"
trials=json.loads((KW/"results.json").read_text())["trials"]
sets={"train":[],"validation":[]};bindings={};ids={"train":[],"validation":[]}
for t in trials:
 if t["policy"]!="kinematic":continue
 split="train" if t["seed"] in cfg["training_seeds"] else "validation"
 assert t["seed"] in cfg["training_seeds"]+cfg["validation_seeds"]
 path=checked(KW/t["trace"]);bindings[str(path.relative_to(P))]=sha(path);ids[split].append(t["trial_id"])
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line in f:
   row=json.loads(line);d=row["decision"]
   if d["research_speed_mmps"] is None or not d.get("scene_budget_satisfied"):continue
   lo=F(d["witness_low_n"],d["witness_low_d"]);hi=F(d["witness_high_n"],d["witness_high_d"])
   x=features(dict(base=d["base_decision"],packets=row["packets"],scene_packet=row["scene_packet"],now_us=row["now_us"]),lo,hi)
   label=row["ego_speed_mps"]-float((lo+hi)/2000)
   sets[split].append((x,label,lo,hi,row["ego_speed_mps"],t["trial_id"],row["step"]))
assert set(ids["train"]).isdisjoint(ids["validation"])
models={
 "ridge":make_pipeline(StandardScaler(),Ridge(alpha=1.)),
 "random_forest":RandomForestRegressor(n_estimators=100,max_depth=8,min_samples_leaf=8,random_state=7,n_jobs=1),
 **{f"mlp_{seed}":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,16),activation="tanh",solver="lbfgs",alpha=.1,random_state=seed,max_iter=500,max_fun=25000)) for seed in (7,19,31)}}
X=np.array([r[0] for r in sets["train"]]);y=np.array([r[1] for r in sets["train"]])
V=np.array([r[0] for r in sets["validation"]]);records=[];(R/"models").mkdir(exist_ok=True)
for name,model in models.items():
 start=time.perf_counter()
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter("always");model.fit(X,y)
 pred=model.predict(V);errors=[];raw=[];clipped=0
 for residual,row in zip(pred,sets["validation"]):
  _,_,lo,hi,target,_,_=row;point=float((lo+hi)/2)+float(residual)*1000
  output=project_point(point,lo,hi);assert output is not None
  errors.append(abs(output/1000-target));raw.append(abs(point/1000-target));clipped+=output!=round(point)
 path=R/"models"/(name+".joblib");joblib.dump(model,path)
 records.append(dict(name=name,validation_projected_mae_mps=float(np.mean(errors)),validation_raw_mae_mps=float(np.mean(raw)),
  validation_projected_p95_mps=float(np.quantile(errors,.95)),projection_count=clipped,training_seconds=time.perf_counter()-start,
  warnings=[str(w.message) for w in ws],sha256=sha(path)))
 print(name,records[-1]["validation_projected_mae_mps"],flush=True)
winner=min(records,key=lambda x:(x["validation_projected_mae_mps"],x["name"]))["name"]
write("model_selection.json",dict(selected=winner,candidates=records,feature_names=FEATURE_NAMES,
 train_rows=len(X),validation_rows=len(V),trial_ids=ids,input_bindings=bindings,sklearn_version=sklearn.__version__,
 test_data_accessed=False,all_training_data_previously_explored_development=True,
 training_scope="synthetic longitudinal SUMO; learned residual estimator, not an original algorithm principle"))
write("model_lock.json",{str(q.relative_to(P)):sha(q) for q in [R/"PROTOCOL.json",R/"learned_core.py",R/"train_models.py",R/"model_selection.json",*sorted((R/"models").glob("*.joblib"))]})
print("TRAINING_FROZEN",winner,len(X),len(V),flush=True)
