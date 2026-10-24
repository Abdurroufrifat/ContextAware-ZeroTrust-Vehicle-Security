from shared import *
import runpy
names=["PROTOCOL.md","test_neural_core.py","neural_core.py","shared.py","prepare_data.py","train_models.py","evaluate_models.py"]
value={name:sha(R/name) for name in names}
if (R/"experiment_lock.json").exists():assert read("experiment_lock.json")==value
else:write("experiment_lock.json",value)
runpy.run_path(str(R/"prepare_data.py"),run_name="__main__")
