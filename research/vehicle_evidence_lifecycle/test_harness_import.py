from pathlib import Path
import importlib.util
def test_namespaced_engine_imports_without_implicit_json():
 p=Path(__file__).with_name("harness_engine.py")
 s=importlib.util.spec_from_file_location("test_lifecycle_engine",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert m.CONFIG["expected_trials"]==495 and callable(m.run_trial)
