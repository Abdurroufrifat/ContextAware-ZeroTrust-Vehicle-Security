from pathlib import Path
import importlib.util,numpy as np
def test_native_kellner_exact_stationary_scene_velocity():
 p=Path(__file__).with_name("native_compat.py")
 assert p.exists(),"NumPy scalar-shape compatibility adapter is missing"
 spec=importlib.util.spec_from_file_location("native_compat",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 theta=np.linspace(-.8,.8,101)
 out=m.KellnerCompat(b=0,l=2.380,beta=0,trials=300).estimate(-5*np.cos(theta),theta)
 np.testing.assert_allclose(out,[5.,0.],atol=1e-10)
