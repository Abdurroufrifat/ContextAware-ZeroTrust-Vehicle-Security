from pathlib import Path
import importlib.util,numpy as np,pytest
def core():
 p=Path(__file__).with_name("evidence_core.py")
 assert p.exists(),"matched metrics/ablation core missing"
 s=importlib.util.spec_from_file_location("evidence_core",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_unknown_reference_not_success_or_removed_from_denominator():
 x=core().metrics([1.,2.,3.],[1.,np.nan,8.],[True,True,False])
 assert x["attempts"]==3 and x["admitted"]==2 and x["scored"]==1 and x["unknown_reference"]==1 and x["availability"]==2/3
def test_rejected_large_error_not_counted_as_successful_scored_prediction():
 x=core().metrics([100.,1.],[0.,1.],[False,True])
 assert x["above2"]==0 and x["scored"]==1 and x["attempts"]==2
def test_missing_prediction_cannot_be_admitted():
 x=core().metrics([np.nan],[0.],[True]);assert x["admitted"]==0 and x["invalid_prediction"]==1
def test_pairing_preserved_by_temporal_resampling():
 a=np.array([0.,20.,3.,5.,100.,2.,1.]);b=a+2
 x=core().paired_mae_ci(a,b,3,100,42)
 assert x["difference"]==-2 and x["low"]==-2 and x["high"]==-2
def test_identical_methods_have_zero_interval():
 x=core().paired_mae_ci(np.arange(20.),np.arange(20.),5,100,42);assert x["low"]==x["high"]==0
def test_bootstrap_reproducible():
 m=core();a=np.arange(20.)**2;b=np.arange(20.)
 assert m.paired_mae_ci(a,b,5,100,42)==m.paired_mae_ci(a,b,5,100,42)
def test_missing_reference_cannot_enter_paired_bootstrap():
 with pytest.raises(ValueError):core().paired_mae_ci([1.,np.nan],[2.,3.],2,10,1)
def test_stale_optical_does_not_silently_disable_radar_only_baseline():
 m=core().decisions(np.array([8.]),np.array([8.]),np.array([0]),np.array([100_000_001]),np.array([True]),True)
 assert m["radar_only"][0] and not m["optical_only"][0] and not m["agreement_guard"][0]
def test_single_and_common_sensor_corruption_distinguished():
 c=core();z=np.array([0]);s=np.array([True])
 single=c.decisions(np.array([11.]),np.array([8.]),z,z,s,True)
 both=c.decisions(np.array([11.]),np.array([11.]),z,z,s,True)
 assert single["radar_only"][0] and not single["agreement_guard"][0] and both["agreement_guard"][0]
def test_future_and_unsupported_sources_rejected():
 z=np.array([0]);r=core().decisions(np.array([8.]),np.array([8.]),np.array([-1]),z,np.array([True]),True)
 assert not r["radar_only"][0] and not r["agreement_guard"][0]
 r=core().decisions(np.array([8.]),np.array([8.]),z,z,np.array([False]),True)
 assert not any(v.any() for v in r.values())
