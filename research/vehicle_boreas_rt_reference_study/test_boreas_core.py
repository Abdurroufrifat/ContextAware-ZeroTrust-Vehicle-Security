import struct
import numpy as np
import pytest
def core():
 import boreas_core as c
 return c
def cloud(n=500,seed=17):
 rng=np.random.default_rng(seed);rays=rng.normal(size=(n,3));rays/=np.linalg.norm(rays,axis=1)[:,None]
 t=np.linspace(0,.1,n);v=np.array([10.,1.,.2]);b=np.array([.2,.1,-.05])
 y=np.sum(rays*(v+(t[:,None]-.1)/.1*b),axis=1)
 p=np.zeros((n,8));p[:,:3]=rays*20;p[:,3]=-y;p[:,7]=t
 return p,v,b
def test_decoder_preserves_uint64_flags_and_ignores_nan_flag_bits():
 raw=struct.pack("<8fQ",3,4,0,-5,1,2,3,100000000,0xffffffffffffffff)
 p,f=core().decode(raw)
 assert p.shape==(1,8) and f.dtype==np.dtype("<u8") and int(f[0])==2**64-1
 assert np.isfinite(p).all() and p[0,3]==-5
 assert p[0,7]==pytest.approx(.1)
@pytest.mark.parametrize("raw",[b"",b"x",b"x"*39,b"x"*41])
def test_decoder_rejects_truncated_or_empty(raw):
 with pytest.raises(ValueError):core().decode(raw)
def test_constant_velocity_exact():
 p,v,b=cloud();p[:,3]=-p[:,:3].dot(v)/20
 for method in ["ols3","huber3","huber6time"]:
  s=core().fit(p,method);assert s["status"]=="estimated";np.testing.assert_allclose(core().predict(s,0),v,atol=1e-10)
def test_linear_scan_end_and_query_extrapolation():
 p,v,b=cloud();s=core().fit(p,"huber6time")
 assert s["status"]=="estimated"
 for age in [0,50000,100000]:np.testing.assert_allclose(core().predict(s,age),v+b*age/100000,atol=1e-10)
def test_constant_fit_does_not_pretend_scan_end_trend():
 p,v,b=cloud();s=core().fit(p,"ols3")
 np.testing.assert_allclose(core().predict(s,0),core().predict(s,100000))
 assert np.linalg.norm(core().predict(s,0)-v)>.05
def test_huber_reduces_doppler_outlier_bias():
 p,v,b=cloud();p[:,3]=-p[:,:3].dot(v)/20;p[:20,3]+=20
 a=core().fit(p,"ols3");b=core().fit(p,"huber3")
 assert np.linalg.norm(core().predict(b,0)-v)<np.linalg.norm(core().predict(a,0)-v)/4
def test_rank_deficient_frame_rejected():
 p,v,b=cloud();p[:,:3]=[20,0,0]
 assert core().fit(p,"huber3")["status"]=="rejected"
def test_unobservable_time_columns_rejected():
 p,v,b=cloud();p[:,7]=.1
 assert core().fit(p,"huber6time")["status"]=="rejected"
def test_range_and_nonfinite_rows_are_shared_filters():
 p,v,b=cloud();extra=p[:4].copy();extra[0,:3]=[1,0,0];extra[1,:3]=[200,0,0];extra[2,3]=np.nan;extra[3,7]=np.nan
 s=core().fit(np.r_[p,extra],"huber3");assert s["n_eligible"]==500
def test_too_few_points_rejected():assert core().fit(cloud(29)[0],"ols3")["reason"]=="too_few_points"
def test_bad_scan_timing_rejected():
 p,v,b=cloud();p[:,7]+=.3
 assert core().fit(p,"huber3")["reason"]=="unsupported_scan_timing"
def test_reference_interpolates_vector_before_norm():
 t=np.array([1000000,1100000],dtype=np.int64);v=np.array([[3.,4.,0],[0,0,10.]])
 assert core().reference_speed(t,v,1050000)==pytest.approx(np.linalg.norm([1.5,2,5]))
def test_reference_exact_endpoint_and_missing_support():
 t=np.array([1000000,1100000],dtype=np.int64);v=np.array([[3.,4.,0],[0,0,10.]])
 assert core().reference_speed(t,v,1000000)==5
 assert core().reference_speed(t,v,999999) is None
 assert core().reference_speed(t,v,1100001) is None
def test_reference_rejects_large_gap():
 assert core().reference_speed(np.array([0,200001]),np.zeros((2,3)),100000) is None
def test_invalid_method_and_query_age_rejected():
 p,v,b=cloud()
 with pytest.raises(ValueError):core().fit(p,"unknown")
 s=core().fit(p,"ols3")
 for age in [True,-1,100001,.1]:
  with pytest.raises(ValueError):core().predict(s,age)

def test_scan_end_uses_valid_time_even_if_other_feature_invalid():
 p,v,b=cloud();last=p[-1:].copy();last[0,4]=np.nan;last[0,7]=.15
 s=core().fit(np.r_[p,last],"huber3")
 assert s["end_offset_s"]==pytest.approx(.15) and s["n_eligible"]==500
def test_bad_feature_does_not_hide_out_of_window_timestamp():
 p,v,b=cloud();last=p[-1:].copy();last[0,4]=np.nan;last[0,7]=.25
 assert core().fit(np.r_[p,last],"huber3")["reason"]=="unsupported_scan_timing"
def test_missing_frame_preserved_as_rejection():
 assert core().estimate_frame(None,"0"*64,"huber3")["solution"]["reason"]=="missing_frame"
def test_changed_frame_bytes_rejected_before_fit():
 assert core().estimate_frame(b"x"*40,"0"*64,"huber3")["solution"]["reason"]=="frame_hash_mismatch"
def test_truncated_frame_with_matching_hash_rejected():
 import hashlib
 raw=b"x"*39
 assert core().estimate_frame(raw,hashlib.sha256(raw).hexdigest(),"huber3")["solution"]["reason"]=="invalid_frame_schema"
def test_complete_frame_receives_same_fit():
 import hashlib
 p,v,b=cloud();raw=b"".join(struct.pack("<8fQ",*(list(row[:7])+[row[7]*1e9]),0) for row in p)
 r=core().estimate_frame(raw,hashlib.sha256(raw).hexdigest(),"huber6time")
 assert r["raw_points"]==500 and r["solution"]["status"]=="estimated"
 np.testing.assert_allclose(core().predict(r["solution"],0),v,atol=1e-5)
