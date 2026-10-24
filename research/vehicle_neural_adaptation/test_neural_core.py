import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
def core():
    p=Path(__file__).with_name("neural_core.py")
    assert p.exists(),"neural_core implementation missing"
    spec=importlib.util.spec_from_file_location("tested_neural_core",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_temporal_split_has_explicit_two_second_purge():
    m=core();t=np.arange(101,dtype=np.int64)*1_000_000_000
    s=m.split_labels(t,"Library01")
    assert set(np.flatnonzero(s=="train"))==set(range(58))
    assert set(np.flatnonzero(s=="validation"))==set(range(62,78))
    assert set(np.flatnonzero(s=="temporal_test"))==set(range(82,101))
    assert sum(s=="purge")==8
def test_transfer_never_becomes_training():
    m=core();assert set(m.split_labels(np.arange(10),"Library02"))=={"transfer"}
@pytest.mark.parametrize("name",["Library03","RAV4","Urban02 NIGHT","unknown"])
def test_reserved_or_unknown_sequences_rejected(name):
    with pytest.raises(ValueError):core().split_labels(np.arange(5),name)
def test_split_rejects_duplicate_or_unsorted_stamps():
    with pytest.raises(ValueError):core().split_labels(np.array([1,1,2]),"Library01")
    with pytest.raises(ValueError):core().split_labels(np.array([2,1,3]),"Library01")
def test_features_use_only_rays_doppler_and_rotation():
    m=core();r=np.array([[1.,0,0],[0,1,0],[0,0,1]])
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    f=m.point_features(r,np.array([-20.,10.,0.]),rot)
    np.testing.assert_allclose(f[:,:3],r@rot.T)
    np.testing.assert_allclose(f[:,3],[1,-.5,0])
    assert f.shape==(3,4) and f.dtype==np.float32
@pytest.mark.parametrize("bad",["nan","empty","nonunit","rotation"])
def test_invalid_point_inputs_fail_explicitly(bad):
    m=core();a=np.array([[1.,0,0]]);d=np.array([1.]);rot=np.eye(3)
    if bad=="nan":d[0]=np.nan
    if bad=="empty":a=a[:0];d=d[:0]
    if bad=="nonunit":a*=2
    if bad=="rotation":rot[0,0]=2
    with pytest.raises(ValueError):m.point_features(a,d,rot)
def test_pooling_ignores_padding_and_point_order():
    m=core();torch.manual_seed(11);net=m.SpeedSet().eval()
    a=np.arange(28,dtype=np.float32).reshape(7,4)/20;b=np.ones((3,4),np.float32)
    x,mask=m.pack([a,b]);alone,am=m.pack([b]);perm,pm=m.pack([a[::-1].copy(),b[::-1].copy()])
    with torch.inference_mode():
        y=net(x,mask);single=net(alone,am);other=net(perm,pm)
    torch.testing.assert_close(y[1],single[0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(y,other,atol=1e-6,rtol=1e-6)
    assert y.shape==(2,) and bool((y>=0).all())
def test_empty_mask_is_rejected():
    m=core()
    with pytest.raises(ValueError):m.SpeedSet()(torch.zeros(1,3,4),torch.zeros(1,3,dtype=torch.bool))
def test_padding_rejects_nonfinite_and_empty_frames():
    m=core()
    with pytest.raises(ValueError):m.pack([np.zeros((0,4),np.float32)])
    with pytest.raises(ValueError):m.pack([np.full((2,4),np.nan,np.float32)])
def test_epoch_selection_uses_validation_only_and_earliest_tie():
    m=core()
    rows=[{"epoch":1,"validation_mae":.4,"test_mae":0.},
          {"epoch":2,"validation_mae":.2,"test_mae":99.},
          {"epoch":3,"validation_mae":.2,"test_mae":-1.}]
    assert m.selected_epoch(rows)==2
def test_missing_validation_cannot_select_checkpoint():
    with pytest.raises(ValueError):core().selected_epoch([{"epoch":1,"validation_mae":float("nan")}])
def test_model_can_learn_synthetic_doppler_speed_relation():
    m=core();torch.set_num_threads(1);torch.manual_seed(4)
    data=[np.tile(np.array([1,0,0,v/20],np.float32),(8,1)) for v in [1.,2.,3.,4.]]
    x,mask=m.pack(data);target=torch.tensor([1.,2.,3.,4.]);net=m.SpeedSet()
    opt=torch.optim.Adam(net.parameters(),lr=.01)
    before=float(torch.mean((net(x,mask)-target)**2).detach())
    for _ in range(120):
        opt.zero_grad();loss=torch.mean((net(x,mask)-target)**2);loss.backward();opt.step()
    after=float(torch.mean((net(x,mask)-target)**2).detach())
    assert after<before*.2
