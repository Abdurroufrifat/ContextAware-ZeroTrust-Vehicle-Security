import importlib.util,struct
from pathlib import Path
import numpy as np
import pytest
def core():
 p=Path(__file__).with_name("pilot_core.py")
 assert p.exists(),"pilot decoder and timestamp adapter are not implemented"
 spec=importlib.util.spec_from_file_location("pilot_core",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_literal_lzf():
 assert core().lzf_decode(b"\x02abc",3)==b"abc"
def test_overlapping_backreference():
 assert core().lzf_decode(b"\x00a\x80\x00",7)==b"aaaaaaa"
@pytest.mark.parametrize("b,n",[(b"\x02ab",3),(b"\x20\x01",3),(b"\x00a",2),(b"\x01ab",1)])
def test_corrupt_lzf_rejected(b,n):
 with pytest.raises(ValueError):core().lzf_decode(b,n)
def test_integer_epoch_keeps_nanoseconds():
 assert core().time_ns("1680409351","060973254")==1680409351060973254
@pytest.mark.parametrize("sec,ns",[("1","1000000000"),("1","-1"),("1.5","0")])
def test_invalid_time_rejected(sec,ns):
 with pytest.raises(ValueError):core().time_ns(sec,ns)
def test_pcd_field_major_not_interleaved():
 raw=np.array([1.,2.],dtype="<f4").tobytes()+np.array([7,8],dtype="<u2").tobytes()
 comp=bytes([len(raw)-1])+raw
 h=b"VERSION 0.7\nFIELDS x flag\nSIZE 4 2\nTYPE F U\nCOUNT 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary_compressed\n"
 pts=core().decode_pcd(h+struct.pack("<II",len(comp),len(raw))+comp)
 np.testing.assert_array_equal(pts["x"],[1.,2.]);np.testing.assert_array_equal(pts["flag"],[7,8])
def test_reference_matching_keeps_radar_age_separate():
 m=core();assert m.nearest_reference(105,[100,110],max_gap=5)==(0,5)
 assert m.nearest_reference(106,[100,110],max_gap=4)==(1,4)
 assert m.nearest_reference(120,[100,110],max_gap=4) is None
def test_reference_grid_must_increase():
 with pytest.raises(ValueError):core().nearest_reference(100,[100,99],max_gap=10)
