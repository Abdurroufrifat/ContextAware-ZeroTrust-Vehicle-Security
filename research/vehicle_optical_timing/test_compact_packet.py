from pathlib import Path
import importlib.util,struct,hashlib,numpy as np,pytest
# Wrong stride, filtering before sampling, restamping or tamper acceptance
# must make these tests on real packed bytes fail.
def adapter():
 p=Path(__file__).with_name("compact_packet.py")
 assert p.exists(),"compact input adapter missing"
 spec=importlib.util.spec_from_file_location("compact_packet",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def cloud(n):
 return b"".join(struct.pack("<fffffIBf",float(i),1.,2.,3.,-float(i),i,7,8.) for i in range(n))
def test_exact_uniform_records_and_original_source_identity():
 m=adapter();b=cloud(1003);h=hashlib.sha256(b).hexdigest();p=m.encode(b,123456789,h);r=m.decode(p)
 want=b"".join(b[i*29:(i+1)*29] for i in np.linspace(0,1002,512,dtype=int))
 assert r["payload"]==want and r["source_count"]==1003 and r["stamp_ns"]==123456789 and r["source_sha256"]==h
@pytest.mark.parametrize("n",[0,1,19,512,513])
def test_short_and_full_clouds_keep_expected_records(n):
 m=adapter();b=cloud(n);r=m.decode(m.encode(b,1,hashlib.sha256(b).hexdigest()))
 assert len(r["payload"])==min(512,n)*29 and r["source_count"]==n
def test_invalid_records_not_removed_before_uniform_selection():
 m=adapter();b=bytearray(cloud(1024));struct.pack_into("<f",b,0,float("nan"))
 r=m.decode(m.encode(bytes(b),1,hashlib.sha256(b).hexdigest()));assert np.isnan(struct.unpack_from("<f",r["payload"])[0])
@pytest.mark.parametrize("offset",[0,10,-1,100])
def test_metadata_and_payload_corruption_rejected(offset):
 m=adapter();b=cloud(513);p=bytearray(m.encode(b,5,hashlib.sha256(b).hexdigest()));p[offset]^=1
 with pytest.raises(ValueError):m.decode(bytes(p))
@pytest.mark.parametrize("b",[b"a",b"\0"*28,b"\0"*30])
def test_incomplete_source_record_rejected(b):
 with pytest.raises(ValueError):adapter().encode(b,1,"00"*32)
def test_truncated_packet_rejected():
 m=adapter();b=cloud(513);p=m.encode(b,1,hashlib.sha256(b).hexdigest())
 with pytest.raises(ValueError):m.decode(p[:-1])
def test_delayed_packet_source_stamp_not_replaced():
 m=adapter();b=cloud(512);r=m.decode(m.encode(b,1_000_000_000,hashlib.sha256(b).hexdigest()))
 query=1_100_000_001;assert query-r["stamp_ns"]>100_000_000
