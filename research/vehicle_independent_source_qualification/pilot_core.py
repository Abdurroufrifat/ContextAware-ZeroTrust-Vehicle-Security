"""Bounded compressed-PCD and exact timestamp helpers; no policy decisions."""
import struct,re
import numpy as np
def lzf_decode(data,expected):
 if not 0<=expected<=16*1024*1024:raise ValueError("decompressed size cap")
 out=bytearray();i=0
 try:
  while i<len(data):
   ctrl=data[i];i+=1
   if ctrl<32:
    n=ctrl+1
    if i+n>len(data) or len(out)+n>expected:raise ValueError("literal size")
    out.extend(data[i:i+n]);i+=n
   else:
    n=ctrl>>5;ref=len(out)-((ctrl&31)<<8)-1
    if n==7:n+=data[i];i+=1
    ref-=data[i];i+=1;n+=2
    if ref<0 or len(out)+n>expected:raise ValueError("backreference")
    for j in range(n):out.append(out[ref+j])
 except IndexError as e:raise ValueError("truncated LZF") from e
 if len(out)!=expected:raise ValueError("LZF decoded size")
 return bytes(out)
def decode_pcd(data):
 match=re.search(rb"(?m)^DATA binary_compressed\r?\n",data[:8192])
 if not match:raise ValueError("expected bounded binary_compressed PCD")
 header={}
 for line in data[:match.start()].decode("ascii").splitlines():
  if line and not line.startswith("#"):
   k,*v=line.split();header[k]=v
 fields=header["FIELDS"];sizes=list(map(int,header["SIZE"]));types=header["TYPE"]
 counts=list(map(int,header.get("COUNT",["1"]*len(fields))))
 if not len(fields)==len(sizes)==len(types)==len(counts) or len(set(fields))!=len(fields):raise ValueError("field schema")
 if any(c!=1 for c in counts):raise ValueError("scalar fields only")
 n=int(header["POINTS"][0])
 if n!=int(header["WIDTH"][0])*int(header["HEIGHT"][0]) or n<0:raise ValueError("point count")
 body=data[match.end():]
 if len(body)<8:raise ValueError("PCD lengths")
 compressed,expanded=struct.unpack("<II",body[:8])
 if expanded!=n*sum(sizes) or len(body)!=8+compressed:raise ValueError("PCD size mismatch")
 raw=lzf_decode(body[8:],expanded);offset=0;out={}
 for name,size,kind in zip(fields,sizes,types):
  if (kind,size) not in {("F",4),("F",8),("I",1),("I",2),("I",4),("U",1),("U",2),("U",4)}:raise ValueError("PCD type")
  dt=np.dtype("<"+{"F":"f","I":"i","U":"u"}[kind]+str(size))
  out[name]=np.frombuffer(raw,dtype=dt,count=n,offset=offset).copy();offset+=n*size
 return out
def time_ns(seconds,nanos):
 s=int(seconds);n=int(nanos)
 if not 0<=n<1_000_000_000:raise ValueError("invalid nanos")
 return s*1_000_000_000+n
def nearest_reference(query,times,max_gap):
 t=np.asarray(times,dtype=np.int64)
 if t.ndim!=1 or np.any(np.diff(t)<=0):raise ValueError("reference must strictly increase")
 if len(t)==0:return None
 i=int(np.searchsorted(t,query))
 candidates=[j for j in [i-1,i] if 0<=j<len(t)]
 best=min(candidates,key=lambda j:(abs(int(t[j])-int(query)),j))
 gap=abs(int(t[best])-int(query))
 return (best,gap) if gap<=max_gap else None
