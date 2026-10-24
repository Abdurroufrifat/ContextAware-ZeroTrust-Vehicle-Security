"""Fixed-record optical representation; integrity digest is not authentication."""
import struct,hashlib,hmac,numpy as np
HEADER=struct.Struct("<8sQQI32s");MAGIC=b"COACTOP1"
SLOT=HEADER.size+512*29+32
def encode(b,stamp_ns,source_sha256):
 if len(b)%29 or len(b)>32*1024**2:raise ValueError("invalid raw record length")
 if not isinstance(stamp_ns,int) or not 0<=stamp_ns<2**64:raise ValueError("invalid original stamp")
 try:identity=bytes.fromhex(source_sha256)
 except (TypeError,ValueError) as ex:raise ValueError("invalid source identity") from ex
 if len(identity)!=32:raise ValueError("invalid source identity")
 count=len(b)//29;take=min(count,512)
 idx=np.linspace(0,count-1,take,dtype=int) if take else np.empty(0,dtype=int)
 payload=np.frombuffer(b,dtype="V29")[idx].tobytes()
 body=HEADER.pack(MAGIC,stamp_ns,count,take,identity)+payload
 return body+hashlib.sha256(body).digest()
def decode(packet):
 if not HEADER.size+32<=len(packet)<=SLOT:raise ValueError("invalid packet length")
 body=packet[:-32]
 if not hmac.compare_digest(hashlib.sha256(body).digest(),packet[-32:]):raise ValueError("packet integrity failed")
 magic,stamp,count,take,identity=HEADER.unpack_from(body)
 if magic!=MAGIC or count*29>32*1024**2 or take!=min(count,512) or len(body)!=HEADER.size+29*take:raise ValueError("invalid packet schema")
 return {"stamp_ns":stamp,"source_count":count,"source_sha256":identity.hex(),"payload":body[HEADER.size:]}
