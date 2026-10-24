"""Strict SMS legacy target-list decoder, based on pinned UniBwTAS BSD-3 driver.
Upstream license retained under vendor/LICENSE. No CAN or INS values are used.
"""
import binascii,math,struct
from collections import Counter
def parse_targets(payload):
    if len(payload)<24: raise ValueError("Short port header")
    port,major,minor,stamp,size,endian,index,hmajor,hminor=struct.unpack_from(">IHHQI4B",payload)
    if port!=66: return None
    if (major,minor) not in ((1,0),(2,0),(2,1)) or hmajor!=2 or hminor>2 or endian not in (1,2):
        raise ValueError("Unsupported target-list format")
    if size>len(payload) or size<32: raise ValueError("Invalid announced port size")
    fmt=">" if endian==1 else "<"
    cycle,count,acq=struct.unpack_from(fmt+"fHH",payload,24)
    if count>110 or size!=32+56*count: raise ValueError("Invalid target count/size")
    rows=[]
    for i in range(count):
        start=32+56*i
        row=[struct.unpack_from(fmt+"f",payload,start+k)[0] for k in (0,4,8,12,32,44,48)]
        if not all(math.isfinite(v) for v in row): raise ValueError("Nonfinite target")
        rows.append(row)
    return dict(targets=rows,cycle_seconds=cycle,sensor_clock=stamp,port_version=[major,minor])
class Decoder:
    def __init__(self):
        self.pool={};self.counts=Counter()
    def feed(self,payload,sensor,header_ns,receipt_ns):
        data=bytes(payload)
        if len(data)<12 or data[:2]!=b"\x7e\x01": raise ValueError("Invalid SMS prefix")
        length=data[2];size=struct.unpack_from(">H",data,3)[0];app=data[5]
        flags=struct.unpack_from(">I",data,6)[0]
        if flags & ~109: raise ValueError("Unsupported transport flags")
        mc=bool(flags&1);skip=bool(flags&4);source=bool(flags&8);identifier=bool(flags&32);seg=bool(flags&64)
        expected=12+2*mc+4*source+2*identifier+2*seg
        if length!=expected or len(data)!=length+size+2*(not skip): raise ValueError("Transport length mismatch")
        if binascii.crc_hqx(data[:length-2],0xffff)!=struct.unpack_from(">H",data,length-2)[0]:
            raise ValueError("Transport header CRC mismatch")
        if not skip and binascii.crc_hqx(data[length:length+size],0xffff)!=struct.unpack_from(">H",data,length+size)[0]:
            raise ValueError("Transport payload CRC mismatch")
        if app!=8:
            self.counts["non_port"]+=1;return None
        offset=10;part=0;source_id=sensor;data_id=0;total=1
        if mc: part=struct.unpack_from(">H",data,offset)[0];offset+=2
        if source: source_id=struct.unpack_from(">I",data,offset)[0];offset+=4
        if identifier: data_id=struct.unpack_from(">H",data,offset)[0];offset+=2
        if seg: total=struct.unpack_from(">H",data,offset)[0]
        chunk=data[length:length+size]
        available=max(receipt_ns,header_ns)
        if seg:
            if not mc or not identifier or not 1<=total<=16 or part>=total: raise ValueError("Invalid segmentation")
            key=(sensor,source_id,data_id,total)
            if part==0:
                if key in self.pool: self.counts["discarded_incomplete"]+=1
                self.pool[key]=dict(chunks=[],next=0,raw_header_ns=header_ns,available_ns=available)
            entry=self.pool.get(key)
            if entry is None:
                self.counts["orphan_fragment"]+=1;return None
            if part!=entry["next"] or not 0<=header_ns-entry["raw_header_ns"]<=10_000_000:
                self.pool.pop(key,None);self.counts["invalid_fragment_sequence"]+=1;return None
            entry["chunks"].append(chunk);entry["next"]+=1;entry["available_ns"]=max(entry["available_ns"],available)
            if part+1<total: return None
            joined=b"".join(entry["chunks"]);self.pool.pop(key)
            first=entry["raw_header_ns"];available=entry["available_ns"]
        else:
            joined=chunk;first=header_ns
        decoded=parse_targets(joined)
        if decoded is None:
            self.counts["non_target_port"]+=1;return None
        decoded.update(sensor=sensor,raw_header_ns=first,available_ns=available,
            payload_sha256=__import__("hashlib").sha256(joined).hexdigest())
        self.counts["complete_target_frames"]+=1
        return decoded
