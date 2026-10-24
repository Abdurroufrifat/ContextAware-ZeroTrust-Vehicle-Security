import binascii,struct,unittest
from radar_decoder import Decoder,parse_targets
def packet(payload,part=0,count=1,data_id=100):
    flags=1|4|8|32|64
    head=struct.pack(">BBBHB I H I H H",126,1,22,len(payload),8,flags,part,123,data_id,count)
    return head+struct.pack(">H",binascii.crc_hqx(head,0xffff))+payload
def target_port():
    body=struct.pack(">fHH",.055,1,0)
    record=bytearray(56)
    for offset,value in [(0,10.),(4,-3.),(8,.1),(12,.02),(32,1.),(44,20.),(48,10.)]: struct.pack_into(">f",record,offset,value)
    total=24+8+56
    head=struct.pack(">IHHQI4B",66,1,0,1,total,1,0,2,2)
    return head+body+record
class DecoderTests(unittest.TestCase):
    def test_fragmented_target_decodes(self):
        data=target_port();decoder=Decoder()
        self.assertIsNone(decoder.feed(packet(data[:40],0,2),35,100,101))
        result=decoder.feed(packet(data[40:],1,2),35,102,103)
        self.assertEqual(result["targets"][0][0],10.)
        self.assertEqual(result["targets"][0][1],-3.)
        self.assertEqual(result["available_ns"],103)
        self.assertEqual(result["raw_header_ns"],100)
    def test_bad_crc_rejected(self):
        data=bytearray(packet(target_port()));data[16]^=1
        with self.assertRaises(ValueError): Decoder().feed(bytes(data),35,100,101)
    def test_incomplete_and_stale_fragments_not_used(self):
        decoder=Decoder();data=target_port()
        self.assertIsNone(decoder.feed(packet(data[:40],0,2),35,0,0))
        self.assertIsNone(decoder.feed(packet(data[40:],1,2),35,20_000_000,20_000_000))
    def test_out_of_order_and_bad_port_sizes_rejected(self):
        self.assertIsNone(Decoder().feed(packet(target_port()[40:],1,2),35,100,100))
        with self.assertRaises(ValueError): parse_targets(target_port()[:-1])
    def test_big_endian_values_match_struct(self):
        result=parse_targets(target_port())
        self.assertAlmostEqual(result["targets"][0][2],.1,places=6)
if __name__=="__main__": unittest.main()
