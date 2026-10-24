import io, json, struct, unittest, zlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import transfer_core as m


class TransferTests(unittest.TestCase):
    def test_reserved_before_open(self):
        for name in ('RAV4/data', 'b0c9d2329ad1606b/x', 'Urban02_NIGHT/a', 'vehicle_final_evaluation/a'):
            with self.assertRaises(ValueError): m.safe(Path(name))

    def test_only_declared_civic_member(self):
        for name in ('Chunk_3/evil/0/x','../x','Chunk_3/99c94dc769b5d96e|x/0/video.hevc'):
            with self.assertRaises(ValueError): m.validate_member(name)

    def test_causal_alignment_boundary(self):
        rows, audit=m.align(np.array([1.,1.1,1.100001,1.2]),np.array([[0,0,10,0,0,0]]*4),
                           np.array([1.,1.2]),np.array([[9,10,11,12],[20,20,20,20]]))
        self.assertEqual([r['wheel_speed_mps'] for r in rows],[10.5,10.5,None,20.])
        self.assertEqual(rows[2]['wheel_utime'],1000000)
        self.assertEqual(audit['stale_or_missing_wheel_rows'],1)

    def test_future_wheel_never_used(self):
        rows,_=m.align(np.array([1.]),np.array([[0,0,2,0,0,0]]),np.array([1.01]),np.ones((1,4)))
        self.assertIsNone(rows[0]['wheel_speed_mps'])

    def test_invalid_values_preserve_reference_denominator(self):
        rows,_=m.align(np.array([1.,2.]),np.array([[0,0,-1,0,0,0],[0,0,float('nan'),0,0,0]]),
                       np.array([1.,2.]),np.array([[1,2,3,float('nan')],[-1,2,3,4]]))
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['pose_speed_mps'] is None and r['wheel_speed_mps'] is None for r in rows))

    def test_no_clock_sorting_or_duplicate_repair(self):
        for t in (np.array([2.,1.]),np.array([1.,1.]),np.array([1.,float('nan')])):
            with self.assertRaises(ValueError): m.align(t,np.ones((2,6)),np.array([1.]),np.ones((1,4)))

    def test_shape_contract(self):
        with self.assertRaises(ValueError): m.align(np.array([1.]),np.ones((1,2)),np.array([1.]),np.ones((1,4)))

    def test_range_requires_partial_and_exact_length(self):
        class Response(io.BytesIO):
            status=200
            headers={}
        with patch('urllib.request.urlopen',return_value=Response(b'abc')):
            with self.assertRaises(ValueError):m.range_get('https://example.test',0,2,100)

    def test_crc_and_decompression_bound(self):
        b=b'example';e={'size':7,'crc':zlib.crc32(b),'method':8}
        compressed=zlib.compress(b)[2:-4]
        self.assertEqual(m.decode(e,compressed),b)
        for change in ({'crc':0},{'size':6}):
            with self.assertRaises(ValueError):m.decode(dict(e,**change),compressed)

    def test_pickle_rejected(self):
        b=io.BytesIO();np.save(b,np.array([object()],dtype=object))
        with self.assertRaises(ValueError):m.load_array(b.getvalue())

    def test_fixed_selection_no_day_leak(self):
        d=json.loads((Path(__file__).parent/'selection.json').read_text())
        dev={r['day'] for r in d['recordings'] if r['split']=='development'}
        ev={r['day'] for r in d['recordings'] if r['split']=='evaluation'}
        self.assertFalse(dev&ev);self.assertEqual(len(d['recordings']),21)
        self.assertEqual(sum(r['split']=='evaluation' for r in d['recordings']),18)

if __name__=='__main__':unittest.main()
