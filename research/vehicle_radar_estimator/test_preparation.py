"""Regression: radar features do not inherit OBD/GNSS availability."""
import gzip, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import make_dataset as m

class RadarPreparationTests(unittest.TestCase):
 def test_unavailable_or_manipulated_claims_cannot_change_radar_features(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);folder=root/'inputs'/'fit';folder.mkdir(parents=True)
   def write(name,value):(folder/name).write_text(json.dumps(value),encoding='utf-8')
   write('windows.json',[{'id':'fit_0','start_ns':0}])
   label={'role':'fit','window_id':'fit_0','time_ns':6500000000,'reference_horizontal_mps':3.,'label_mps':None,'gnss_mps':None,'feature_signed':None,'prediction_mps':None,'valid':False}
   write('labels.json',[label])
   with gzip.open(folder/'packets.jsonl.gz','wt',encoding='utf-8') as file:
    for i in range(131):file.write(json.dumps({'file':'radars.bag','data':'','time_ns':i*50000000})+'\n')
   def decode(kind,data,t):return {'bag_time_ns':t,'header_time_ns':t,'speed_raw':-2.}
   with patch.object(m,'ROOT',root),patch.object(m.c,'decode_message',side_effect=decode):
    a,index=m.build('fit',False);self.assertTrue(index[0]['ready'])
    for key in ('label_mps','gnss_mps','feature_signed','prediction_mps'):label[key]=99999.
    label['valid']=True;write('labels.json',[label]);b,index2=m.build('fit',False)
   self.assertEqual(index,index2)
   for key in a:np.testing.assert_array_equal(a[key],b[key])
   self.assertEqual(a['median_feature'][0],2.)
if __name__=='__main__':unittest.main()
