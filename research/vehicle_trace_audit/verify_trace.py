"""Full fitting-source replay plus independent payload/metric checks."""
import io as string_io,unittest
from trace_core import *
from verify_diagnosis import close,independent_metric
import test_trace

def compare_probes(actual,saved):
 def canonical(records):
  if not isinstance(records,list) or any(not isinstance(q,dict) or not isinstance(q.get('source'),str) for q in records):raise ValueError('invalid source probe metadata')
  return [{**q,'source':q['source'].replace('\\','/')} for q in records]
 close(canonical(actual),canonical(saved))

def independent_stats(values):
 values=list(map(float,values));a=sorted(x for x in values if math.isfinite(x));n=len(a)
 def quantile(q):
  z=(n-1)*q;lo=int(math.floor(z));hi=int(math.ceil(z));return a[lo]+(z-lo)*(a[hi]-a[lo])
 return {'total':len(values),'finite':n,'missing':len(values)-n,'min':a[0] if n else None,'median':quantile(.5) if n else None,'p95':quantile(.95) if n else None,'max':a[-1] if n else None}

def audit():
 freeze()
 for name,h in read(ROOT/'sha256.json').items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(ROOT) or digest(f)!=h:raise ValueError('trace audit payload changed: '+name)
 result,traces,probes=compute();close(result,read(ROOT/'results.json'));compare_probes(probes,read(ROOT/'decode_probes.json'))
 with np.load(ROOT/'row_traces.npz',allow_pickle=False) as z:
  if set(z.files)!=set(traces):raise ValueError('trace row schema changed')
  for key,value in traces.items():
   if not np.array_equal(value,z[key],equal_nan=True):raise ValueError('source trace differs: '+key)
 for q in probes:
  b=base64.b64decode(q['payload_base64'],validate=True);close(independent_message(q['message_kind'],b),q['expected_fields'])
 members=read(RESEARCH/'vehicle_route_validation/membership.json');parent=read(RESEARCH/'vehicle_failure_diagnosis/results.json');window=np.array([m['window_id'] for m in members])
 with np.load(RESEARCH/'vehicle_matched_baselines/oof_predictions.npz',allow_pickle=False) as z:y=z['target_mps'];prediction=z['context_boosting__prediction_mps']
 speed=d.speed_bins(y);mixed=traces['mixed_sign'];age=traces['reference_age_bin'];cells=[(np.ones(len(y),dtype=bool),result['pooled'])];cells.extend((speed==i,result['by_reference_speed'][name]) for i,name in enumerate(d.SPEED_NAMES));cells.extend((age==k,result['by_reference_age'][str(k)]) for k in range(3));cells.extend((window==name,cell) for name,cell in result['by_window'].items())
 for i,name in enumerate(d.SPEED_NAMES):
  cells.extend([((speed==i)&mixed,result['by_speed_and_sign'][name]['mixed']),((speed==i)&~mixed,result['by_speed_and_sign'][name]['other'])]);cells.extend(((speed==i)&(age==k),result['by_speed_and_reference_age'][name][str(k)]) for k in range(3))
 fields=['reference_age_ns','local_age_proxy_mps','radar_header_age_ns','sign_cancellation_raw']
 for selected,cell in cells:
  if cell['count']!=int(selected.sum()):raise ValueError('trace stratum denominator differs')
  close(cell['hgb'],independent_metric(y[selected],prediction[selected]))
  for key in fields:close(cell[key],independent_stats(traces[key][selected]))
  if cell['mixed_sign_rows']!=int(mixed[selected].sum()) or cell['positive_cancellation_rows_above_numeric_tolerance']!=int(np.sum(traces['sign_cancellation_raw'][selected]>1e-6)):raise ValueError('sign diagnostic differs')
 close(result['pooled']['hgb'],parent['pooled']['models']['context_boosting'])
 for name,cell in result['by_reference_speed'].items():close(cell['hgb'],parent['by_speed'][name]['models']['context_boosting'])
 for name,cell in result['by_window'].items():close(cell['hgb'],parent['by_window'][name]['models']['context_boosting'])
 day=read(RESEARCH/'vehicle_radar_multidrive/inputs/fit/windows.json');night=read(RESEARCH/'vehicle_night_training/data_report.json')['windows'];expected={q['id']:q['raw_radar_in_block'] for q in day};expected.update({q['id']:q['raw'] for q in night});close(expected,result['raw_window_endpoint_counts'])
 if np.any(traces['reference_age_ns']<0) or np.any(traces['reference_age_ns']>100_000_000) or np.any(traces['radar_header_age_ns']<0) or np.any(traces['radar_header_age_ns']>250_000_000):raise ValueError('supported source age budget differs')
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_trace);buf=string_io.StringIO();tests=unittest.TextTestRunner(stream=buf).run(suite)
 if not tests.wasSuccessful():raise ValueError(buf.getvalue())
 print(f'TRACE SOURCE AUDIT: PASS ({tests.testsRun} tests;{len(y)} source labels;{result["raw_tensor_values_reproduced"]} radar tensor values;{len(probes)} independent decode probes;{len(cells)} independent error cells;{len(cells)*len(fields)} independent source-stat cells)')
 print('Reference age maximum:',result['pooled']['reference_age_ns']['max']/1e6,'ms; local reference-age proxy P95:',result['pooled']['local_age_proxy_mps']['p95'],'m/s (not a bound)')
 print(result['decision']+'; NO ROOT CAUSE CLAIM. WITNESS DISABLED.')
if __name__=='__main__':audit()
