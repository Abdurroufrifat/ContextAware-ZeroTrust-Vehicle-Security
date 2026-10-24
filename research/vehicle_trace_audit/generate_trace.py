from trace_core import *
def generate():
 if any((ROOT/n).exists() for n in ['results.json','row_traces.npz','decode_probes.json']):raise ValueError('trace results already exist; no selective regeneration')
 result,arrays,probes=compute();write(ROOT/'results.json',result);write(ROOT/'decode_probes.json',probes);np.savez_compressed(ROOT/'row_traces.npz',**arrays)
 print(result['decision']);print('Reproduced source labels:',result['raw_reference_labels_reproduced'],'tensor values:',result['raw_tensor_values_reproduced']);print('REFERENCE AGE NS',result['pooled']['reference_age_ns']);print('LOCAL AGE PROXY MPS',result['pooled']['local_age_proxy_mps'])
 for name,s in result['by_speed_and_sign'].items():print('SPEED/SIGN',name,{k:{'count':c['count'],'p95':c['hgb']['p95_mps'],'cancellation':c['positive_cancellation_rows_above_numeric_tolerance']} for k,c in s.items()})
if __name__=='__main__':generate()
