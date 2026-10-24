"""Describe missing uncertainty and timestamp metadata without changing scoring."""
import json
import run_reference as run
def compute_findings():
 result={}
 for role in run.read('protocol.json')['recordings']:
  decoded,_=run.decode_bag(role);x=decoded['INSPVAX'];fields={}
  for k in ('north_velocity_stdev','east_velocity_stdev','up_velocity_stdev'):
   bad=[q for q in x if q[k] is None];negative=[q for q in x if q[k] is not None and q[k]<0]
   fields[k]={'rows':len(x),'missing_nonfinite':len(bad),'negative':len(negative),'first_missing_header_time_ns':bad[0]['header_time_ns'] if bad else None}
  result[role]={'velocity_uncertainty_fields':fields,'unknown_gps_time_status_20_messages':sum(q['time_status']==20 for values in decoded.values() for q in values),'all_bag_times_equal_header_times':all(q['bag_time_ns']==q['header_time_ns'] for values in decoded.values() for q in values),'uncertainty_imputed':False,'uncertainty_used_as_uniform_bound':False,'interpretation':'Post-processed writer timestamps are offline alignment metadata. Missing uncertainty is not filled or used to remove otherwise usable primary velocity samples.'}
 return result
if __name__=='__main__':
 result=compute_findings();(run.ROOT/'quality_findings.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{'missing_north_sigma':v['velocity_uncertainty_fields']['north_velocity_stdev']['missing_nonfinite'],'unknown_time_status':v['unknown_gps_time_status_20_messages']} for k,v in result.items()},indent=2))
