"""Run once on the existing Windows project; reruns verify existing results."""
import csv,io,math,subprocess,sys
from monitor_core import *
def close(a,b,path=''):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('summary keys '+path)
  for k in a:close(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,float):
  if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=0,abs_tol=1e-9):raise ValueError('summary number '+path)
 elif a!=b:raise ValueError('summary mismatch '+path)

def generate():
 check_bundle()
 if (ROOT/'local_results.json').exists():return
 old,hashes,times,truth,events,masks,previous=prior_inputs()
 if (ROOT/'local_replays').exists():raise ValueError('Incomplete previous run: retain its files and share the error; do not rescore over it.')
 write(ROOT/'local_run_lock.json',{'monitor_lock':read(ROOT/'analysis_lock.json'),'prior_sha256':hashes,'project_manifest_sha256':digest(PROJECT/'PROJECT_MANIFEST.json'),'truth_used_by_monitor':False,'scope':'Reused development; fixed post-selector heuristic, not a source-authentication or independent-witness solution.'})
 results={};csv_rows=[];output_files=['local_run_lock.json']
 for case in CASES:
  saved=load(old/'replays'/(case+'.npz'));result={};arrays={}
  for method in METHODS:
   out=proposal(saved,method,times,events);baseline=summary(out,times,truth,masks);close(baseline,previous['cases'][case][method])
   guard=monitor(times,out['prediction_mps'],out['route']>0);guarded=summary(filtered(out,guard['allowed']),times,truth,masks)
   first=guard['first_alarm'];result[method]={'baseline':baseline,'quarantine':guarded,'monitor':{'first_alarm_index':first,'first_alarm_time_ns':None if first is None else int(times[first]),'additional_withheld_requests':int(np.sum((out['route']>0)&~guard['allowed'])),'gap_reanchors':int(np.sum(guard['reason']==4))}}
   arrays[method+'__allowed']=guard['allowed'];arrays[method+'__reason']=guard['reason']
   for variant,v in [('baseline',baseline),('quarantine',guarded)]:
    for pop,cell in v.items():
     e=cell['accepted_errors'];csv_rows.append({'case':case,'method':method,'variant':variant,'population':pop,'requests':cell['requests'],'accepted_requests':cell['accepted_requests'],'availability':cell['availability'],'p95_mps':e['p95_mps'],'above_2_mps':e['above_2_mps'],'joint_gate_pass':cell['joint_gate_pass']})
  results[case]=result;name='local_replays/'+case+'.npz';write_npz(ROOT/name,arrays);output_files.append(name)
  v=result['extra_trees7__deadband__fallback']['quarantine']['during'];print(case,'quarantine ET during:',v['accepted_requests'],'/',v['requests'],'P95',v['accepted_errors']['p95_mps'],'>2m/s',v['accepted_errors']['above_2_mps'],flush=True)
 buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows);atomic(ROOT/'local_comparison.csv',buffer.getvalue().encode());output_files.append('local_comparison.csv')
 write(ROOT/'local_results.json',{'execution_status':'QUARANTINE_COMPARISON_COMPLETE_NO_PROMOTION','cases':results,'synthetic_counterexamples':fixture_results(),'models_refitted':False,'labels_changed':False,'source_policy_changed':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'independent_physical_witness_established':False});output_files.append('local_results.json')
 write(ROOT/'local_outputs_sha256.json',{name:digest(ROOT/name) for name in output_files})
 print('QUARANTINE_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)

def main():
 generate();subprocess.run([sys.executable,str(ROOT/'verify_monitor.py')],check=True)
if __name__=='__main__':main()
