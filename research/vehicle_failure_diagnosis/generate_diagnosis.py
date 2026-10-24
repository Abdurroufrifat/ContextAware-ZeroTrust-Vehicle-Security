"""Create one descriptive diagnostic from previously inspected frozen outputs."""
from diagnosis_core import *
def generate():
 freeze()
 if any((ROOT/n).exists() for n in ['results.json','row_assignments.npz','error_runs.json']):raise ValueError('diagnostic outputs already exist; no selective regeneration')
 members,y,p,support=load_inputs();result,arrays,runs=build(members,y,p);result['raw_support_costs']=support
 write(ROOT/'results.json',result);write(ROOT/'error_runs.json',runs);np.savez_compressed(ROOT/'row_assignments.npz',**arrays)
 print(result['decision']);print('All',len(y),'supported rows retained;',len(result['failed_hgb_windows']),'failed windows')
 for name,cell in result['by_speed'].items():
  v=cell['models']['context_boosting'];print('SPEED',name,'rows',cell['count'],'P95',v['p95_mps'],'above0.5',v['above_half_mps'])
if __name__=='__main__':generate()
