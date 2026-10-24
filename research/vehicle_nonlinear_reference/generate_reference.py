from reference_core import *
def generate():
 if (ROOT/'results.json').exists() or (ROOT/'row_identity.npz').exists():raise ValueError('already scored; no retry')
 result,arrays=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'row_identity.npz',**arrays)
 for scope in ['fit','full']:
  pooled=result['populations'][scope]['pooled']['all'];print(scope,'parent',pooled['parent_rows'],'matched',pooled['matched_rows'],'missing',pooled['missing_rows'])
 for scope,window,cohort in [('fit','evaluation_urban04_0','all_matched'),('full','calibration_0','nonlinear_accepted_matched')]:
  v=result['populations'][scope]['by_window'][window][cohort]['models']['nonlinear7'];print(scope,window,cohort,'primary P95',v['primary_reference']['p95_mps'],'OBD P95',v['obd']['p95_mps'])
 print(result['decision'])
if __name__=='__main__':generate()
