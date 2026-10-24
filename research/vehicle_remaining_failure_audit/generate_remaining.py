from remaining_core import *
def generate():
 freeze()
 if any((ROOT/n).exists() for n in ['results.json','row_diagnostics.npz','error_runs.json']):raise ValueError('diagnosis already generated; no overwrite')
 rows=load_rows();result,records=build(rows);write(ROOT/'results.json',result);write(ROOT/'error_runs.json',records);np.savez_compressed(ROOT/'row_diagnostics.npz',**{scope+'__'+key:value for scope,q in rows.items() for key,value in q.items()});print(result['decision']);print('Remaining fit windows',result['remaining_failed_fit_windows']);print('Remaining development blocks',result['remaining_failed_old_development_blocks'])
if __name__=='__main__':generate()
