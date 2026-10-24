from axis_core import *
def train():
 freeze();values=cache()
 if (ROOT/'heads').exists() or (ROOT/'head_lock.json').exists():raise ValueError('correction fitting already started; no retry')
 data,*_=m.r.load_data();(ROOT/'heads').mkdir();runs=[];files={}
 for scope in SCOPES:
  x,y,w,ids,native=training_inputs(scope,data,values['fit_axes']);head=fit_head(x,y,w);folder=ROOT/'heads'/scope;folder.mkdir();write(folder/'head.json',head);pred=head_predict(head,x);np.savez_compressed(folder/'training_check.npz',correction=pred,parent_native=native);runs.append({'scope':scope,'training_ids':ids.tolist(),'training_examples':len(x),'feature_sha256':m.g.array_digest(x),'residual_sha256':m.g.array_digest(y),'weight_sha256':m.g.array_digest(w),'parent_native_sha256':m.g.array_digest(native),'alpha':ALPHA,'feature_count':6})
  for name in ['head.json','training_check.npz']:files[f'heads/{scope}/{name}']=digest(folder/name)
  print('SIGNED-AXIS HEAD FIT ONCE',scope,len(x),'examples',flush=True)
 write(ROOT/'training_trace.json',{'runs':runs,'model':'Four fixed alpha=1 standardized weighted ridge residual heads; seven regression coefficients plus fit-only means/scales each','numpy':np.__version__});write(ROOT/'head_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'data_lock_sha256':digest(ROOT/'data_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_four_locked_before_new_scoring':True});print('ALL FOUR HEADS LOCKED BEFORE NEW SCORING',flush=True)
if __name__=='__main__':train()
