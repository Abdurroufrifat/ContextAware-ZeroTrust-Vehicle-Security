from magnitude_core import *
def compute():
 freeze();check_models();data,members,groups,day,support,median_abs=load_data();base=original_base(data);pred={name:np.full(len(groups),np.nan) for name in VARIANTS};assignment=np.full(len(groups),-1,dtype=np.int16)
 for fold,held in enumerate(r.GROUPS):
  calibrations=read(ROOT/'folds'/held/'calibrations.json')
  for variant in VARIANTS:
   cal=calibrations[variant];ids=np.array(cal['held_ids']);median=(data['median_feature'] if variant==VARIANTS[0] else median_abs)[ids];anchor=c.e.affine(cal['anchor'],median);features=assemble(base[ids],anchor,median);model=read(ROOT/'folds'/held/(variant+'.json'))
   if model['feature_names']!=feature_names(variant) or not np.all(groups[ids]==held):raise ValueError('model schema/held population differs')
   pred[variant][ids]=c.predict(model,features,anchor)
  ids=np.array(calibrations[VARIANTS[0]]['held_ids'])
  if np.any(assignment[ids]!=-1):raise ValueError('duplicate held assignment')
  assignment[ids]=fold
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['target_mps'],data['y']) or not np.array_equal(z['fold_id'],assignment):raise ValueError('original control rows differ')
  pred['original_context']=z['prediction_mps'].copy()
 result=summarize(members,data['y'],pred,assignment,support);arrays={name+'__prediction_mps':v for name,v in pred.items()};arrays.update({'target_mps':data['y'],'fold_id':assignment});return result,arrays
def score():
 if (ROOT/'results.json').exists() or (ROOT/'oof_predictions.npz').exists():raise ValueError('magnitude held outcomes already exist; no result-driven retries')
 result,arrays=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'oof_predictions.npz',**arrays)
 for name,s in result['pooled']['models'].items():print(name,'P95',s['p95_mps'],'RMSE',s['rmse_mps'],'maximum',s['maximum_mps'],'failed windows',len(result['diagnostic_failures'][name]['windows']),flush=True)
 print(result['decision'],flush=True)
if __name__=='__main__':score()
