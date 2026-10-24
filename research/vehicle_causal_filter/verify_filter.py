"""Source integrity, complete replay and independent scalar Kalman verification."""
import io,unittest
import numpy as np
from filter_core import ROOT,read,digest,freeze
from run_filter import compute,sources,noise
from verify_continuous import close
import test_filter
def scalar(times,z,source_quality,peer,r):
 out=np.full(len(z),np.nan);keep=np.zeros(len(z),dtype=bool);v=a=p00=p01=p11=last=None
 for i,tick in enumerate(times):
  t=int(tick)
  if not source_quality[i]:v=a=p00=p01=p11=last=None;continue
  if last is None or t-last>100000000:v=float(z[i]);a=0.;p00=r;p01=0.;p11=1.
  else:
   dt=(t-last)/1e9;v=v+dt*a;aa=p00+2*dt*p01+dt*dt*p11+dt**3/3;bb=p01+dt*p11+dt*dt/2;cc=p11+dt;s=aa+r;k0=aa/s;k1=bb/s;innovation=z[i]-v;v=v+k0*innovation;a=a+k1*innovation;p00=aa-aa*aa/s;p01=bb-aa*bb/s;p11=cc-bb*bb/s
  out[i]=max(0.,v);keep[i]=abs(out[i]-peer[i])<=1.;last=t
  if not keep[i]:v=a=p00=p01=p11=last=None
 return out,keep
def audit():
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or digest(p)!=h:raise ValueError('filter study changed: '+name)
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('prior evidence changed: '+name)
 freeze();result,arrays=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'filter_replay.npz',allow_pickle=False) as archived:
  if set(archived.files)!=set(arrays):raise ValueError('filter replay keys differ')
  for name,value in arrays.items():
   if value.dtype.kind=='f':np.testing.assert_allclose(value,archived[name],rtol=1e-7,atol=1e-7,equal_nan=True)
   else:np.testing.assert_array_equal(value,archived[name])
 streams=sources();variance=noise()['variance_mps2'];scalar_points=0
 for name,s in streams.items():
  times=np.array([p['time_ns'] for p in s['index']],dtype=np.int64);out,quality=scalar(times,s['pred'],s['quality'],s['peer'],variance);np.testing.assert_allclose(out,arrays[name+'__filter__estimate_mps'],rtol=0,atol=1e-9,equal_nan=True);np.testing.assert_array_equal(quality,arrays[name+'__filter__quality']);scalar_points+=len(times)
  stats=result['methods'][name];overall=stats['filtered_requests']['overall']
  if sum(overall['status_counts'].values())!=37340 or sum(x['requests'] for x in stats['filtered_requests']['segments_300s'].values())!=37340:raise ValueError('scheduled request costs lost')
  if np.any(quality&~s['quality']):raise ValueError('rejected source revived')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_filter))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'CAUSAL FILTER AUDIT: PASS ({tests.testsRun} tests;{scalar_points} scalar filter steps;no neural refit)')
 for name,stats in result['methods'].items():
  q=stats['filtered_requests']['overall'];print(f"FILTER {name}: {100*q['usable_fraction']:.2f}% requests usable; accepted P95 {q['accepted']['p95_mps']:.4f}m/s; accepted recorded errors>2m/s {q['accepted']['above_2_mps']}.")
 print('CAUSAL FILTER DECISION:',result['decision']);print('COVARIANCE IS NOT A PHYSICAL ERROR BOUND. FINAL TEST UNOPENED. WITNESS DISABLED.')
if __name__=='__main__':audit()
