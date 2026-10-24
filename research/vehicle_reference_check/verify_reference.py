"""Offline source-checksum, selection, matching, metric and qualification audit."""
from pathlib import Path
import contextlib,hashlib,io,json,math,re,unittest
import run_reference as run
from quality_findings import compute_findings
import test_reference
ROOT=Path(__file__).resolve().parent
def read_published_checksums(path):
 text=path.read_text(encoding='utf-8')
 published=dict(re.findall(r'Filename/Fichier: ([^\n]+)\nSHA256 Checksum/Sommes de contrôle: (\w+)',text))
 if not published:raise ValueError('No publisher checksums found in the UTF-8 source file')
 return published
def audit():
 for name,sha in run.read('sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('reference payload changed: '+name)
 protocol=run.read('protocol.json');published=read_published_checksums(ROOT/'source_attribution/checksums.txt')
 for role,cfg in protocol['recordings'].items():
  if cfg['reference_bag_publisher_sha256']!=published[cfg['route']+'/reference.bag']:raise ValueError('captured publisher checksum changed')
  if (ROOT.parent/'vehicle_temporal_policy').is_dir():
   for n,sha in cfg['input_sha256'].items():
    p=ROOT.parent/'vehicle_temporal_policy'/'roles'/role/n
    if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('prior frozen endpoint input mismatch')
 if run.digest('source_attribution/setup_transformations_1.txt')!=published['CalibrationData/setup_transformations_1.txt']:raise ValueError('geometry source checksum mismatch')
 result,paired=run.compute()
 if result!=run.read('results.json') or paired!=run.read('matched_endpoints.json'):raise ValueError('source/reference reconstruction mismatch')
 if compute_findings()!=run.read('quality_findings.json'):raise ValueError('uncertainty quality findings changed')
 if len({(q['role'],q['window_id'],q['time_ns']) for q in paired})!=len(paired):raise ValueError('endpoint join multiplied rows')
 for q in paired:
  if q['reference_header_time_ns'] is not None and not 0<=q['time_ns']-q['reference_header_time_ns']<=100000000:raise ValueError('future or stale offline reference')
 if result['witness_enabled'] or result['production_policy_changed'] or result['live_witness_qualified'] or result['current_guaranteed_detectable_shift_mps'] is not None:raise ValueError('unsupported physical reference promotion')
 if any(q['qualified'] for q in result['qualification_requirements'].values()):raise ValueError('unsupported reference qualification')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_reference))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 for role,metrics in result['recording_metrics'].items():
  values=sorted(abs(q['prediction_mps']-q['reference_horizontal_mps']) for q in paired if q['role']==role and q['reference_horizontal_mps'] is not None and q['prediction_mps'] is not None);z=.95*(len(values)-1);i=int(z);p95=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(z-i)
  if not math.isclose(p95,metrics['metrics']['prediction_mps']['p95_disagreement_mps'],abs_tol=1e-12):raise ValueError('independent P95 calculation mismatch')
 count=sum(sum(v['messages'] for v in p.values()) for p in result['full_reference_source_profiles'].values())
 print(f'REFERENCE SOURCE AND REPLAY INTEGRITY: PASS (4 publisher-verified complete bags; {count} messages; {len(paired)} scored endpoints; {tests.testsRun} tests)')
 print('OFFLINE REFERENCE MATCHING:',str(result['reference_matches'])+'/'+str(len(paired)))
 print('REFERENCE ACCURACY DIAGNOSTIC TARGET:', 'PASS' if result['all_reference_diagnostic_targets_pass'] else 'NOT MET')
 print('LIVE WITNESS QUALIFIED: NO. GUARANTEED DETECTABLE SHIFT: NOT ESTABLISHED.')
 print('WITNESS ENABLED: NO. See research/vehicle_reference_check/REFERENCE_DECISION.md.')
 return result
if __name__=='__main__':audit()
