from pathlib import Path
import csv, hashlib, json, io
root=Path('D:/CO-ACT-CPS/phase2_joint_history')
out=root/'research/vehicle_comparison_scope_audit'
sources=[
'research/vehicle_matched_evidence/published_method_registry.json',
'research/vehicle_matched_evidence/baseline_accuracy.csv',
'research/vehicle_matched_evidence/FINDINGS.md',
'research/vehicle_matched_evidence/query_ablation.csv',
'research/vehicle_matched_evidence/attack_ablation.csv',
'research/vehicle_neural_adaptation/matched_metrics.csv',
'research/vehicle_neural_adaptation/FINDINGS.md',
'research/vehicle_neural_baseline/EVALUATION_SCOPE.json',
'research/vehicle_inertial_consistency/FINDINGS.md',
'research/vehicle_reference_qualification/reference_registry.json',
'research/vehicle_reference_qualification/REFERENCE_EVIDENCE_REQUIREMENTS.json',
'research/vehicle_development_release/FINAL_READINESS.json',
'research/vehicle_development_release/NUMERICAL_CORRECTION.json',
'research/vehicle_evidence_freeze/FINAL_VALIDATION_REQUIREMENTS.json']
def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
before={s:sha(root/s) for s in sources}
extra=json.loads((out/'scope_additions.json').read_text(encoding='utf-8'))
prior=json.loads((root/sources[0]).read_text(encoding='utf-8'))
registry={'review_date':extra['review_date'],'scope':extra['scope'],'inherited_registry':sources[0],'inherited_records_not_newly_fulltext_reviewed':prior['records'],'additional_records':extra['additional_records'],'decision':extra['decision']}
(out/'method_scope.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
def readrows(rel):
 with (root/rel).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
classic=readrows(sources[1])
neural=readrows(sources[5])
assert len(classic)==6
groups=('Library01_temporal_test','Library02_development_transfer')
selected=[x for x in neural if x['group'] in groups]
assert len(selected)==14
rows=[]
for row in classic:
 rows.append({'cohort':'sparse_classical_development','group':row['sequence'],'source_file':sources[1],**{k:v for k,v in row.items() if k!='sequence'}})
for row in selected:
 rows.append({'cohort':'small_data_neural_development','source_file':sources[5],**row})
fields=['cohort','group','source_file','method','model','attempts','admitted','invalid_prediction','availability','scored','unknown_reference','mae_mps','p95_mps','max_mps','above2']
with (out/'matched_comparisons.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
claims=[
('motion_component_comparison','bounded_supported','Lower recorded MAE on the specified classical development cohort','Architecture superiority with equal information',sources[1]+';'+sources[2],'Candidate has additional IMU attitude; reference is a proxy'),
('adapted_neural_comparison','bounded_supported','Lower matched MAE than each of three fixed small-data neural seeds','Native neural-paper or general neural superiority',sources[5]+';'+sources[6],'Adaptation; small training cohort; all seeds retained'),
('temporal_accuracy','condition_failed','Candidate temporal P95 exceeds 0.5 m/s','All development accuracy conditions pass',sources[5],'61 scores; original failed full-query window remains separate'),
('agreement_guard','bounded_supported','Fixed interface single-source disagreement rejection with coverage','Improved benign estimate or coordinated-compromise protection',sources[3]+';'+sources[4],'Same radar value on common admissions; both-source bias passes'),
('inertial_consistency','bounded_supported','Differential causal acceleration consistency diagnostics','Independent absolute-speed anchor',sources[8],'Constant and slow coordinated offsets pass'),
('physical_accuracy','unestablished','Reference disagreement and hypothetical conditional bounds','Certified physical error bound',sources[9]+';'+sources[10],'Reference uncertainty, clocks, output point and calibration unresolved'),
('broad_zt_novelty','unsupported','Implementation of a combined conditional access-policy study','First multisource context-aware vehicle zero-trust method',sources[0],'Additional overlap recorded in method_scope.json; targeted screen only'),
('security_framework_ranking','unestablished','Component and local policy-control comparisons','Superior to full published AV security frameworks',sources[0]+';'+sources[7],'No common full-system inputs or native framework run'),
('reference_bound_arithmetic','bounded_supported','Exact arithmetic for stated hypothetical reference budgets','Measured physical reference uncertainty',sources[12],'Stored-input calculation correctness is separate from physical evidence'),
('final_confirmation','blocked','Development evidence package with known failures','Untouched final confirmation or journal readiness',sources[11]+';'+sources[13],'Original stop rules and reserved groups unchanged')]
with (out/'claim_evidence.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f); w.writerow(['claim_id','status','supported_scope','unsupported_extension','local_evidence','limitation']); w.writerows(claims)
(out/'source_hashes.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf-8')
# Read the saved tables back; compare every source value and all identifiers.
with (out/'matched_comparisons.csv').open(newline='',encoding='utf-8') as f: saved=list(csv.DictReader(f))
assert len(saved)==20
for actual,expected in zip(saved,rows):
 for key,value in expected.items(): assert actual[key]==value,(key,actual[key],value)
for row in saved:
 assert int(row['scored'])+int(row['unknown_reference'])==int(row['admitted'])
 assert int(row['admitted'])+int(row['invalid_prediction'])==int(row['attempts'])
assert len({(x['cohort'],x['group'],x.get('method') or x.get('model')) for x in rows})==20
for group in groups:
 candidates=[x for x in saved if x['cohort']=='small_data_neural_development' and x['group']==group]
 assert {x['model'] for x in candidates}=={'candidate','author_3d_ransac','ols_3d','train_median','neural_seed_17','neural_seed_29','neural_seed_43'}
for group in ('Library01','Library02'):
 c=[x for x in saved if x['cohort']=='sparse_classical_development' and x['group']==group]
 assert len(c)==3 and all(x['unknown_reference']=='1' for x in c)
temporal=[x for x in saved if x['group']=='Library01_temporal_test' and x['model'] in ('candidate','author_3d_ransac')]
assert len(temporal)==2 and all(float(x['p95_mps'])>0.5 for x in temporal)
assert before=={s:sha(root/s) for s in sources}
assert registry['decision']['specific_algorithmic_novelty_established'] is False
assert all(x['native_framework_reproduced'] is False for x in registry['additional_records'])
validation={'status':'passed','copied_rows_checked':20,'claim_rows':10,'unchanged_prior_inputs':len(before),'new_primary_source_records':2,'new_experiments_run':False,'novelty_established':False,'journal_readiness_established':False,'scope':'Report/table consistency only; no raw-data or estimator rerun'}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
manifest={x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file() and x.name!='file_manifest.json'}
(out/'file_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
assert all(sha(out/name)==digest for name,digest in manifest.items())
print(json.dumps({**validation,'package_files_checked':len(manifest),'path':str(out)},indent=2))
