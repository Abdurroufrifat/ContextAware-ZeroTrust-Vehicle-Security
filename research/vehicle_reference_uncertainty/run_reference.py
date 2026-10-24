"""Qualify the frozen nominal bounded-policy comparison without changing its outputs."""
import argparse,json,sys
from pathlib import Path
from fractions import Fraction
import reference_bounds as bounds
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vehicle_source_conflict_audit'))
import conflict_audit as previous
prior=previous.prior

def freeze():
    return dict(previous_evidence=previous.check_inputs(),
                code_sha256={p.name:prior.io.digest(p) for p in sorted(HERE.glob('*.py'))},
                contract_sha256=prior.io.digest(HERE/'reference_contract.json'),
                source_assessment_sha256=prior.io.digest(HERE/'source_assessment.json'))

def run(output):
    output=prior.io.safe(output)
    if output==HERE or output in HERE.parents or output==previous.EXTERNAL or previous.EXTERNAL in output.parents:
        raise ValueError('Output overlaps source/evidence')
    frozen=freeze();output.mkdir(parents=True,exist_ok=True)
    previous.immutable_write(output/'input_lock.json',previous.json_bytes(frozen))
    spec=json.loads((HERE/'reference_contract.json').read_text())
    # This run is a retrospective qualification of the current unknown reference.
    # Supplying a new bound requires a separate experiment and its evidence appraisal.
    prior.equal(spec['numeric_bound_status'],'unestablished','current_reference_status')
    prior.equal(spec['instantaneous_speed_absolute_bound_mps'],None,'unknown_is_not_zero')
    selection=json.loads((previous.EXTERNAL/'selection.json').read_text())
    saved=json.loads((previous.EXTERNAL/'external_results/summary.json').read_text());pooled=[];records=[];bad=0
    for record in selection['recordings']:
        if record['split']!='evaluation':continue
        rows,_=prior.rows_from(previous.EXTERNAL/'public_speed_arrays',record)
        events=prior.conditional.events(rows,'nominal');check=prior.check_source.reconstruct(rows,'nominal')
        prior.equal(events,check,'unchanged_nominal_decisions')
        data=[(e[5][2],r['pose_speed_mps'],None) for e,r in zip(events,rows)]
        pooled.extend(data);metrics=bounds.summarize(data)
        point_large=sum(v is not None and r is not None and abs(v-r)>2 for v,r,u in data);bad+=point_large
        records.append(dict(local_id=record['local_id'],day=record['day'],reference_qualification=metrics,
                            unchanged_point_gnss_disagreement_gt2_count=point_large))
    result=bounds.summarize(pooled);old=saved['evaluation_case_metrics']['nominal']['conditional_bounded_candidate']['all']
    prior.equal(result['requests'],old['requests'],'requests');prior.equal(result['accepted'],old['accepted'],'accepted')
    prior.equal(bad,old['accepted_disagreement_gt2_count'],'point_disagreements')
    # Independent cardinality check: every accepted current output lacks an appraised bound.
    expected=sum(v is not None for v,r,u in pooled)
    prior.equal(result['accepted_without_qualified_numeric_bound'],expected,'unknown_denominator')
    prior.equal(result['accepted_with_qualified_numeric_bound'],0,'no_invented_bound')
    # Mathematical illustration only. The paper's static accuracy is not asserted as a hard dynamic bound.
    example=bounds.position_difference_budget(Fraction(3,10),Fraction(3,10),Fraction(1,5))
    report=dict(scope='Retrospective reference qualification of nominal bounded candidate on the frozen 18-recording evaluation',
                evaluation_recordings=len(records),reference_contract=spec,pooled=result,recordings=records,
                unchanged_point_proxy_metrics=old,
                illustrative_position_difference_budget=dict(assumed_position_bound_each_endpoint_m=.3,interval_seconds=.2,
                    interval_mean_velocity_vector_error_budget_mps=float(example),hypothetical_assumptions_only=True,
                    bound_on_instantaneous_speed=False,actual_wmid_dynamic_bound_established=False),
                production_policy_changed=False,physical_witness_enabled=False,promotion_allowed=False)
    prior.equal(freeze(),frozen,'unchanged_evidence')
    verification=dict(verification_passed=True,evaluation_recordings=len(records),checked_nominal_decisions=len(pooled),
                      preserved_large_point_disagreements=bad,unknown_reference_bound_preserved=True,
                      physical_reference_appraised=False,physical_validation_complete=False,promotion_allowed=False)
    for name,value in [('summary.json',report),('verification.json',verification)]:
        previous.immutable_write(output/name,previous.json_bytes(value))
    previous.immutable_write(output/'result_hashes.json',previous.json_bytes({n:prior.io.digest(output/n) for n in ('input_lock.json','summary.json','verification.json')}))
    print(json.dumps(verification,sort_keys=True));print('REFERENCE_QUALIFICATION_COMPLETE_NO_PROMOTION')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'reference_results');a=p.parse_args()
    try:run(a.output_dir)
    except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as e:print('REFERENCE_QUALIFICATION_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
