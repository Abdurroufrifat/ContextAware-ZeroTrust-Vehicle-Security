"""Metadata and arithmetic checks only; native scene flags are not replayed here."""
import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_scene_comparison'))
import scene_core as scene


def require(ok,reason):
    if not ok:raise ValueError(reason)


def validate(report):
    a=report['analysis'];v=report['verification'];n=a['scene_count']
    require(type(n) is int and n>0,'Scene count')
    require(a['protocol_id']=='matched-scene-comparison-v1','Audit protocol')
    require(v['verification_passed'] is True and v['verified_scenes']==n,'Audit completion flag')
    require(v['verified_policy_phase_flags']==n*14*7*5 and v['verified_paired_tables']==350,'Audit count arithmetic')
    require(v['input_lock_sha256']==a['input_lock_sha256'],'Audit input-lock identity')
    require(a['cases']==list(scene.CASES) and a['policies']==list(scene.POLICIES) and a['phases']==list(scene.GROUPS),'Audit inventory')
    require([(p['candidate'],p['comparator']) for p in a['pairs']]==list(scene.PAIRS),'Audit pair inventory')
    require(set(a['paired_scene_tables'])==set(scene.CASES),'Audit table inventory')
    checked=0;shared={}
    for case in scene.CASES:
        require(len(a['paired_scene_tables'][case])==5,'Pair count')
        for pi,groups in enumerate(a['paired_scene_tables'][case]):
            require(set(groups)==set(scene.GROUPS),'Phase inventory')
            for group,x in groups.items():
                fields=('scenes','both_pass','candidate_only','comparator_only','both_fail','empty_request_scenes','candidate_pass_scenes','comparator_pass_scenes')
                require(all(type(x[k]) is int and x[k]>=0 for k in fields),'Audit counter type')
                require(x['scenes']==n and sum(x[k] for k in ('both_pass','candidate_only','comparator_only','both_fail'))==n,'Audit outcome partition')
                require(x['empty_request_scenes']<=x['both_fail'],'Empty phase count')
                require(x['candidate_pass_scenes']==x['both_pass']+x['candidate_only'] and x['comparator_pass_scenes']==x['both_pass']+x['comparator_only'],'Audit pass count')
                require(math.isclose(x['candidate_minus_comparator_percentage_points'],100*(x['candidate_only']-x['comparator_only'])/n,abs_tol=1e-10),'Audit percentage-point difference')
                for policy,field in zip(scene.PAIRS[pi],('candidate_pass_scenes','comparator_pass_scenes')):
                    key=(case,group,policy)
                    require(key not in shared or shared[key]==x[field],'Shared policy scene pass count')
                    shared[key]=x[field]
                checked+=1
        require(set(a['pooled_metrics'][case])==set(scene.POLICIES),'Pooled policy inventory')
        for policy in scene.POLICIES:scene.validate_groups(a['pooled_metrics'][case][policy],policy in scene.NEW)
        for track,model in a['calibration_diagnostics'][case].items():
            require(track in ('unprotected','protected'),'Calibration track')
            require(model['scenes']==n and model['valid_scenes']+model['short_prefix_scenes']+model['zero_sd_scenes']==n,'Calibration scene partition')
            require(0<=model['subnormal_sd_scenes']<=model['valid_scenes'],'Subnormal scene count')
    for name in ('physical_witness_enabled','production_policy_changed','promotion_allowed','real_source_protection_established'):require(a[name] is False,'Unsupported physical/project claim')
    return dict(full_audit_metadata_and_arithmetic_passed=True,scene_count=n,paired_tables_checked=checked,pooled_cells_checked=490,
                raw_scene_flags_recomputed_here=False,nominal_zero_sd_scenes=a['calibration_diagnostics']['nominal']['unprotected']['zero_sd_scenes'],
                nominal_subnormal_sd_scenes=a['calibration_diagnostics']['nominal']['unprotected']['subnormal_sd_scenes'])
