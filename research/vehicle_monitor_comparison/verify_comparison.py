"""Independent residual calibration, state recurrences, wrappers and scalar reduction."""
import argparse,gzip,json,math,sys
from array import array
from decimal import Decimal,localcontext
from pathlib import Path
import run_comparison as run
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_conditional_source'))
import verify_conditional as sources
POLICIES=('cusum_unprotected','ewma_unprotected','cusum_protected','ewma_protected')
GROUPS=('all','before','during','after','unavailable')


def calibration(data):
    n=len(data);mu=None;sd=None
    # Decimal residual arithmetic avoids squaring tiny floats into zero.
    # Constant inputs are checked exactly before any rounded decimal division.
    with localcontext() as context:
        context.prec=80
        values=[Decimal.from_float(float(x)) for x in data]
        if n:
            mean=sum(values)/n;mu=float(mean)
        if n>1:
            sd=0. if all(x==data[0] for x in data) else float((sum((x-mean)**2 for x in values)/(n-1)).sqrt())
    return dict(n=n,mean=mu,sd=sd,valid=bool(n>1 and sources.numeric(mu) and sources.numeric(sd) and sd>0))


def reconstruct(rows,case):
    valid_times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
    boundary=valid_times[0]+5000000 if valid_times else None;fitted=None;prefix=[[],[]]
    accum=[[0.,0.],0.,[0.,0.],0.];result=[];disabled=[False,False]
    for e in sources.reconstruct(rows,case):
        a=e[0];i=a[0];t=rows[i].get('pose_utime');c=a[4];pose=a[5];anchor=e[1]
        wheel_ok=sources.numeric(c) and c>=0 and sources.receipt_ok(t,a[3])
        anchor_ok=(e[4] is True and sources.numeric(anchor) and anchor>=0 and sources.numeric(e[2]) and 0<=e[2]<=.5 and sources.receipt_ok(t,e[3]))
        residuals=[c-pose if wheel_ok and sources.numeric(pose) and pose>=0 else None,c-anchor if wheel_ok and anchor_ok else None]
        warm=type(t) is int and boundary is not None and t<boundary
        if type(t) is int and not warm and fitted is None:fitted=list(map(calibration,prefix))
        status=[];alarms=[None,None,None,None];values=[None]*4;codes=[0]*4
        for track in range(2):
            state=('unavailable' if type(t) is not int else 'warmup' if warm else
                   'invalid_calibration' if not fitted[track]['valid'] else 'numerical_failure' if disabled[track] else 'missing' if residuals[track] is None else 'monitored')
            if state=='monitored':
                base=track*2;old_hi,old_lo=accum[base];old_ewma=accum[base+1]
                z=(residuals[track]-fitted[track]['mean'])/fitted[track]['sd']
                high=max(0.,old_hi+z-.5);low=max(0.,old_lo-z-.5)
                smoothed=.3*z+.7*old_ewma
                if all(math.isfinite(x) for x in (z,high,low,smoothed)):
                    accum[base]=[high,low];accum[base+1]=smoothed
                    alarms[base]=high>5 or low>5
                    alarms[base+1]=abs(smoothed)>3*math.sqrt(.3/(2-.3))
                else:disabled[track]=True;state='numerical_failure'
            status.append(state)
            if warm and residuals[track] is not None:prefix[track].append(residuals[track])
            for offset in range(2):
                j=track*2+offset
                if track==0:
                    if warm and a[6]&4:values[j]=c;codes[j]=1
                    elif state=='monitored' and alarms[j] is False:values[j]=c;codes[j]=1
                elif anchor_ok:
                    values[j]=c if state=='monitored' and alarms[j] is False else anchor
                    codes[j]=1 if state=='monitored' and alarms[j] is False else 2
        result.append([i,residuals,status,[list(x) if isinstance(x,list) else x for x in accum],alarms,values,codes])
    if fitted is None:fitted=list(map(calibration,prefix))
    return result,dict(zip(('unprotected','protected'),fitted))


def empty():
    return dict(requests=0,accepted=0,unscored=0,errors=array('d'),bad_trace=0,max_trace=None,codes=[0,0,0],
                monitored=0,alarms=0,warmup=0,invalid_calibration=0,numerical_failure=0,missing=0,unavailable=0)


class Metrics:
    def __init__(self):self.cells={p:{g:empty() for g in GROUPS[1:]} for p in POLICIES}
    def add(self,event,reference,wheel,phase):
        for j,p in enumerate(POLICIES):
            cell=self.cells[p][phase];cell['requests']+=1;cell['codes'][event[6][j]]+=1
            cell[event[2][j//2]]+=1;cell['alarms']+=event[4][j] is True
            v=event[5][j]
            if v is None:continue
            cell['accepted']+=1
            if sources.numeric(reference):cell['errors'].append(abs(v-reference))
            else:cell['unscored']+=1
            if sources.numeric(wheel):
                d=abs(v-wheel);cell['bad_trace']+=d>.5+1e-12
                cell['max_trace']=d if cell['max_trace'] is None else max(d,cell['max_trace'])
    @staticmethod
    def reduce(c):
        e=sorted(c['errors']);n=len(e);coverage=100*c['accepted']/c['requests'] if c['requests'] else None;p95=None
        if n:
            pos=.95*(n-1);lo=math.floor(pos);hi=math.ceil(pos);p95=e[lo]*(hi-pos)+e[hi]*(pos-lo) if lo!=hi else e[lo]
        bad=sum(x>2 for x in e)
        monitor={k:c[k] for k in ('requests','monitored','alarms','warmup','invalid_calibration','numerical_failure','missing','unavailable')}
        monitor['alarm_percent_of_monitored']=100*c['alarms']/c['monitored'] if c['monitored'] else None
        return dict(requests=c['requests'],accepted=c['accepted'],coverage_percent=coverage,scored_accepted=n,
                    accepted_unscorable=c['unscored'],mae_reference_disagreement_mps=sum(e)/n if n else None,
                    p95_reference_disagreement_mps=p95,max_reference_disagreement_mps=e[-1] if n else None,
                    accepted_disagreement_gt2_count=bad,trace_bound_violation_count=c['bad_trace'],max_deviation_from_original_wheel_mps=c['max_trace'],
                    selection_counts=dict(zip(('abstain','candidate','protected_copy'),c['codes'])),monitor=monitor,
                    diagnostic_gates_pass=bool(c['requests'] and coverage>=95 and p95 is not None and p95<=.5 and bad==0 and c['unscored']==0))
    def report(self):
        result={}
        for p,cells in self.cells.items():
            total=empty()
            for c in cells.values():
                for k in total:
                    if k=='errors':total[k].extend(c[k])
                    elif k=='codes':total[k]=[x+y for x,y in zip(total[k],c[k])]
                    elif k=='max_trace':
                        if c[k] is not None:total[k]=c[k] if total[k] is None else max(total[k],c[k])
                    else:total[k]+=c[k]
            result[p]={'all':self.reduce(total),**{g:self.reduce(c) for g,c in cells.items()}}
        return result


def verify(data,baseline,output,baseline_checked=False):
    data=run.io.safe(data);baseline=run.io.safe(baseline);output=run.io.safe(output)
    if not baseline_checked:
        run.equal(run.io.read(baseline/'verification.json'),run.previous.baseverify.verify(data,baseline),'baseline_verification')
    run.equal(run.io.read(output/'input_lock.json'),run.freeze(baseline),'input_lock')
    base=run.io.read(baseline/'batch_summary.json');names={'input_lock.json','summary.json','scene_metrics.jsonl.gz'}|{s['scene']+'_events.jsonl.gz' for s in base['scenes']}
    hashes=run.io.read(output/'result_hashes.json')
    if set(hashes)!=names:raise ValueError('Comparison inventory differs')
    for n in sorted(names):
        if run.io.digest(output/n)!=hashes[n]:raise ValueError('Comparison artifact changed: '+n)
    pooled={c:Metrics() for c in sources.CASES};count=0;scenes=0
    with gzip.open(output/'scene_metrics.jsonl.gz','rt',encoding='utf-8') as stats:
        for s in base['scenes']:
            rows=run.previous.ledger(baseline,s['scene']);times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
            first=times[0] if times else None;per={};cal={};delay={}
            with gzip.open(output/(s['scene']+'_events.jsonl.gz'),'rt',encoding='utf-8') as stream:
                for ci,case in enumerate(sources.CASES):
                    events,cal[case]=reconstruct(rows,case);local=Metrics();delay[case]={p:None for p in POLICIES}
                    for event in events:
                        line=stream.readline()
                        if not line:raise ValueError('Truncated comparison ledger')
                        run.equal(json.loads(line),[ci]+event,s['scene']+'.'+case+'.event')
                        r=rows[event[0]];phase=sources.previous.partition(r.get('pose_utime'),first)
                        args=(event,r.get('pose_speed_mps'),r.get('wheel_speed_mps'),phase)
                        local.add(*args);pooled[case].add(*args);count+=1
                        for j,p in enumerate(POLICIES):
                            if phase=='during' and event[4][j] is True and delay[case][p] is None:delay[case][p]=r['pose_utime']-first-5000000
                    per[case]=local.report()
                if stream.readline():raise ValueError('Extra comparison ledger row')
            line=stats.readline()
            if not line:raise ValueError('Truncated scene metrics')
            run.equal(json.loads(line),dict(scene=s['scene'],calibration=cal,first_during_alarm_elapsed_us=delay,metrics=per),'scene_metrics')
            scenes+=1
            if scenes%10==0 or scenes==base['scene_count']:print('Verified comparison scenes: '+str(scenes)+'/'+str(base['scene_count']),flush=True)
        if stats.readline():raise ValueError('Extra scene metrics')
    expected=run.summary(base,{c:pooled[c].report() for c in sources.CASES},scenes,run.io.digest(output/'input_lock.json'))
    run.equal(run.io.read(output/'summary.json'),expected,'comparison_summary')
    if run.freeze(baseline)!=run.io.read(output/'input_lock.json'):raise ValueError('Inputs changed during verification')
    return dict(verification_passed=True,verified_scenes=scenes,verified_event_rows=count,verified_decisions=count*4,
                verified_policy_windows=(scenes+1)*14*4*5,input_lock_sha256=expected['input_lock_sha256'],
                baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                method='Separate attack/source reconstruction, residual fit, CUSUM/EWMA states, policy wrappers and scalar reductions',
                native_recent_av_method_reproduction_complete=False,conditional_simulation_only=True,real_source_protection_established=False,promotion_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','baseline','output'):p.add_argument('--'+key+'-dir',type=Path,required=True)
    a=p.parse_args()
    try:print(json.dumps(verify(a.data_dir,a.baseline_dir,a.output_dir),indent=2))
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as e:
        print('MONITOR_VERIFICATION_FAILED: '+str(e),file=sys.stderr);sys.exit(1)
