"""Separate flood-fill DBSCAN, brute-force silhouette and causal reconstruction."""
import argparse,gzip,json,math,sys
from functools import lru_cache
from fractions import Fraction
from decimal import Decimal,localcontext
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_monitor_comparison'))
import verify_comparison as oldcheck
equal=oldcheck.run.equal
sources=oldcheck.sources
POLICIES=('fixed_density_unprotected','adaptive_density_unprotected','fixed_density_protected','adaptive_density_protected')
ARMS=tuple((e,m) for e in (1.2,1.,.8,.5,.4,.2) for m in (15,12,10,8,6,4))


@lru_cache(maxsize=6)
def neighborhoods(points,eps):
    return tuple(tuple(j for j,y in enumerate(points) if abs(x-y)<=eps) for x in points)


def reference_model(values,eps,minpts):
    points=tuple(sorted(values));near=neighborhoods(points,eps)
    labels=[None]*len(points);visited=set();count=0
    for seed in range(len(points)):
        if seed in visited:continue
        visited.add(seed)
        if len(near[seed])<minpts:labels[seed]=-1;continue
        cluster=count;count+=1;labels[seed]=cluster;queue=list(near[seed]);queued=set(queue)
        for i in queue:
            if i not in visited:
                visited.add(i)
                if len(near[i])>=minpts:
                    for j in near[i]:
                        if j not in queued:queue.append(j);queued.add(j)
            if labels[i] in (None,-1):labels[i]=cluster
    clusters=[[points[i] for i,label in enumerate(labels) if label==k] for k in range(count)]
    if count==0:reward=-1.
    elif count==1:reward=minpts/eps
    else:
        means=[]
        for k,group in enumerate(clusters):
            scores=[]
            for x in group:
                if len(group)==1:scores.append(0.);continue
                within=sum(abs(x-y) for y in group)/(len(group)-1)
                between=min(sum(abs(x-y) for y in other)/len(other) for j,other in enumerate(clusters) if j!=k)
                scores.append((between-within)/max(within,between) if max(within,between)>0 else 0.)
            means.append(sum(scores)/len(scores))
        reward=max(means)
    return dict(eps=eps,minpts=minpts,core=[x for x,neighbors in zip(points,near) if len(neighbors)>=minpts],clusters=count,reward=round(reward,12))


@lru_cache(maxsize=16)
def reference_fit(values):
    # Exact rational centering precedes the square root, so a huge symmetric
    # prefix cannot gain a spurious mean through finite-context cancellation.
    n=len(values);mean=None;sd=None
    if n:
        rational=[Fraction.from_float(float(x)) for x in values]
        center=sum(rational,Fraction())/n;mean=float(center)
        if n>1:
            variance=sum(((x-center)**2 for x in rational),Fraction())/(n-1)
            with localcontext() as context:
                context.prec=80
                sd=float((Decimal(variance.numerator)/Decimal(variance.denominator)).sqrt())
    fit=dict(n=n,mean=mean,sd=sd,valid=bool(n>1 and mean is not None and sd is not None and math.isfinite(mean) and math.isfinite(sd) and sd>0))
    for key in ('mean','sd'):
        if fit[key] is not None and not math.isfinite(fit[key]):fit[key]=None;fit['valid']=False
    models=None
    if fit['valid']:
        z=tuple((x-fit['mean'])/fit['sd'] for x in values)
        if not all(math.isfinite(x) for x in z):fit['valid']=False
        else:
            all_models=[reference_model(z,e,m) for e,m in ARMS]
            pulls=[1 for _ in ARMS]
            def choice(t):
                best=None;best_score=-math.inf
                for i,model in enumerate(all_models):
                    score=round(model['reward']+(2*math.log(t)/pulls[i])**.5,12)
                    if score>best_score:best=i;best_score=score
                return best
            for t in range(37,73):pulls[choice(t)]+=1
            selected=choice(73);models=[all_models[21],all_models[selected]]
            # eps .5 is the fourth block; MinPts 8 is fourth within that block.
            def describe(m):return {k:v for k,v in m.items() if k!='core'}|dict(core_count=len(m['core']))
            fit=fit|dict(fixed=describe(models[0]),adaptive=describe(models[1])|dict(arm_index=selected,pull_counts=pulls))
    return fit,models


def reconstruct(rows,case):
    times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
    boundary=times[0]+5000000 if times else None
    histories=[[],[]];last=[None,None];cal=None;models=None;disabled=[False,False];out=[]
    for event in sources.reconstruct(rows,case):
        a=event[0];i=a[0];t=rows[i].get('pose_utime');c=a[4];peer=a[5];anchor=event[1]
        valid=sources.numeric(c) and c>=0 and sources.receipt_ok(t,a[3])
        eligible=(event[4] is True and sources.numeric(anchor) and anchor>=0 and sources.numeric(event[2]) and 0<=event[2]<=.5 and sources.receipt_ok(t,event[3]))
        residual=[c-peer if valid and sources.numeric(peer) and peer>=0 else None,c-anchor if valid and eligible else None]
        diff=[]
        for j,value in enumerate(residual):
            previous=last[j]
            delta=value-previous[0] if previous is not None and value is not None and type(t) is int and 0<=t-previous[1]<=100000 else None
            diff.append(delta);last[j]=(value,t) if value is not None and type(t) is int else None
        warm=type(t) is int and boundary is not None and t<boundary
        if type(t) is int and not warm and cal is None:
            fitted=[reference_fit(tuple(x)) for x in histories];cal=[x[0] for x in fitted];models=[x[1] for x in fitted]
        status=[];states=[None]*4;alarm=[None]*4;values=[None]*4;codes=[0]*4
        for track in range(2):
            delta=diff[track]
            if delta is not None and not sources.numeric(delta):disabled[track]=True;diff[track]=None
            s=('unavailable' if type(t) is not int else 'warmup' if warm else 'invalid_calibration' if not cal[track]['valid'] else
               'numerical_failure' if disabled[track] else 'missing' if delta is None else 'monitored')
            if warm and delta is not None and sources.numeric(delta) and len(histories[track])<1000:histories[track].append(delta)
            if s=='monitored':
                z=(delta-cal[track]['mean'])/cal[track]['sd']
                if not math.isfinite(z):disabled[track]=True;s='numerical_failure'
                else:
                    for k in range(2):
                        model=models[track][k];states[track*2+k]=z
                        alarm[track*2+k]=not any(abs(z-point)<=model['eps'] for point in model['core'])
            status.append(s)
            for k in range(track*2,track*2+2):
                if track==0:
                    if warm and a[6]&4:values[k]=c;codes[k]=1
                    elif s=='monitored' and alarm[k] is False:values[k]=c;codes[k]=1
                elif eligible:
                    values[k]=c if s=='monitored' and alarm[k] is False else anchor
                    codes[k]=1 if s=='monitored' and alarm[k] is False else 2
        out.append([i,diff,status,states,alarm,values,codes])
    if cal is None:cal=[reference_fit(tuple(x))[0] for x in histories]
    return out,dict(zip(('unprotected','protected'),cal))


class Metrics:
    def __init__(self):self.inner=oldcheck.Metrics()
    def add(self,*args):self.inner.add(*args)
    def report(self):return dict(zip(POLICIES,self.inner.report().values()))


def verify(data,baseline,output,baseline_checked=False):
    import run_density as run
    data=run.io.safe(data);baseline=run.io.safe(baseline);output=run.io.safe(output)
    if not baseline_checked:equal(run.io.read(baseline/'verification.json'),run.previous.baseverify.verify(data,baseline),'baseline')
    equal(run.io.read(output/'input_lock.json'),run.freeze(baseline),'input_lock')
    base=run.io.read(baseline/'batch_summary.json')
    names={'input_lock.json','summary.json','scene_metrics.jsonl.gz'}|{s['scene']+'_events.jsonl.gz' for s in base['scenes']}
    hashes=run.io.read(output/'result_hashes.json')
    if set(hashes)!=names:raise ValueError('Density artifact inventory differs')
    for name in names:
        if run.io.digest(output/name)!=hashes[name]:raise ValueError('Density artifact hash differs: '+name)
    pooled={case:Metrics() for case in sources.CASES};count=0;scenes=0
    with gzip.open(output/'scene_metrics.jsonl.gz','rt') as stats:
        for record in base['scenes']:
            rows=run.previous.ledger(baseline,record['scene']);times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
            first=times[0] if times else None;per={};cal={};delay={}
            with gzip.open(output/(record['scene']+'_events.jsonl.gz'),'rt') as stream:
                for ci,case in enumerate(sources.CASES):
                    events,cal[case]=reconstruct(rows,case);local=Metrics();delay[case]={p:None for p in POLICIES}
                    for event in events:
                        line=stream.readline()
                        if not line:raise ValueError('Truncated density events')
                        equal(json.loads(line),[ci]+event,'density.event')
                        row=rows[event[0]];phase=sources.previous.partition(row.get('pose_utime'),first)
                        args=(event,row.get('pose_speed_mps'),row.get('wheel_speed_mps'),phase)
                        local.add(*args);pooled[case].add(*args);count+=1
                        for j,p in enumerate(POLICIES):
                            if phase=='during' and event[4][j] is True and delay[case][p] is None:delay[case][p]=row['pose_utime']-first-5000000
                    per[case]=local.report()
                if stream.readline():raise ValueError('Extra density event')
            line=stats.readline()
            if not line:raise ValueError('Truncated density scene metrics')
            equal(json.loads(line),dict(scene=record['scene'],calibration=cal,first_during_alarm_elapsed_us=delay,metrics=per),'density.scene')
            scenes+=1
            if scenes%10==0 or scenes==base['scene_count']:print('Verified density scenes: '+str(scenes)+'/'+str(base['scene_count']),flush=True)
        if stats.readline():raise ValueError('Extra density scene metrics')
    expected=run.summary(base,{c:pooled[c].report() for c in sources.CASES},scenes,run.io.digest(output/'input_lock.json'))
    equal(run.io.read(output/'summary.json'),expected,'density.summary')
    equal(run.freeze(baseline),run.io.read(output/'input_lock.json'),'unchanged_inputs')
    return dict(verification_passed=True,verified_scenes=scenes,verified_event_rows=count,verified_decisions=count*4,
                verified_policy_windows=(scenes+1)*14*4*5,input_lock_sha256=expected['input_lock_sha256'],
                baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                method='Separate source/attack reconstruction, Decimal prefix statistics, flood-fill DBSCAN, brute-force silhouette, UCB replay, policy wrappers and scalar reductions',
                native_recent_av_method_reproduction_complete=False,conditional_simulation_only=True,
                real_source_protection_established=False,promotion_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','baseline','output'):p.add_argument('--'+name+'-dir',type=Path,required=True)
    args=p.parse_args()
    try:print(json.dumps(verify(args.data_dir,args.baseline_dir,args.output_dir),indent=2))
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as e:
        print('DENSITY_VERIFICATION_FAILED: '+str(e),file=sys.stderr);sys.exit(1)
