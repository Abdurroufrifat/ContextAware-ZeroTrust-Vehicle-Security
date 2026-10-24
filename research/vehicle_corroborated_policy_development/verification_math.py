"""Independent baseline/control arithmetic. Does not import new policy, baseline_core or plant."""
import itertools,math
def union(intervals):
    if not intervals:return []
    events=[]
    for a,b in intervals:
        if a<=b:events.extend([(a,1),(b,-1)])
    events.sort(key=lambda x:(x[0],-x[1]))
    depth=0;start=None;result=[]
    for x,change in events:
        if change==1:
            if depth==0:start=x
            depth+=1
        else:
            depth-=1
            if depth==0:result.append((start,x))
    return result
def reference_sets(previous,observations,accepted,dt,temporal):
    m=len(observations)
    if m<2:return []
    hypotheses=[]
    prediction=[]
    if temporal:
        for a,b in previous:
            prediction.append((max(0,a+dt*max(-4,min(2,accepted-1))),max(0,b+dt*max(-4,min(2,accepted+1)))))
    for indices in itertools.combinations(range(m),m-1):
        a=max(observations[i][0] for i in indices);b=min(observations[i][1] for i in indices)
        if a>b:continue
        if temporal:
            for p,q in prediction:
                if max(a,p)<=min(b,q):hypotheses.append((max(a,p),min(b,q)))
        else:hypotheses.append((a,b))
    return union(hypotheses)
def reference_context(lo,hi,scene,radar=None):
    if scene is None:return None
    g=[scene["gap_um"]*1e-6-.0015,scene["gap_um"]*1e-6+.0015]
    v=[max(0,scene["lead_mmps"]*.001-.075),scene["lead_mmps"]*.001+.075]
    if radar is not None:
        g=[max(g[0],radar["gap_um"]*1e-6-.05),min(g[1],radar["gap_um"]*1e-6+.05)]
        v=[max(v[0],radar["relative_mmps"]*.001-.1+lo),min(v[1],radar["relative_mmps"]*.001+.1+hi)]
    if g[0]>g[1] or v[0]>v[1]:return None
    return {"ego_low":lo,"ego_high":hi,"gap_low":g[0],"gap_high":g[1],"lead_low":v[0],"lead_high":v[1],"gap_m":sum(g)/2,"lead_mps":sum(v)/2}
def reference_cbf(u,e,l,g):
    h0=g-2-e;values=[g-2,h0]
    for tick in range(10):
        e=max(0,e+.01*min(2,max(-4,u*.001+1)));l=max(0,l-.04);g=g+.01*(l-e)
        values.extend([g-2,g-2-e])
    values.append(g-2-e-.8*h0)
    return min(values)
def close(a,b,tol=1e-8):
    if a is None or b is None:assert a is b,(a,b);return
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:close(a[k],b[k],tol)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),(a,b)
        for x,y in zip(a,b):close(x,y,tol)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<=tol,(a,b)
    else:assert a==b,(a,b)

def midpoint_integer_valid(value,lo,hi):
    return type(value) is int and abs(value-(lo+hi)*500)<=.50000001
