"""Single causal signed-median candidate, offline and disabled for policy use."""
import statistics
from base_loader import baseline


def signed_features(radars,window_start_ns):
    if any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(radars,radars[1:])):
        raise ValueError('event timestamps must be strictly increasing')
    history=[];head=0;result={}
    for event,r in enumerate(radars):
        t=r['bag_time_ns'];good=baseline.valid_sample(r,250000000)
        if good:history.append(r)
        while head<len(history) and history[head]['bag_time_ns']<t-1000000000:head+=1
        if t<window_start_ns+1000000000 or not good or len(history)-head<10:continue
        result[event]=abs(statistics.median(q['speed_raw'] for q in history[head:]))
    return result


def paired_records(radars,obds,window_start_ns):
    features=signed_features(radars,window_start_ns)
    original,stats=baseline.causal_pairs(radars,obds,window_start_ns)
    events={r['bag_time_ns']:i for i,r in enumerate(radars)}
    result=[]
    for row in original:
        event=events[row['time_ns']]
        if event not in features:raise ValueError('candidate lost a baseline endpoint')
        result.append({**row,'feature_raw':features[event]})
    return result,stats
