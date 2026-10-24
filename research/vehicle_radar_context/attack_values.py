"""Compromised CAN signers; attacks use no independent reference."""
import copy
def craft(case,candidate,features,radar_speed):
    if candidate is None or features is None: return None,None
    f=copy.deepcopy(features);c=candidate
    if case=="nominal": pass
    elif case=="coordinated_plus_3": c+=3.;f["peer"]+=3.
    elif case=="coordinated_minus_3_clipped": c=max(0.,c-3.);f["peer"]=max(0.,f["peer"]-3.)
    elif case=="coordinated_zero": c=0.;f["peer"]=0.
    elif case=="coordinated_times_2": c*=2.;f["peer"]*=2.
    elif case=="coordinated_radar_match":
        if radar_speed is None: return None,None
        c=radar_speed;f["peer"]=radar_speed
    else: raise ValueError("Unknown frozen attack")
    c=round(c*1000)/1000;f["peer"]=round(f["peer"]*1000)/1000
    return c,f
