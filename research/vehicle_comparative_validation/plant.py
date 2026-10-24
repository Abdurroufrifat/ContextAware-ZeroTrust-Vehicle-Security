"""10ms right-endpoint plant; initial state is SUMO's first inserted state at10ms."""
import math
def step(state,ego_accel,lead_accel,dt=.01):
    if dt<=0 or not all(math.isfinite(v) for v in (*state.values(),ego_accel,lead_accel,dt)):raise ValueError("plant")
    ego=max(0,state["ego_v"]+ego_accel*dt);lead=max(0,state["lead_v"]+lead_accel*dt)
    return {"ego_v":ego,"lead_v":lead,"ego_x":state["ego_x"]+ego*dt,"lead_x":state["lead_x"]+lead*dt}
def public_window(source,count,offset):
    n=len(source["age_us"])
    if n==0 or count<0 or offset<0:raise ValueError("public sequence")
    return {k:[source[k][(offset+i)%n] for i in range(count)] for k in ("age_us","innovation_mps")}
