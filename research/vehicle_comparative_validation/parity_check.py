from cv_common import *
install_guard(local_sumo=True)
import traci,random,math,contextlib
from plant import step
def main():
    cfg=json.loads((B/"PROTOCOL.json").read_text());out=[]
    target=B/"runtime"/"sumo_parity";target.mkdir(exist_ok=True)
    for name,c in cfg["profiles"].items():
        route=target/(name+".rou.xml")
        route.write_text('<routes><vType id="car" accel="2" decel="4" emergencyDecel="4" sigma="0" length="5" minGap="0" maxSpeed="70"/><route id="r" edges="road"/><vehicle id="lead" type="car" route="r" depart="0" departPos="'+str(65+c["gap"])+'" departSpeed="'+str(c["lead_speed"])+'"/><vehicle id="ego" type="car" route="r" depart="0" departPos="60" departSpeed="'+str(c["ego_speed"])+'"/></routes>',encoding="utf-8")
        # The existing frozen network's edge ID is obtained from its explicit routes, not assumed.
        import xml.etree.ElementTree as ET
        edge=ET.parse(OLD/"emergency_follow.rou.xml").getroot().find("route").attrib["edges"]
        route.write_text(route.read_text().replace('edges="road"','edges="'+edge+'"'),encoding="utf-8")
        state={"ego_v":c["ego_speed"],"lead_v":c["lead_speed"],"ego_x":60.,"lead_x":65.+c["gap"]}
        max_speed=max_position=0.;zero_steps=0
        with (target/(name+".log")).open("w",encoding="utf-8") as f,contextlib.redirect_stdout(f):
            traci.start([str(BINARY),"-n",str(OLD/"road.net.xml"),"-r",str(route),"--step-length","0.01","--no-step-log","true","--no-warnings","true","--collision.action","warn","--collision.mingap-factor","0","--time-to-teleport","-1"],stdout=f)
            try:
                traci.simulationStep(.01);assert set(traci.vehicle.getIDList())=={"ego","lead"}
                for v in ("ego","lead"):traci.vehicle.setSpeedMode(v,0);traci.vehicle.setSpeedFactor(v,1);traci.vehicle.setLaneChangeMode(v,512)
                rng=random.Random(3011);held=0;segments=set()
                for i in range(1999):
                    if i>=400 and i%10==0:held=rng.randint(-4000,2000)/1000
                    u=-4 if i<300 else 2 if i<400 else held
                    segments.add(min(4,(10000+i*10000)//4000000))
                    lead=c["accelerations"][min(4,(10000+i*10000)//4000000)]
                    traci.vehicle.setAcceleration("ego",u,.01);traci.vehicle.setAcceleration("lead",lead,.01)
                    state=step(state,u,lead);traci.simulationStep((20000+i*10000)/1e6)
                    for v in ("ego","lead"):
                        se=abs(state[v+"_v"]-traci.vehicle.getSpeed(v));pe=abs(state[v+"_x"]-traci.vehicle.getLanePosition(v))
                        max_speed=max(max_speed,se);max_position=max(max_position,pe)
                        assert se<1e-7 and pe<1e-7,(name,i,v,se,pe)
                    zero_steps+=state["ego_v"]==0
            finally:traci.close()
        out.append({"profile":name,"steps":1999,"lead_segments_covered":sorted(segments),"max_speed_error_mps":max_speed,"max_position_error_m":max_position,"zero_contact_steps":zero_steps})
    dump(B/"sumo_parity.json",{"passed":True,"checks":out,"independent_simulator":"SUMO1.27.1","scope":"new plant parity under independent braking/restart commands, not a rerun of old540 trials"})
    print("SUMO_PARITY_PASS",json.dumps(out),flush=True)
if __name__=="__main__":main()
