from sg_common import *
install_guard(local_sumo=True)
import traci,contextlib,xml.etree.ElementTree as ET
root=R/"runtime"/"api_pilot";root.mkdir(exist_ok=True)
cmd=[str(BINARY),"-n",str(R/"road.net.xml"),"-r",str(R/"controlled.rou.xml"),"--step-length","0.01","--no-step-log","true","--precision","6","--collision.action","warn","--collision-output",str(root/"collisions.xml"),"--fcd-output",str(root/"fcd.xml"),"--fcd-output.acceleration","true"]
with (root/"console.log").open("w",encoding="utf-8") as console,contextlib.redirect_stdout(console):
 traci.start(cmd,stdout=console)
 try:
  traci.simulationStep(.01);traci.vehicle.setSpeedMode("ego",0);traci.vehicle.setSpeedMode("lead",0)
  v0=traci.vehicle.getSpeed("ego");traci.vehicle.setAcceleration("ego",1.,.2)
  traci.simulationStep(.1)
  observation={"traci_time_s":traci.simulation.getTime(),"ego_speed_mps":traci.vehicle.getSpeed("ego"),"ego_position":traci.vehicle.getLanePosition("ego"),"initial_speed":v0,"version":traci.getVersion()}
 finally:traci.close()
data=ET.parse(root/"fcd.xml").getroot();observations=[]
for step in data.findall("timestep"):
 for v in step.findall("vehicle"):
  if v.attrib["id"]=="ego":observations.append({"fcd_time_s":float(step.attrib["time"]),**v.attrib})
matching=[r for r in observations if abs(float(r["speed"])-observation["ego_speed_mps"])<1e-5]
assert matching and abs(observation["ego_speed_mps"]-v0-.09)<1e-6
observation["matching_fcd_rows"]=matching;write("api_pilot.json",observation)
print(json.dumps(observation,indent=2))
