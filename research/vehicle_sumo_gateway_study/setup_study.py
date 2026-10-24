from sg_common import *
import subprocess,platform,importlib.metadata
install_guard()
(R/"runtime").mkdir(exist_ok=True)
(R/"nodes.nod.xml").write_text('<nodes><node id="n0" x="0" y="0" type="priority"/><node id="n1" x="5000" y="0" type="priority"/></nodes>',encoding="utf-8")
(R/"edges.edg.xml").write_text('<edges><edge id="road" from="n0" to="n1" numLanes="1" speed="30"/></edges>',encoding="utf-8")
run=subprocess.run([str(NETCONVERT),"--node-files",str(R/"nodes.nod.xml"),"--edge-files",str(R/"edges.edg.xml"),"--output-file",str(R/"road.net.xml")],capture_output=True,text=True)
(R/"network_build.log").write_text(run.stdout+run.stderr,encoding="utf-8");assert run.returncode==0,run.stderr
for reference in (False,True):
 root='<routes><vType id="control" accel="2" decel="4" emergencyDecel="9" sigma="0" length="5" minGap="2" maxSpeed="30" speedFactor="1" carFollowModel="IDM" tau="1.5"/><vType id="idm" accel="2" decel="4" emergencyDecel="9" sigma="0" length="5" minGap="2" maxSpeed="18" speedFactor="1" carFollowModel="IDM" tau="1.5"/><route id="route" edges="road"/><vehicle id="lead" type="control" route="route" depart="0" departPos="100" departSpeed="12"/><vehicle id="ego" type="TYPE" route="route" depart="0" departPos="60" departSpeed="10"/></routes>'.replace("TYPE","idm" if reference else "control")
 (R/("reference.rou.xml" if reference else "controlled.rou.xml")).write_text(root,encoding="utf-8")
version=subprocess.run([str(BINARY),"--version"],capture_output=True,text=True)
write("environment.json",{"python":sys.version,"platform":platform.platform(),"sumo_binary":str(BINARY),"sumo_version":version.stdout,"sumo_sha256":hashlib.sha256(BINARY.read_bytes()).hexdigest(),"traci_version":importlib.metadata.version("traci"),"netconvert_sha256":hashlib.sha256(NETCONVERT.read_bytes()).hexdigest()})
print("Network and simulator metadata prepared")
