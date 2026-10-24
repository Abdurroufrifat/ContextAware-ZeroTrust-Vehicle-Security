import pytest,subprocess
from cps_common import R,BINARY
import traci
@pytest.mark.parametrize("profile",["emergency_follow","highway_braking","stop_restart"])
def test_both_vehicles_exist_at_first_physical_step(profile):
    traci.start([str(BINARY),"-n",str(R/"road.net.xml"),"-r",str(R/(profile+".rou.xml")),"--step-length",".01","--no-step-log","true"],stdout=subprocess.DEVNULL)
    try:
        traci.simulationStep(.01)
        assert set(traci.vehicle.getIDList())=={"ego","lead"}
    finally:traci.close()
