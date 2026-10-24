"""Offline test subprocess; public fixture input arrives only through stdin."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from coact_cps.vehicle.durable_policy import DurableVehiclePolicy
from coact_cps.vehicle.policy import VehicleRequest
def main():
    p=json.load(sys.stdin)
    e=DurableVehiclePolicy(p['db'],{k:bytes.fromhex(v) for k,v in p['keys'].items()},bytes.fromhex(p['cap_key']),{tuple(v) for v in p['acl']})
    req=VehicleRequest(**p['request'])
    result=e.consume_capability(p['token'],req,p['frames'],p['now'])
    if p.get('exit_after_commit'): os._exit(0)
    print(json.dumps(dict(authorized=result.authorized,reasons=result.reasons)))
if __name__=='__main__': main()
