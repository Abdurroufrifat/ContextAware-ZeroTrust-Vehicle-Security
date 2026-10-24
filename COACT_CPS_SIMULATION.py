"""Entry point for the simulation-focused CO-ACT-CPS research release."""
from pathlib import Path
import sys,json,argparse,subprocess
P=Path(__file__).resolve().parent
R=P/"research"/"vehicle_closed_loop_cps"
def main():
    if len(sys.argv)>1 and sys.argv[1]=="core":
        return subprocess.run([sys.executable,str(P/"COACT_CPS_RESEARCH.py"),"core",*sys.argv[2:]]).returncode
    parser=argparse.ArgumentParser(description="CO-ACT-CPS closed-loop simulation evidence")
    parser.add_argument("command",nargs="?",choices=("status","verify","operator","report"),default="status")
    parser.add_argument("--scenario",default="scene_disinformation")
    parser.add_argument("--step",type=int,default=80)
    args=parser.parse_args()
    if args.command=="verify":
        return subprocess.run([sys.executable,str(R/"run_all.py")],cwd=P).returncode
    if args.command=="report":
        print(R/"RESULTS.md");print(R/"operator_view.html");return 0
    if args.command=="operator":
        cards=json.loads((R/"operator_examples.json").read_text())
        if args.scenario not in cards or not 0<=args.step<200:parser.error("unknown scenario or step")
        card=cards[args.scenario][args.step]
        print("Reason:",card["reason"])
        print("Accepted acceleration:",card["accepted_command_mmps2"]/1000,"m/s^2")
        print("Scene:",card["scene_status"],"| Range:",card["radar_status"],"| Command:",card["actuator_status"])
        print("Advisory:",card["untrusted_advisory"])
        print("Assurance:",card["assurance"])
        return 0
    file=R/"completion_status.json"
    if not file.exists():
        print("Simulation study is running; completed trial records:",len(list((R/"simulator_outputs").glob("trial_*/summary.json"))))
    else:print(json.dumps(json.loads(file.read_text()),indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
