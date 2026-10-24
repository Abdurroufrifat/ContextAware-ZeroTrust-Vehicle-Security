"""Single source-tree entry point for CO-ACT-CPS research."""
from pathlib import Path
import sys,json,argparse,subprocess
PROJECT=Path(__file__).resolve().parent
RELEASE=PROJECT/"research"/"project_release"
sys.path.insert(0,str(RELEASE));sys.path.insert(0,str(PROJECT/"src"))
from release_common import install_guard
install_guard()
def main():
 args=sys.argv[1:]
 if args and args[0]=="core":
  # Preserve the original diagnosis CLI and its own explicit command choices.
  import os
  os.chdir(PROJECT)
  sys.argv=["coact-cps",*args[1:]]
  from coact_cps.cli import main as original_main
  original_main();return 0
 parser=argparse.ArgumentParser(description="CO-ACT-CPS offline research: status, evidence verification, signed vehicle demo, or original core CLI")
 parser.add_argument("command",choices=("status","verify","demo"),nargs="?",default="status")
 parsed=parser.parse_args(args)
 if parsed.command=="verify":return subprocess.run([sys.executable,str(RELEASE/"verify_project.py")]).returncode
 if parsed.command=="demo":
  from demo import run_demo
  print(json.dumps(run_demo(),indent=2));return 0
 report=json.loads((RELEASE/"completion_status.json").read_text(encoding="utf-8"))
 print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
