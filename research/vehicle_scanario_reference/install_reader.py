"""Install the recorded isolated reader; reruns use the exact dependency lock."""
import subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent
lock = root / "requirements-lock.txt"
requirements = ["-r", str(lock)] if lock.exists() else ["rosbags==0.11.9"]
subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
lock.write_text(freeze)
print("READER ENVIRONMENT READY")
