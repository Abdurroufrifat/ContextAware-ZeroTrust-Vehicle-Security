"""Load the previously delivered, immutable calibration probe."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent/'witness_calibration_probe'
if not (BASE/'probe_core.py').is_file():
    raise SystemExit('Missing earlier calibration package: extract CO_ACT_CPS_Radar_Calibration_Probe.zip into D:\\CO-ACT-CPS first.')
sys.path.insert(0,str(BASE))
import probe_core as baseline
import analyze_probe as baseline_analysis
