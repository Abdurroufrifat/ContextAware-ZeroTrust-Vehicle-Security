from pathlib import Path
import sys
R=Path(__file__).resolve().parent;P=R.parent.parent;B=P/"research/vehicle_comparative_validation";F=P/"research/vehicle_failure_analysis_20261024"
sys.path.insert(0,str(B))
from cv_common import KEYS,sha,dump
sys.path.insert(0,str(F));sys.path.insert(0,str(R))
