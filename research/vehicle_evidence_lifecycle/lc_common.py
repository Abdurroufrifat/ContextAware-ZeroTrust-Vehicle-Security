from pathlib import Path
import sys
R=Path(__file__).resolve().parent;P=R.parent.parent;BASE=P/"research/vehicle_comparative_validation"
sys.path.insert(0,str(BASE))
from cv_common import KEYS,checked,install_guard,BINARY,sha,dump
B=R
sys.path.insert(0,str(R))
