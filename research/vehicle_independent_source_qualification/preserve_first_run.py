from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent
for name in ['pilot_predictions.json','pilot_summary.json']:
 dst=R/('native_attempt1_'+name)
 assert not dst.exists()
 shutil.copyfile(R/name,dst)
(R/'COMPATIBILITY_NOTE.md').write_text('The unchanged author KellnerMethod fails under installed NumPy2.5 because its least-squares two coefficients are length-one arrays passed to math.atan2. A subclass flattens the coefficient array before the unchanged native formulas. This is a scalar-shape compatibility adaptation, not parameter or algorithm tuning. Original source and failed outcomes are retained. All comparator limits and missing-input outcomes remain visible.\n',encoding='utf-8')
