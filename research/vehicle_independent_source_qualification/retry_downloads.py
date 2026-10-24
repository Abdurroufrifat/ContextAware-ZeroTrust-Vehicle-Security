from pathlib import Path
import runpy,shutil
R=Path(__file__).resolve().parent
backup=R/'pilot_data_lock_attempt1.json'
assert not backup.exists()
shutil.copyfile(R/'pilot_data_lock.json',backup)
runpy.run_path(str(R/'acquire_pilot.py'),run_name='__main__')
