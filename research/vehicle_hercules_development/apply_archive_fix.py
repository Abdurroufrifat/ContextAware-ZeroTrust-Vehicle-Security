from pathlib import Path
import shutil,hashlib,json
R=Path(__file__).resolve().parent
assert not (R/"development_summary.json").exists()
for name in ["hercules_core.py","run_development.py","experiment_lock.json"]:
 dst=R/("attempt1_"+name);assert not dst.exists();shutil.copyfile(R/name,dst)
p=R/"run_development.py";s=p.read_text()
s=s.replace("latest_imu,fit_planar","latest_imu,fit_planar,load_archive")
s=s.replace(' with tarfile.open(D/"Radar"/"Continental.tar.gz","r:gz") as archive:', ' payloads=load_archive(D/"Radar"/"Continental.tar.gz")\n with tarfile.open(D/"Radar"/"Continental.tar.gz","r:gz") as archive:')
s=s.replace('b=archive.extractfile(files[int(stamp)]).read()','b=payloads[int(stamp)]')
p.write_text(s,encoding="utf-8")
(R/"EXPERIMENT_PROTOCOL.md").open("a",encoding="utf-8").write("\nLoader performance amendment before any completed sequence scores: interrupted original replay because timestamp-order seeks repeatedly decompress the archive. Stream each archive once into a bounded in-memory timestamp map; unchanged bytes/membership/estimator/guards/reference. Initial run and lock preserved as attempt1. The final timing replay runs without concurrent core tests.\n")
print("Applied archive access fix; original files and lock preserved.")
