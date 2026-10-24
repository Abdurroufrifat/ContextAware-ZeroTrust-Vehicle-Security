"""Read-only replay using SciPy solver, scalar metrics and separate raw parser."""
from common import *
install_guard()
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import csv,math,struct,tarfile
import numpy as np
from scipy.linalg import lstsq
inputs=json.loads((R/"input_lock.json").read_text())
assert all(sha(P/name)==digest for name,digest in inputs.items())
a=np.load(R/"frames_and_predictions.npz",allow_pickle=False)
membership=np.load(N/"evaluation.npz",allow_pickle=False)
for key in ("stamps","frame_ids","sequence","roles","reference"):
    np.testing.assert_array_equal(a[key],membership[key])
assert len(a["stamps"])==564 and len(a["offsets"])==565
assert a["offsets"][0]==0 and a["offsets"][-1]==len(a["features"])
assert np.all(np.diff(a["offsets"])>=20)
results=json.loads((R/"results.json").read_text())
methods={"planar_ols":(2,False),"planar_huber20":(2,True),"spatial_ols":(3,False),"spatial_huber20":(3,True)}
assert {name:(value["dimensions"],value["huber"]) for name,value in results["methods"].items()}==methods
worst=0.;recomputed={name:[] for name in methods}
for i in range(564):
    frame=a["features"][a["offsets"][i]:a["offsets"][i+1]]
    for name,(dim,robust) in methods.items():
        design=frame[:,:dim];target=-frame[:,3]
        assert np.linalg.matrix_rank(design)==dim
        v=lstsq(design,target,lapack_driver="gelsy")[0]
        for iteration in range(20 if robust else 0):
            residual=design@v-target
            w=np.where(abs(residual)>.3,.3/np.maximum(abs(residual),1e-12),1.)
            v=lstsq(design*np.sqrt(w)[:,None],target*np.sqrt(w),lapack_driver="gelsy")[0]
        speed=float(math.sqrt(sum(float(x)*float(x) for x in v)))
        worst=max(worst,abs(speed-float(a[name][i])))
        assert math.isclose(speed,float(a[name][i]),rel_tol=2e-10,abs_tol=2e-9),(i,name,speed,a[name][i])
        residual=design@v-target
        w=np.minimum(1.,.3/np.maximum(abs(residual),1e-12)) if robust else np.ones(len(residual))
        score=max(abs(design.T@(w*residual)))
        assert math.isclose(float(score),float(a[name+"_stationarity"][i]),rel_tol=1e-6,abs_tol=2e-9)
        assert a[name+"_reason"][i]=="numeric"
        recomputed[name].append(speed)
np.testing.assert_allclose(a["planar_huber20"],a["historical_candidate"],rtol=0,atol=1e-10)
assert results["prior_candidate_max_difference"]<=1e-10
assert results["prior_unrotated_ols_max_difference"]<=1e-10
observed_shift=float(np.max(abs(a["spatial_ols"]-a["historical_unrotated_ols"])))
assert observed_shift==results["transformed_spatial_ols_max_difference_from_historical"]
# Independent decoding and quaternion formula for first/middle/last per sequence.
def matrix(path):
    values=[]
    for line in path.read_text().splitlines():
        if ":" in line:
            try:parsed=[float(v) for v in line.split(":",1)[1].split()]
            except ValueError:continue
            if len(parsed)==12:return np.array(parsed).reshape(3,4)[:,:3]
    raise AssertionError("transform absent")
raw_oracles=0
for seqid,seq in enumerate(("Library01","Library02")):
    rows=np.flatnonzero(a["sequence"]==seqid);chosen=[int(rows[0]),int(rows[len(rows)//2]),int(rows[-1])]
    wanted={int(a["stamps"][i]):i for i in chosen}
    directory=H/"data"/seq
    imu=[line.split(",") for line in (directory/"sensor_data/xsens_imu.csv").read_text().splitlines() if line.strip()]
    times=np.array([int(v[0]) for v in imu],dtype=np.int64)
    calibration=matrix(directory/"Calibration/IMU_LiDAR.txt")@matrix(directory/"Calibration/Continental_LiDAR.txt").T
    seen=set()
    with tarfile.open(directory/"Radar/Continental.tar.gz","r|gz") as archive:
        for member in archive:
            if member.isdir():continue
            stamp=int(Path(member.name).stem)
            if stamp not in wanted:continue
            assert member.isfile() and member.size%29==0
            raw=archive.extractfile(member).read();rays=[];doppler=[]
            for x,y,z,d,distance,rcs,az,el in struct.iter_unpack("<fffffBff",raw):
                if not all(math.isfinite(v) for v in (x,y,z,d,distance)) or not 3<=distance<=100:continue
                norm=math.sqrt(x*x+y*y+z*z)
                if norm<=0:continue
                rays.append([x/norm,y/norm,z/norm]);doppler.append(d)
            j=int(np.searchsorted(times,stamp,side="right")-1)
            assert j>=0 and 0<=stamp-int(times[j])<=100_000_000
            q=np.array([float(v) for v in imu[j][1:5]]);q=q/np.linalg.norm(q);x,y,z,w=q
            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@calibration
            frame=np.column_stack([np.array(rays)@rotation.T,np.array(doppler)])
            i=wanted[stamp];stored=a["features"][a["offsets"][i]:a["offsets"][i+1]]
            np.testing.assert_allclose(frame,stored,rtol=0,atol=2e-13)
            seen.add(stamp);raw_oracles+=1
    assert seen==set(wanted)
def p95(values):
    ordered=sorted(values);position=(len(ordered)-1)*.95;left=int(position);right=math.ceil(position)
    return ordered[left]+(ordered[right]-ordered[left])*(position-left)
groups={"Library01_sparse_development":a["sequence"]==0,"Library02_development_transfer":a["sequence"]==1,
        "Library01_temporal_development":a["roles"]=="temporal_test"}
with (R/"metrics.csv").open(newline="",encoding="utf-8") as f:csvrows=list(csv.DictReader(f))
assert len(csvrows)==12 and len(results["metrics"])==12
checks=0
for group,mask in groups.items():
    for name in methods:
        pred=a[name][mask];ref=a["reference"][mask]
        errors=[abs(float(p)-float(y)) for p,y in zip(pred,ref) if math.isfinite(p) and math.isfinite(y)]
        expected={"attempts":len(pred),"numeric_estimates":sum(math.isfinite(p) for p in pred),
                  "invalid_estimates":sum(not math.isfinite(p) for p in pred),"scored":len(errors),
                  "unknown_reference":sum(math.isfinite(p) and not math.isfinite(y) for p,y in zip(pred,ref)),
                  "mae_mps":math.fsum(errors)/len(errors),"p95_mps":p95(errors),"max_mps":max(errors),
                  "above2":sum(e>2 for e in errors)}
        selected=[x for x in results["metrics"] if x["group"]==group and x["method"]==name]
        selected_csv=[x for x in csvrows if x["group"]==group and x["method"]==name]
        assert len(selected)==len(selected_csv)==1
        for key,value in expected.items():
            assert math.isclose(value,float(selected[0][key]),rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(value,float(selected_csv[0][key]),rel_tol=1e-12,abs_tol=1e-12);checks+=1
        diagnostic=results["diagnostics"][group][name]
        assert diagnostic["reason_counts"]=={"numeric":int(mask.sum())}
        assert diagnostic["condition_over100"]==int(np.sum(a[name+"_condition"][mask]>100))
        assert math.isclose(diagnostic["stationarity_inf_p95"],p95(list(a[name+"_stationarity"][mask])),abs_tol=1e-12)
with (R/"paired_mae.csv").open(newline="",encoding="utf-8") as f:pairedcsv=list(csv.DictReader(f))
assert len(pairedcsv)==len(results["paired_mae"])==9
for row,tablerow in zip(results["paired_mae"],pairedcsv):
    assert row["group"]==tablerow["group"] and row["candidate"]==tablerow["candidate"] and row["comparator"]==tablerow["comparator"]
    mask=groups[row["group"]]&np.isfinite(a["reference"])
    for name in methods:mask&=np.isfinite(a[name])
    vals=[abs(float(p)-float(y))-abs(float(q)-float(y)) for p,q,y in zip(a["planar_huber20"][mask],a[row["comparator"]][mask],a["reference"][mask])]
    expected=math.fsum(vals)/len(vals)
    assert len(vals)==row["common_scored"]==int(tablerow["common_scored"])
    assert math.isclose(expected,row["candidate_minus_comparator_mae"],abs_tol=1e-12)
    assert math.isclose(expected,float(tablerow["candidate_minus_comparator_mae"]),abs_tol=1e-12)
for name in ("fresh_final_holdout","physical_reference_qualified","published_full_framework_reproduced","novelty_established","promotion_allowed"):
    assert results[name] is False
assert all(sha(P/name)==digest for name,digest in inputs.items())
print(json.dumps({"passed":True,"frames":564,"solver_predictions":2256,"raw_feature_oracles":raw_oracles,
                  "metric_scalar_checks":checks,"paired_cells":9,"input_files":len(inputs),
                  "max_solver_difference_mps":worst,"historical_predictions_preserved":True},indent=2))
