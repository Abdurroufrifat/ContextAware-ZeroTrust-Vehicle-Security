from pathlib import Path
import json,hashlib,importlib.util,sys,time
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from pilot_core import decode_pcd,time_ns,nearest_reference
from native_compat import KellnerCompat
R=Path(__file__).resolve().parent
lock=json.loads((R/"author_source_lock.json").read_text())
for e in lock["files"].values():assert hashlib.sha256((R/e["local"]).read_bytes()).hexdigest()==e["sha256"]
spec=importlib.util.spec_from_file_location("locked_author_methods",R/"author_src_utils_methods.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rt=[l.split() for l in (R/"rtk_schema.sample").read_text().splitlines()]
rtimes=np.array([time_ns(l[1],l[2]) for l in rt],dtype=np.int64)
assert np.all(np.diff(rtimes)>0)
lla=np.array([[float(x) for x in l[3:6]] for l in rt])
lat,lon=np.deg2rad(lla[:,:2]).T;h=lla[:,2];a=6378137.;e2=6.6943799901413165e-3
n=a/np.sqrt(1-e2*np.sin(lat)**2)
ecef=np.column_stack(((n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)))
# NED is a fixed orthonormal rotation of ECEF differences: their norms are identical.
raw_speed=np.linalg.norm(np.diff(ecef,axis=0),axis=1)/(np.diff(rtimes)*1e-9)
ref_speed=savgol_filter(raw_speed,51,4);label_times=rtimes[1:]
radar_times={l[0]:time_ns(l[1],l[2]) for l in [x.split() for x in (R/"radar_times.sample").read_text().splitlines()]}
rows=[];flags={};time_gaps=np.diff(np.array(list(radar_times.values()),dtype=np.int64))*1e-9
for entry in json.loads((R/"pilot_data_lock.json").read_text())["files"]:
 row={"file":entry["file"],"radar_time_ns":radar_times[entry["file"][:-4]],"estimates":{}}
 try:
  data=(R/"pilot_data"/entry["file"]).read_bytes()
  assert hashlib.sha256(data).hexdigest()==entry["sha256"]
  pc=decode_pcd(data);row["points"]=len(pc["x"])
  for f in ["dotFlags","denoiseFlag","historyFrameFlag","dopplerCorrectionFlag"]:
   vals,counts=np.unique(pc[f],return_counts=True)
   for v,c in zip(vals,counts):flags.setdefault(f,{}).setdefault(str(int(v)),0);flags[f][str(int(v))]+=int(c)
  names=["x","y","z","alpha","beta","doppler","power"]
  arr=np.column_stack([pc[x] for x in names]).astype(np.float64)
  arr=arr[np.all(np.isfinite(arr),axis=1)]
  arr=arr[(arr[:,2]<=50)&(arr[:,0]>=-42)&(arr[:,0]<=42)&(arr[:,1]>=-.7)&(arr[:,1]<=10)]
  arr[:,:3]=arr[:,[2,0,1]]*np.array([1,-1,-1]);row["native_crop_points"]=len(arr)
  theta=np.deg2rad(arr[:,3]);phi=np.deg2rad(arr[:,4]);vr=arr[:,5]
  for name in ["kellner_native","alternative_native"]:
   start=time.perf_counter_ns()
   try:
    if name=="kellner_native":
     mask=(arr[:,2]>=-.15)&(arr[:,2]<=.15)
     est=KellnerCompat(b=0.,l=2.380,beta=0.,trials=300)
     out=est.estimate(vr[mask],theta[mask])[0]
    else:
     rot=Rotation.from_euler("zyx",[-90,180,0],degrees=True).as_matrix().T
     est=m.AlternativeMethod(R=rot,trials=300)
     out=est.estimate(vr,theta,phi)
    out=float(out)
    if not np.isfinite(out):raise ValueError("nonfinite estimate")
    row["estimates"][name]={"mps":out,"compute_ms":(time.perf_counter_ns()-start)/1e6}
   except Exception as ex:row["estimates"][name]={"error":str(ex)}
  match=nearest_reference(row["radar_time_ns"],label_times,max_gap=39_999_999)
  if match is not None and label_times[0]<=row["radar_time_ns"]<=label_times[-1]:
   j,gap=match;row.update(reference_mps=float(ref_speed[j]),reference_time_ns=int(label_times[j]),reference_gap_ns=gap)
 except Exception as ex:row["error"]=str(ex)
 rows.append(row)
 print(row["file"],row.get("reference_mps"),{k:v.get("mps",v.get("error")) for k,v in row["estimates"].items()},flush=True)
def metrics(name):
 valid=[r for r in rows if "reference_mps" in r and "mps" in r["estimates"].get(name,{})]
 error=np.array([abs(r["estimates"][name]["mps"]-r["reference_mps"]) for r in valid])
 return {"attempts":len(rows),"scored":len(valid),"numeric_outputs":sum("mps" in r["estimates"].get(name,{}) for r in rows),"mae_mps":float(np.mean(error)) if len(error) else None,"p95_mps":float(np.quantile(error,.95)) if len(error) else None,"maximum_mps":float(np.max(error)) if len(error) else None,"above_2_mps":int(sum(error>2)),"compute_p95_ms":float(np.quantile([r["estimates"][name]["compute_ms"] for r in valid],.95)) if valid else None}
summary={"scope":"50-frame training-source development pilot; offline smoothed GPS position-difference proxy; NOT security acceptance or final evaluation","author_commit":lock["commit"],"methods":{n:metrics(n) for n in ["kellner_native","alternative_native"]},"flags_point_counts":flags,"radar_stream":{"frames":len(radar_times),"median_gap_s":float(np.median(time_gaps)),"p95_gap_s":float(np.quantile(time_gaps,.95)),"max_gap_s":float(np.max(time_gaps)),"gaps_above_100ms":int(sum(time_gaps>.1))},"reference":{"positions":len(rt),"velocity_is_position_derivative":True,"smoothing":"offline centered Savitzky-Golay window51 polynomial4","physical_error_bound_mps":None,"clock_offset_bound_ns":None},"physical_witness_admitted":False,"production_changed":False}
(R/"pilot_predictions.json").write_text(json.dumps(rows,indent=2,allow_nan=False),encoding="utf-8")
(R/"pilot_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps(summary,indent=2),flush=True)
