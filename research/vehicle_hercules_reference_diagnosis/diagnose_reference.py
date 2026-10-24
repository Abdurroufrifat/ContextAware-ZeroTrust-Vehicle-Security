from pathlib import Path
import json,sys,hashlib
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development"
sys.path.insert(0,str(H));from hercules_core import load_archive,decode_radar
def stats(v):
 v=np.asarray(v);v=v[np.isfinite(v)]
 return {"count":len(v),"median":float(np.median(v)) if len(v) else None,"p95":float(np.quantile(v,.95)) if len(v) else None,"max":float(np.max(v)) if len(v) else None}
def interp(q,t,y):
 base=int(t[0]);q=(q-base)*1e-9;t=(t-base)*1e-9
 if y.ndim==1:return np.interp(q,t,y,left=np.nan,right=np.nan)
 return np.column_stack([np.interp(q,t,y[:,j],left=np.nan,right=np.nan) for j in range(y.shape[1])])
def ecef(lla):
 lat,lon=np.deg2rad(lla[:,:2]).T;h=lla[:,2];e2=6.6943799901413165e-3;a=6378137.;n=a/np.sqrt(1-e2*np.sin(lat)**2)
 return np.column_stack(((n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)))
out={}
for seq in ["Library01","Library02"]:
 D=H/"data"/seq;a=np.load(H/(seq+"_replay.npz"),allow_pickle=False);t=a["stamps"]
 rows=[l.split(",") for l in (D/"sensor_data"/"inspva.csv").read_text().splitlines() if l.strip()]
 it=np.array([int(r[0]) for r in rows],dtype=np.int64)
 x=np.array([[float(v) for v in r[1:10]] for r in rows]);pos=ecef(x[:,:3]);vel=x[:,3:6].copy();vel[:,2]*=-1
 # Magnitudes invariant to fixed world-coordinate rotations; no alignment fitted.
 secant=np.full(len(it),np.nan);secant[1:-1]=np.linalg.norm(pos[2:]-pos[:-2],axis=1)/((it[2:]-it[:-2])*1e-9)
 direct=np.linalg.norm(vel,axis=1)
 raw_pos=interp(t,it,pos);raw_pos_speed=np.full(len(t),np.nan);raw_pos_speed[1:-1]=np.linalg.norm(raw_pos[2:]-raw_pos[:-2],axis=1)/((t[2:]-t[:-2])*1e-9)
 raw_ins_speed=interp(t,it,direct)
 imurows=[l.split(",") for l in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if l.strip()]
 mt=np.array([int(r[0]) for r in imurows],dtype=np.int64);gyro=np.array([[float(v) for v in r[8:11]] for r in imurows])
 gn=interp(t,mt,np.linalg.norm(gyro,axis=1))
 # Compare scalar travelled distance, and raw displacement norm to integrated INS speed upper bound.
 dt=np.diff(it)*1e-9;steps=np.linalg.norm(np.diff(pos,axis=0),axis=1);travel=.5*(direct[:-1]+direct[1:])*dt
 bad_residual=np.maximum(0,steps-travel)
 bands={}
 for name,mask in {"all":np.ones(len(t),dtype=bool),"last15s":t>=t[-1]-15_050_000_000,"last60s":t>=t[-1]-60_000_000_000}.items():
  bands[name]={"candidate_vs_pose":stats(abs(a["speed"][mask]-a["reference_primary"][mask])),"candidate_vs_direct_INS":stats(abs(a["speed"][mask]-raw_ins_speed[mask])),"pose_derivative_vs_raw_position_derivative":stats(abs(a["reference_primary"][mask]-raw_pos_speed[mask])),"raw_position_derivative_vs_direct_INS":stats(abs(raw_pos_speed[mask]-raw_ins_speed[mask])),"gyro_norm_radps":stats(gn[mask])}
 seconds=[];base=int(t[-1]-20_000_000_000)
 for j in range(20):
  mask=(t>=base+j*1_000_000_000)&(t<base+(j+1)*1_000_000_000)
  seconds.append({"seconds_before_end":20-j,"radar":stats(a["speed"][mask]),"pose_derivative":stats(a["reference_primary"][mask]),"raw_position_derivative":stats(raw_pos_speed[mask]),"direct_INS":stats(raw_ins_speed[mask]),"gyro":stats(gn[mask])})
 out[seq]={"bands":bands,"last20_seconds":seconds,"INS_dt_ms":stats(dt*1000),"radar_dt_ms":stats(np.diff(t)*1e-6),"position_step_excess_over_integrated_speed_m":stats(bad_residual),"source_hashes":{n:hashlib.sha256((D/"sensor_data"/n).read_bytes()).hexdigest() for n in ["inspva.csv","xsens_imu.csv"]}}
 np.savez_compressed(R/(seq+"_reference_diagnostics.npz"),stamps=t,candidate=a["speed"],pose_speed=a["reference_primary"],raw_position_speed=raw_pos_speed,direct_ins_speed=raw_ins_speed,gyro_norm=gn)
 print(seq,json.dumps({"bands":bands,"last_five_seconds":seconds[-5:]}),flush=True)
(R/"reference_diagnosis.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
