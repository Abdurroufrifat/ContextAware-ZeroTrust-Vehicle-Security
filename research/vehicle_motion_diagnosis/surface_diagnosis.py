import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
def surface_metrics(source,target,pose):
 tree=cKDTree(target);distance,ids=tree.query(source@pose[:3,:3].T+pose[:3,3])
 _,near=tree.query(target,k=min(16,len(target)))
 p=target[near];p=p-p.mean(axis=1,keepdims=True)
 eigen,vectors=np.linalg.eigh(np.einsum("nki,nkj->nij",p,p)/p.shape[1])
 normal=vectors[:,:,0]
 moved=source@pose[:3,:3].T+pose[:3,3];keep=distance<=1.5
 n=normal[ids[keep]];residual=np.sum((moved[keep]-target[ids[keep]])*n,axis=1)
 jac=np.column_stack((n,np.cross(moved[keep],n)))
 singular=np.linalg.svd(jac,compute_uv=False);rank=int(np.linalg.matrix_rank(jac))
 condition=float(singular[0]/singular[-1]) if rank==6 else None
 tangential=np.sqrt(np.maximum(0,distance[keep]**2-residual**2))
 return dict(inliers=int(keep.sum()),overlap=float(keep.mean()),
  euclidean_p95_m=float(np.quantile(distance[keep],.95)),plane_p95_m=float(np.quantile(np.abs(residual),.95)),
  tangential_p95_m=float(np.quantile(tangential,.95)),jacobian_rank=rank,jacobian_singular_values=singular.tolist(),
  jacobian_condition=condition,physical_error_certified=False,
  limitation="PCA normals and local fixed-correspondence rank describe the fitted geometry; neither guarantees unique global motion nor bounds physical error")
def main():
 R=Path(__file__).resolve().parent;s=json.loads((R/"diagnosis_summary.json").read_text());run=R/s["runtime_relative_path"]
 out=R/"surface_results.json";assert not out.exists()
 protocol=dict(name="temporal_stationary_surface_residual_diagnosis_v1",
  hypothesis="separate acquisitions can sample different positions along the same surfaces, making point-distance residual large despite small motion error",
  decided_after_initial_control_outcomes=True,initial_control_summary_sha256=hashlib.sha256((R/"diagnosis_summary.json").read_bytes()).hexdigest(),
  inputs="all six prespecified near-stationary training/calibration temporal pairs already frozen by the initial protocol",
  metrics="16-neighbor PCA normals, Euclidean/normal/tangential inlier P95 within1.5m, six-DOF point-to-plane Jacobian rank and singular values; identity and previously fitted pose",
  restrictions="diagnostic only; no new estimator, threshold selection, uncertainty certificate, evaluation payload decoding or policy changes")
 pp=R/"surface_protocol_v1.json";assert not pp.exists()
 pp.write_text(json.dumps(protocol,indent=2)+"\n")
 (R/"surface_protocol_freeze.json").write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),before_surface_results=True),indent=2))
 rows=[]
 for r in json.loads((run/"temporal_results_scored.json").read_text()):
  with np.load(run/r["snapshot_file"]) as a:
   metrics=surface_metrics(a["source"],a["target"],np.array(r["estimated_pose"]))
   identity=surface_metrics(a["source"],a["target"],np.eye(4))
  rows.append(dict(anchor=r["anchor"],role=r["role"],snapshot_file=r["snapshot_file"],snapshot_sha256=r["snapshot_sha256"],at_fitted_pose=metrics,at_identity=identity,
   offline_speed_reference_mps=r["offline_speed_reference_mps"],speed_discrepancy_mps=r["speed_discrepancy_mps"]))
 out.write_text(json.dumps(rows,indent=2)+"\n")
 summary=dict(cases=len(rows),unique_recordings=1,roles=["train","calibration"],
  euclidean_p95_median_m=float(np.median([r["at_fitted_pose"]["euclidean_p95_m"] for r in rows])),
  plane_p95_median_m=float(np.median([r["at_fitted_pose"]["plane_p95_m"] for r in rows])),
  all_speed_discrepancies_below_half_mps=all(r["speed_discrepancy_mps"] is not None and r["speed_discrepancy_mps"]<.5 for r in rows),
  maximum_speed_discrepancy_mps=max(r["speed_discrepancy_mps"] for r in rows),
  local_full_rank_cases=sum(r["at_fitted_pose"]["jacobian_rank"]==6 for r in rows),
  physical_speed_authorized=False,promotion_allowed=False,uncertainty_calibrated=False,
  limitation="six correlated near-stationary pairs from one recording cannot validate moving vehicles, attack resistance, freshness or a calibrated uncertainty bound")
 (R/"surface_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
