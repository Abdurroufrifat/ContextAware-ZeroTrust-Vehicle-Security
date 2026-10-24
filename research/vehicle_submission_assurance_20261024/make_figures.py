from pathlib import Path
import json,gzip
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
S=Path(__file__).resolve().parent;P=S.parent.parent;B=P/"research/vehicle_evidence_lifecycle/confirmation_v2"
a=json.loads((S/"analysis.json").read_text());rows=[r for r in a["aggregates"] if r["scope"]=="all" and r["profile"]=="ALL"]
times={r["method"]:[] for r in rows}
for t in json.loads((B/"results.json").read_text())["trials"]:
 with gzip.open(B/t["trace"],"rt") as f:
  times[t["method"]].extend(json.loads(x)["decision_wall_seconds"] for x in f)
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
offset={"joint":(-35,16),"no_learning":(8,-18),"robust_current":(-90,8)}
for r in rows:
 kw={"marker":"x","color":"#075e78","s":75,"linewidths":2.3,"zorder":4} if r["method"]=="joint" else {"marker":"o","facecolors":"none","edgecolors":"#b33b46","s":100,"linewidths":1.5,"zorder":3} if r["method"]=="no_learning" else {"s":60}
 ax[0].scatter(r["mean_progress_m"],r["contact_fraction"]*100,**kw)
 ax[0].annotate(r["method"],(r["mean_progress_m"],r["contact_fraction"]*100),xytext=offset.get(r["method"],(5,6)),textcoords="offset points",fontsize=8)
ax[0].set(xlim=(0,260),ylim=(-1,22),xlabel="Mean progress (m; higher is more motion)",ylabel="Contact trials (%)",title="Empirical safety-progress tradeoff")
for m,values in times.items():
 values=sorted(values);stride=max(1,len(values)//1500);xs=values[::stride]+[values[-1]];ys=[(i*stride+1)/len(values) for i in range(len(xs)-1)]+[1]
 ax[1].plot([v*1000 for v in xs],ys,label=m)
ax[1].axvline(100,color="black",ls="--",lw=1);ax[1].set(xscale="log",xlabel="Recorded loop wall time (ms)",ylabel="Empirical cumulative fraction",title="Observed timing; offline simulation")
ax[1].legend(fontsize=7);fig.suptitle("All predeclared conditions retained");fig.tight_layout();fig.savefig(S/"assurance_results.png",dpi=180);fig.savefig(S/"assurance_results.pdf");plt.close(fig)
print("ASSURANCE_FIGURES_CHECKED_LAYOUT")
