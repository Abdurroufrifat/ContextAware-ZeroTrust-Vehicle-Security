from lc_common import *
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=R/"confirmation_v2";report=json.loads((root/"comparison.json").read_text());audit=json.loads((root/"verification.json").read_text())
rows=[r for r in report["aggregates"] if r["case"]=="ALL"]
counter={r["method"]:0 for r in rows}
for p in audit["per_trial"]:counter[p["method"]]+=p["actual_certificate_counterexamples"]
fig,axes=plt.subplots(1,3,figsize=(15,5))
labels=[r["method"] for r in rows];colors=["#176c80" if m=="joint" else "#94aab3" for m in labels]
for ax,values,label in zip(axes,[[r["contact_trials"] for r in rows],[counter[m] for m in labels],[r["mean_progress_m"] for r in rows]],["Contact trials / 216","Actual-state stopping counterexamples","Mean vehicle progress (m)"]):
 ax.barh(labels,values,color=colors);ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
fig.suptitle("Frozen lifecycle confirmation: safety and progress")
fig.text(.5,.01,"All scenarios, including declared negative controls. Counters describe failed actual-state stopping conditions, not universal safety claims.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(root/"lifecycle_results.png",dpi=180);fig.savefig(root/"lifecycle_results.pdf");plt.close(fig)
print("LIFECYCLE_FIGURES_SAVED",flush=True)
