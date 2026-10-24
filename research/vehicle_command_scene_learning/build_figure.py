from cs_common import *
install_guard()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
a=json.loads((R/"contribution_assessment.json").read_text())
policies=["kinematic","refined_midpoint","refined_learned"];labels=["Original","Refined midpoint","Refined learned"];colors=["#3465a4","#71777b","#167c54"]
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for ax,scope,metric,title,ylabel in [
 (axes[0],"nominal","selected_mae_mps","Nominal driving","Selected-speed MAE (m/s)"),
 (axes[1],"totals","selected_mae_mps","All nine cases","Selected-speed MAE (m/s)"),
 (axes[2],"common_scene_compromise","errors_gt05","Scene domain also compromised","Errors >0.5m/s (count)")]:
 values=[a[scope][p][metric] for p in policies];bars=ax.bar(range(3),values,color=colors)
 for bar,value in zip(bars,values):
  ax.annotate(str(value) if metric=="errors_gt05" else f"{value:.4f}",(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
 ax.set_xticks(range(3),labels,rotation=12,ha="right");ax.set_title(title);ax.set_ylabel(ylabel)
 ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
fig.suptitle("Command-refined scene learning: frozen SUMO evidence",fontsize=12);fig.tight_layout(rect=(0,0,1,.93))
fig.savefig(R/"command_scene_results.png",dpi=180);plt.close(fig)
print("FIGURE_SAVED")
