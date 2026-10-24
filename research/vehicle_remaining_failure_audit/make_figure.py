"""Optional static diagnostic figure; not part of inference or portable audit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def main():
 result=json.loads((ROOT/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True);data={}
 for ax,(scope,window,title) in zip(axes,[('fit','evaluation_urban04_0','Held-route window: evaluation_urban04_0'),('full','calibration_0','Inspected development: calibration_0')]):
  cells=result['populations'][scope]['windows'][window]['by_reference_speed'];names=list(cells);values=[cells[n]['models']['signed_axis_corrected']['p95_mps'] for n in names];counts=[cells[n]['count'] for n in names];data[window]={'speed_bins':names,'p95_mps':values,'row_counts':counts};bars=ax.bar(range(4),values,color=['#c75446' if v>.5 else '#367997' for v in values],width=.65);ax.axhline(.5,color='#333333',linestyle='--',linewidth=1.2);ax.set_xticks(range(4),[f'{label}\nn={n:,}' for label,n in zip(['<1','1–<5','5–<10','≥10'],counts)]);ax.set_title(title,fontsize=11,pad=15);ax.set_xlabel('Reference speed (m/s)',labelpad=10);ax.set_ylim(0,.82);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
  for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.015,f'{v:.3f}',ha='center',fontsize=10)
 axes[0].set_ylabel('P95 absolute speed error (m/s)');fig.suptitle('Remaining failures have different speed patterns',fontsize=14,y=.98);fig.text(.5,.015,'Frozen predictions and original 0.5 m/s limit. Descriptive strata; no model or gate change.',ha='center',fontsize=9,color='#444444');fig.tight_layout(rect=(0,.055,1,.93));folder=ROOT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'remaining_failure_speed.png',dpi=170);plt.close(fig);(folder/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
