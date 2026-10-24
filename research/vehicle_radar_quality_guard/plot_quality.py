from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def figures():
 result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));methods=result['roles']['calibration']['methods'];names=['timing_only','disagreement_guard','matched_random','withhold_all'];labels=['Timing-ready estimates','Disagreement safeguard','Same-budget random control','Withhold every estimate'];colors=['#777777','#2878a8','#c47927','#aaaaaa']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained',sharey=True)
 maxima=[methods[n]['accepted']['maximum_mps'] or 0 for n in names];withheld=[100*methods[n]['whole_block_withheld_fraction'] for n in names]
 axes[0].barh(labels,maxima,color=colors);axes[1].barh(labels,withheld,color=colors)
 for i,(maximum,rejected) in enumerate(zip(maxima,withheld)):
  axes[0].text(maximum+.15,i,'undefined' if names[i]=='withhold_all' else f'{maximum:.3f}',va='center',fontsize=9)
  axes[1].text(min(rejected+1.2,90),i,f'{rejected:.2f}%',va='center',fontsize=9)
 axes[0].set(xlabel='Largest disagreement among accepted estimates (m/s)',xlim=(0,12.2));axes[1].set(xlabel='All-block endpoints withheld, including startup (%)',xlim=(0,106));axes[0].invert_yaxis()
 for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('Development-only quality safeguard: selective accuracy and availability costs',fontsize=12)
 out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'quality_accuracy_availability.png',dpi=180);plt.close(fig)
if __name__=='__main__':figures()
