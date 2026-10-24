from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def figures():
 result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));overall=result['methods']['disagreement_guard'];segments=list(result['segments_300s_no_history_reset'].values());labels=['Whole\nrecording']+[f'{5*i}–{min(5*(i+1),result["duration_s"]/60):.1f}' for i in range(len(segments))];values=[overall]+segments;colors=['#666666']+['#c43c39' if s['whole_acceptance_fraction']<.95 else '#2878a8' for s in segments]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
 losses=[100*(1*m['whole_acceptance_fraction']) for m in values];losses=[100-x for x in losses];p95=[m['accepted']['p95_mps'] for m in values]
 axes[0].bar(labels,losses,color=colors);axes[0].axhline(5,color='#b32624',ls='--',lw=1,label='5% maximum withholding');axes[0].set(ylabel='Withheld observed radar endpoints (%)',xlabel='Reporting interval (minutes)',ylim=(0,8),title='First five-minute interval still misses availability target');axes[0].legend(fontsize=8)
 axes[1].bar(labels,p95,color=colors);axes[1].axhline(.5,color='#b32624',ls='--',lw=1,label='0.5 m/s maximum P95');axes[1].set(ylabel='P95 disagreement of usable estimates (m/s)',xlabel='Reporting interval (minutes)',ylim=(0,.62),title='Larger reporting intervals pass the conditional P95 target');axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=8)
 fig.suptitle('Full-recording development replay: one startup, no history reset at reporting boundaries',fontsize=11)
 out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'continuous_accuracy_and_costs.png',dpi=180);plt.close(fig)
if __name__=='__main__':figures()
