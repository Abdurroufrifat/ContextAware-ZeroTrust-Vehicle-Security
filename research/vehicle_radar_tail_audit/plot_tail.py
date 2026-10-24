"""Scientific figures from the archived post-hoc diagnostic records."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
def figures():
 context=json.loads((ROOT/'worst_event_context.json').read_text(encoding='utf-8'));result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'));center=context['center_ns'];rows=context['endpoints'];time=np.array([(r['time_ns']-center)/1e9 for r in rows]);out=ROOT/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,1,figsize=(10,6.7),sharex=True,layout='constrained')
 axes[0].plot(time,[r['reference_mps'] for r in rows],color='black',lw=2,label='Offline reference')
 for name,color,label in [('causal_residual_tcn','#c43c39','Residual TCN'),('reference_median_affine','#b57918','Median anchor'),('paper_inspired_cnn','#2878a8','Approximate CNN comparator')]:axes[0].plot(time,[r['predictions_mps'][name] for r in rows],color=color,label=label,lw=1.5)
 axes[0].set(ylabel='Speed estimate (m/s)',title='Largest calibration disagreement: frozen models, no observations removed');axes[0].legend(loc='lower left',ncol=2,fontsize=9);axes[0].grid(alpha=.2)
 raw=context['radar_records'];rt=[(r['bag_time_ns']-center)/1e9 for r in raw];rv=np.array([r['speed_raw'] for r in raw]);axes[1].scatter(rt,rv,s=20,color=np.where(abs(rv)<=1,'#c43c39','#2878a8'))
 axes[1].axhspan(-1,1,color='#c43c39',alpha=.08,label='Post-hoc low-return range: |raw| ≤ 1')
 axes[1].axvspan(-1,0,color='#6c757d',alpha=.1,label='Actual one-second anchor history')
 axes[1].set(xlabel='Time relative to worst endpoint (s)',ylabel='Raw radar value (native units)',title='Recent returns pass timing checks but the signed median approaches zero');axes[1].grid(alpha=.2)
 fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.045),ncol=2,fontsize=8)
 for ax in axes:ax.axvline(0,color='#555555',ls=':',lw=1)
 fig.savefig(out/'largest_error_trace.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
 primary=result['roles']['calibration']['methods']['causal_residual_tcn'];tail=primary['tails']['above_2_mps'];count=100*tail['endpoint_fraction'];share=100*tail['squared_error_share']
 fig,ax=plt.subplots(figsize=(8,4.1),layout='constrained');labels=['Share of supported endpoints','Share of total squared error'];bars=ax.barh(labels,[count,share],color=['#2878a8','#c43c39']);ax.set(xlim=(0,100),xlabel='Percent of all 5,546 supported calibration endpoints / their squared error',title='Eight errors above 2 m/s dominate aggregate squared error')
 for bar,value in zip(bars,[count,share]):ax.text(value+1,bar.get_y()+bar.get_height()/2,f'{value:.3f}%',va='center')
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);fig.savefig(out/'tail_error_contribution.png',dpi=180);plt.close(fig)
if __name__=='__main__':figures()
