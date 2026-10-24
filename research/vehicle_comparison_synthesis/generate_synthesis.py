"""Generate deterministic comparison outputs; figures use standard plotting."""
import argparse,csv
import synthesis_core as c
def generate():
    c.check_lock();result=c.compute();c.write(c.ROOT/'results.json',result)
    rows=[]
    for population,groups in [('pooled',{'all':result['pooled']}),('route',result['by_route']),('recording',result['by_recording']),('window',result['by_window'])]:
        for label,models in groups.items():
            for name,values in models.items(): rows.append(dict(population=population,label=label,model=name,**values))
    with (c.ROOT/'comparison.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result
def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(c.METHODS);labels=['Affine median','Ridge context','Huber context*','Radar HGB','Radar + IMU','Signed-axis head']
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.5),layout='constrained')
    colors=['#9cabb4']*3+['#487ba5','#c08b4f','#287a69']
    for i,name in enumerate(names):
        axes[0].barh(i,result['pooled'][name]['p95_mps'],color=colors[i])
        d=result['route_mixture_delta_range'][name]['p95_mps'];point=result['pooled'][name]['p95_mps']-result['pooled'][c.REFERENCE]['p95_mps']
        axes[1].plot([d['minimum'],d['maximum']],[i,i],color=colors[i],linewidth=4)
        axes[1].plot(point,i,'o',color=colors[i],markersize=6)
    axes[0].set_yticks(range(6),labels);axes[1].set_yticks(range(6),['']*6)
    for ax in axes: ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].axvline(.5,color='#a53c3c',linestyle='--',linewidth=1);axes[0].set_xlabel('Pooled P95 disagreement (m/s)');axes[0].set_title('Same 27,178 supported rows')
    axes[1].axvline(0,color='#444444',linestyle='--',linewidth=1);axes[1].set_xlabel('P95 change from radar HGB (m/s)');axes[1].set_title('27 route mixtures: extrema and pooled change')
    fig.suptitle('Frozen development comparison; route ranges are not confidence intervals',fontsize=11)
    fig.text(.01,-.025,'* Huber fit reached its recorded iteration limit. Information budgets differ across pipelines.',fontsize=9)
    fig.savefig(c.ROOT/'comparison_sensitivity.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:c.freeze()
    result=generate();plot(result);print('Six-model real-data synthesis complete:',result['cohort_rows'],'rows,',len(result['route_mixtures']),'route mixtures.')
