"""Pure saved-result figures; all layout rendering uses visualization/."""
from .resources import configure
configure()
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import HERE,save
from .cases import prepare_cache
from visualization.layouts import plot_case,geometry_details

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--panel',default='monitor',choices=['monitor','confirmation'])
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before=json.loads(args.baseline.read_text())['records'];after=json.loads(args.candidate.read_text())['records']
    a={tuple(r['member']):r for r in before};b={tuple(r['member']):r for r in after};members=list(b)
    qa=np.array([a[key]['metrics']['cost'] for key in members]);qb=np.array([b[key]['metrics']['cost'] for key in members])
    ta=np.array([a[key]['seconds'] for key in members]);tb=np.array([b[key]['seconds'] for key in members])
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for q,t,label,color in [(qa,ta,'E25 matched','#d97706'),(qb,tb,'Frozen candidate','#4f46e5')]:
        y=np.arange(1,len(q)+1)/len(q)
        axes[0].step(np.sort(q),y,where='post',label=label,color=color)
        axes[1].step(np.sort(t),y,where='post',label=label,color=color)
    axes[0].set(xlabel='Neutral-runtime Q (lower is better)',ylabel='Empirical cumulative probability',ylim=(.9,1.002),title='Quality tail')
    axes[1].set(xlabel='Complete solve seconds',ylabel='Empirical cumulative probability',title='Runtime distribution')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(args.output/'quality_runtime.png',dpi=180);plt.close(fig)
    cases=prepare_cache(args.panel)['rows'];case_map={tuple(r['member']):r['case'] for r in cases}
    selected=[]
    for index in [int(qa.argmax()),int(qb.argmax()),int((qb-qa).argmax())]:
        if index not in selected:selected.append(index)
    index_rows=[]
    for rank,index in enumerate(selected,1):
        key=members[index];c=np.asarray(case_map[key].solver_inputs[5]).tolist();variants={}
        for name,r in [('baseline',a[key]),('candidate',b[key])]:
            variants[name]={'positions':r['positions'],'metrics':r['metrics'],'soft_details':geometry_details(r['positions'],c)}
        filename=f'layout_{rank:02d}.png'
        metadata=plot_case({'panel':args.panel,'case_id':str(key),'constraints':c,'variants':variants},args.output/filename,
            stages=(('baseline',f'E25 matched Q={qa[index]:.4f}'),('candidate',f'Candidate Q={qb[index]:.4f}')))
        index_rows.append({'member':list(key),'baseline_Q':float(qa[index]),'candidate_Q':float(qb[index]),'file':filename,'plot':metadata})
    save(args.output/'index.json',{'layouts':index_rows,'test_read':False,'source_baseline':str(args.baseline),'source_candidate':str(args.candidate)})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));found=False
    for folder in sorted((HERE/'runs').glob('20261022_training_seed*/*')):
        path=folder/'progress.json'
        if not path.exists():continue
        data=json.loads(path.read_text());history=data.get('history',[])
        if not history:continue
        found=True;x=[r['epoch'] for r in history];label=folder.parent.name.split('_')[-1]+'/'+folder.name
        axes[0].plot(x,[r['dev_monitor']['selection_proxy'] for r in history],label=label)
        axes[1].plot(x,[r['dev_monitor']['repair'] for r in history],label=label)
    if found:
        for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.legend(fontsize=7)
        axes[0].set_title('Fixed module-weighted raw selection proxy');axes[1].set_title('Unweighted raw repair proxy')
        fig.tight_layout();fig.savefig(args.output/'training_curves.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
