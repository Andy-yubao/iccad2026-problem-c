"""Render saved experiment evidence only; no datasets or model execution."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

NAMES=['baseline_mse','baseline_geometry','attention_mse','attention_geometry']
LABELS={'baseline_mse':'GNN / MSE','baseline_geometry':'GNN / geometry',
        'attention_mse':'Attention / MSE','attention_geometry':'Attention / geometry','legacy_frozen':'Legacy frozen',
        'baseline_local':'GNN / local supervision'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();run=args.run_dir;out=run/'evaluation';curves={}
    results=json.loads((out/'results.json').read_text(encoding='utf-8'))
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    train_names=[*NAMES,*[n for n in results['variants'] if n not in [*NAMES,'legacy_frozen']]]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name in train_names:
        folder=Path(config['checkpoints'][name]['path']).parents[1]
        rows=[json.loads(line) for line in (folder/'run.log').read_text(encoding='utf-8').splitlines()]
        monitors=[r for r in rows if r['phase']=='monitor'];curves[name]=monitors
        for ax,key in zip(axes.flat,['geometry','center_mse','shape_mse','overlap']):
            ax.plot([r['variant_seconds']/60 for r in monitors],[r['dev'][key] for r in monitors],label=LABELS[name])
            ax.set_xlabel('Variant wall time (min)');ax.set_ylabel('Fixed dev500: '+key);ax.grid(alpha=.25)
            ax.set_yscale('log')
    axes[0,0].legend(fontsize=8);fig.suptitle('Geometry-only architecture x loss controls; seed 0; equal time caps')
    fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    names=[*train_names,'legacy_frozen'];variants=results['variants']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    specs=[('Raw overlap pair area / block area',lambda v:v['raw']['diagnostics']['pair_overlap_area_fraction']['mean']),
           ('Raw mean center error / local size',lambda v:v['raw']['diagnostics']['center_error_local_mean']['mean']),
           ('Optimistic movement LB with 5% shape (median)',lambda v:v['raw']['diagnostics']['translation_radius_lb_shape5pct']['median']),
           ('Final neutral-runtime weighted Q',lambda v:v['final']['weighted_Q']),
           ('Final mean soft violation rate',lambda v:v['final']['violations_relative_mean']),
           ('Final mean solve seconds',lambda v:v['final']['solve_seconds_mean'])]
    colors=['#577590','#90be6d','#f8961e','#f94144','#43aa8b','#999999']
    for ax,(title,fn) in zip(axes.flat,specs):
        values=[fn(variants[name]) for name in names]
        ax.bar(range(len(names)),values,color=colors)
        ax.set_xticks(range(len(names)),[LABELS[n].replace(' / ','\n') for n in names],fontsize=8)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
        if title.startswith('Optimistic'):
            ax.axhline(.05,color='#a33',linestyle='--',linewidth=1,label='5% position budget')
            ax.legend(fontsize=7)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.margins(y=.15)
    fig.suptitle('Fixed dev_quality100; raw layouts are mostly infeasible; legacy is historical')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    rows=[json.loads(line) for line in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    chosen=[r for r in rows if r['source']['shard'].split('/')[0] in {'worker_0','worker_24','worker_49','worker_74','worker_99'}]
    gallery=[]
    for row in chosen:
        worker=row['source']['shard'].split('/')[0]
        for stage in ('raw','final'):
            layouts=[row['reference'],*[row['variants'][name][stage]['positions'] for name in names]]
            extent=np.concatenate([np.asarray(l)[:,:2]+np.asarray(l)[:,2:] for l in layouts]).max(0)*1.04
            columns=(len(layouts)+1)//2
            fig,axes=plt.subplots(2,columns,figsize=(5*columns,10),constrained_layout=True)
            titles=['Reference',*[LABELS[name] for name in names]]
            for ax,layout,title in zip(axes.flat,layouts,titles):
                a=np.asarray(layout);affected=set()
                constraints=np.asarray(row['public_constraints'])
                for i in range(len(a)):
                    overlap=np.minimum(a[i,:2]+a[i,2:],a[:i,:2]+a[:i,2:])-np.maximum(a[i,:2],a[:i,:2])
                    for j in np.flatnonzero((overlap>1e-6).all(1)):affected.update((i,int(j)))
                for i,(x,y,w,h) in enumerate(a):
                    group=int(constraints[i,3]);mib=int(constraints[i,2])
                    color=plt.cm.tab10((group-1)%10) if group else '#dddddd'
                    hatch=['/','\\','x','-','+','o'][(mib-1)%6] if mib else None
                    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,alpha=.55,hatch=hatch,
                                           edgecolor='red' if i in affected else '#333333',
                                           linewidth=1.5 if constraints[i,1] else 1. if constraints[i,0] else .65))
                    ax.text(x+w/2,y+h/2,str(i),fontsize=4,ha='center',va='center')
                ax.set_xlim(0,extent[0]);ax.set_ylim(0,extent[1]);ax.set_aspect('equal');ax.set_title(title,fontsize=10)
                if title!='Reference':
                    name=names[titles.index(title)-1];m=row['variants'][name][stage]['metrics']
                    ax.set_xlabel(f"hard={m['is_feasible']} Q={m['cost']:.3f} B/G/M={m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']}",fontsize=8)
            for ax in list(axes.flat)[len(layouts):]:ax.axis('off')
            fig.suptitle(f'{worker}: {stage}; same scale; group=color; MIB=hatch; red=overlap; thick=locked specs')
            filename=f'{worker}_{stage}.png';fig.savefig(out/filename,dpi=180);plt.close(fig)
            gallery.append((worker,stage,filename))
    html=['<!doctype html><meta charset="utf-8"><title>E6 geometry-only results</title>',
          '<style>body{max-width:1400px;margin:30px auto;font:16px system-ui}img{width:100%}a{color:#146}section{margin:40px 0}</style>',
          '<h1>E6: geometry-only architecture × loss</h1><p>Seed 0; fixed train/dev workset; no test access. Raw layouts are not submission scores.</p>',
          '<p><a href="analysis.md">Analysis</a> · <a href="results.json">Exact results</a> · <a href="supervised.json">Full train/dev metrics</a></p>',
          '<img src="comparison.png"><img src="training_curves.png">']
    for worker,stage,filename in gallery:html.append(f'<section><h2>{worker}: {stage}</h2><a href="{filename}"><img src="{filename}"></a></section>')
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print(json.dumps({'plots':2+len(gallery),'gallery_cases':len(chosen),'output':str(out/'index.html')}))


if __name__=='__main__':main()
