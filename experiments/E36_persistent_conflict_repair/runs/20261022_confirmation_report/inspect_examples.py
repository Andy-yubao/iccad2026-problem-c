"""Saved-layout success/regression figures and time/quality trajectories; no solves."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E36_persistent_conflict_repair.resources import configure
configure()
import json
import sqlite3
import zlib
import os
import numpy as np
from experiments.E36_persistent_conflict_repair.io import sha,save
from experiments.E36_persistent_conflict_repair.cases import public_cases
from scoring.evaluation import evaluate_case
from visualization.layouts import plot_case,geometry_details

OUT=Path(__file__).resolve().parent;FOLDER=OUT.parent/'20261022_confirmation500'


def main():
    con=sqlite3.connect('file:'+(FOLDER/'outcomes.sqlite3').resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(row[0])) for row in con.execute('SELECT payload FROM records')]
    drafts={i:json.loads(zlib.decompress(blob)) for i,blob in con.execute('SELECT idx,payload FROM drafts')};con.close()
    byarm={arm:{r['index']:r for r in records if r['arm']==arm} for arm in ('reference','control','pivot_repair')}
    summary=json.loads((OUT/'summary.json').read_text());best=summary['paired']['reference']['pivot_repair']['best_cases'][0]['index']
    cases,metadata,_=public_cases('confirmation');selected=[best,179,237];figures=[]
    for i in selected:
        case=cases[i];assert list(metadata[i]['member'])==drafts[i]['member'];c=np.asarray(case.solver_inputs[5]);variants={}
        draft=drafts[i]['draft'];m=evaluate_case(case,draft,0.)['metrics']
        variants['draft']={'positions':draft,'metrics':m,'soft_details':geometry_details(draft,c)}
        for tag,arm in [('baseline','reference'),('pivot','pivot_repair')]:
            r=byarm[arm][i];variants[tag]={'positions':r['positions'],'metrics':r['metrics'],'soft_details':geometry_details(r['positions'],c)}
        name=f'case_{i}_'+('success' if i==best else 'regression')+'.png'
        plot_case({'panel':f'E36 confirmation index {i}','case_id':str(metadata[i]['member']),'constraints':c.tolist(),'variants':variants,
            'caption':'Same input and coordinate scale. Compare quality between final outputs; illegal draft is shown only for geometric context.'},
            OUT/name,stages=(('draft','Frozen model draft'),('baseline','Original postprocessor'),('pivot','Conflict relation repair')))
        figures.append({'index':i,'member':metadata[i]['member'],'file':name,'baseline_Q':byarm['reference'][i]['metrics']['cost'],'pivot_Q':byarm['pivot_repair'][i]['metrics']['cost']})
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for arm,label in [('control','Original path (instrumented)'),('pivot_repair','Conflict relation repair')]:
        r=byarm[arm][179];steps=[t for t in r['stats']['telemetry'] if t['level']==0];x=np.arange(1,len(steps)+1)
        axes[0].plot(x,[s.get('max_penetration',np.nan) for s in steps],marker='.',label=label)
        axes[1].plot(x,[s.get('mean_center_move',np.nan) for s in steps],marker='.',label=label)
    axes[0].set(xlabel='Root shape step',ylabel='Maximum overlap penetration / block length',title='Regression179: improving path was interrupted')
    axes[1].set(xlabel='Root shape step',ylabel='Mean step displacement / block length',yscale='log',title='Wrong relation can cause a new plateau')
    a=byarm['reference'];b=byarm['pivot_repair'];ids=sorted(a);n=[a[i]['n'] for i in ids];dq=[b[i]['metrics']['cost']-a[i]['metrics']['cost'] for i in ids]
    active=[b[i]['pivot_calls']>0 for i in ids]
    for value,label,color in [(False,'No pivot action','#94a3b8'),(True,'Pivot triggered','#e07632')]:
        idx=[k for k,v in enumerate(active) if v==value];axes[2].scatter(np.array(n)[idx],np.array(dq)[idx],label=label,color=color,s=22,alpha=.75)
    axes[2].axhline(0,color='black',lw=.8);axes[2].set(xlabel='Blocks',ylabel='Q change vs original',title='All500 cases; negative is better')
    for i in [best,179,237]:axes[2].annotate(str(i),(a[i]['n'],b[i]['metrics']['cost']-a[i]['metrics']['cost']))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'progress_and_regressions.png',dpi=170);plt.close(fig)
    data={'figures':figures,'no_new_solves':True,'source_database_sha256':sha(FOLDER/'outcomes.sqlite3'),'script_sha256':sha(Path(__file__))}
    save(OUT/'examples.json',data);print(json.dumps(data,ensure_ascii=False))


if __name__=='__main__':main()
