"""Summarize frozen E1 outputs and logged training; never reads dataset tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def train_record(folder):
    result, config = read(folder/'results.json'), read(folder/'config.json')
    logs = [json.loads(line) for line in (folder/'run.log').read_text().splitlines() if line.strip()]
    prior = config['invocations'][-1]['prior_accumulated_seconds']
    windows = []
    for low in range(0, 420, 60):
        values = [r['loss'] for r in logs if low <= r['accumulated_seconds']-prior < low+60]
        if values:
            windows.append({'invocation_seconds':[low,low+60], 'logged_batches':len(values),
                            'median_loss':statistics.median(values), 'min_loss':min(values), 'max_loss':max(values)})
    return {**result, 'folder':str(folder), 'prior_attempt_seconds':prior,
            'invocation_seconds':result['accumulated_seconds']-prior,
            'logged_loss_windows':windows, 'training_source_sha256':config['identity']['source_sha256'],
            'caveat':'logged mini-batches differ in graphs; flow also samples noise and time; loss is not validation quality'}, logs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--det-results',type=Path,required=True)
    parser.add_argument('--flow-results',type=Path,required=True)
    parser.add_argument('--det-train',type=Path,required=True)
    parser.add_argument('--flow-train',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    det,flow=read(args.det_results),read(args.flow_results)
    assert [c['case_id'] for c in det['cases']]==[c['case_id'] for c in flow['cases']]
    names=['heuristic','deterministic','flow1','flow4']
    variants={n:[] for n in names}
    summaries={**det['summary'],**flow['summary']}
    rows=[]
    for a,b in zip(det['cases'],flow['cases']):
        merged={**a['variants'],**b['variants']}
        row={'case_id':a['case_id'],'block_count':a['input_summary']['block_count'],'variants':{}}
        for n in names:
            v=merged[n]; variants[n].append(v)
            stats=v.get('last_stats',{})
            ds=stats.get('candidate_diagnostics',[])
            chosen=ds[stats['chosen_candidate']] if ds else None
            r={'Q':v['metrics']['cost'],'hard_legal':v['metrics']['is_feasible'],
               'solve_seconds':v.get('solve_seconds'),'fallback':stats.get('fallback'),
               'candidate_count':stats.get('candidate_count'), 'chosen_candidate':stats.get('chosen_candidate')}
            if chosen:
                for s in ('raw','corrected'):
                    assessed=chosen[s+'_assessment']
                    r[s]={'metrics':assessed['metrics'],
                          'overlap_pair_count':len(assessed['independent_hard_checks']['overlap_pairs'])}
                for k in ('continuous_displacement_mean','final_displacement_mean','continuous_seconds','geometry_seconds'):
                    r[k]=chosen[k]
            row['variants'][n]=r
        rows.append(row)
    for n in names:
        vs=variants[n]
        summaries[n]['fallback_count']=sum(bool(v.get('last_stats',{}).get('fallback')) for v in vs)
        summaries[n]['wins_vs_heuristic']=sum(v['metrics']['cost']<h['metrics']['cost']-1e-9 for v,h in zip(vs,variants['heuristic']))
        summaries[n]['mean_solve_seconds']=sum(v.get('solve_seconds',0) for v in vs)/len(vs)
        selected=[r['variants'][n] for r in rows if 'raw' in r['variants'][n]]
        for k in ('continuous_displacement_mean','final_displacement_mean'):
            summaries[n]['mean_'+k]=statistics.mean(r[k] for r in selected) if selected else None
        summaries[n]['mean_raw_overlap_pairs']=statistics.mean(r['raw']['overlap_pair_count'] for r in selected) if selected else None
        summaries[n]['mean_corrected_overlap_pairs']=statistics.mean(r['corrected']['overlap_pair_count'] for r in selected) if selected else None
    train_det,logs_det=train_record(args.det_train)
    train_flow,logs_flow=train_record(args.flow_train)
    q=lambda n:summaries[n]['neutral_runtime_weighted_cost_all_cases']
    eligible=[n for n in names[1:] if summaries[n]['feasible_count']==5 and q(n)<q('heuristic') and summaries[n]['wins_vs_heuristic']>=3 and summaries[n]['fallback_count']==0]
    best=min(eligible,key=q) if eligible else None
    report={'scope':'same frozen dev5; seed0; CPU1 inference; 15s/case ceiling; training times unequal; not generalization or contest score',
            'source_results':{'det':str(args.det_results),'flow':str(args.flow_results)},
            'training':{'deterministic':train_det,'flow':train_flow},'summary':summaries,'cases':rows,
            'eligible_vs_common_heuristic':eligible,'best_eligible':best,
            'flow4_wins_vs_flow1':sum(b['metrics']['cost']<a['metrics']['cost']-1e-9 for a,b in zip(variants['flow1'],variants['flow4'])),
            'training_budget_note':'Deterministic is the user-stopped existing checkpoint; flow is one new 300s run. No architecture-causality claim.',
            'historical_baseline_Q':1.9090193724345106,'baseline_note':'Historical search reference only; not rerun under current resources.',
            'local_test_read':False,'convergence_claim':False}
    (args.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=['#718096','#2563eb','#e58b14','#9f4ccf']
    bars=axes[0,0].bar(names,[q(n) for n in names],color=colors)
    axes[0,0].bar_label(bars,fmt='%.3f',padding=3)
    axes[0,0].set(title='Frozen dev5 weighted quality (lower is better)',ylabel='Neutral-runtime Q')
    for n,color in zip(names,colors):
        axes[0,1].plot(range(len(rows)),[r['variants'][n]['Q'] for r in rows],marker='o',label=n,color=color)
    axes[0,1].set_xticks(range(len(rows)),[str(r['block_count']) for r in rows])
    axes[0,1].set(title='Per-case quality (frozen panel order)',xlabel='Block count',ylabel='Q')
    axes[0,1].legend(fontsize=8)
    for ax,label,logs,training,color in zip(axes[1],['Deterministic','Flow'],[logs_det,logs_flow],[train_det,train_flow],colors[1:3]):
        xs=[r['accumulated_seconds']-training['prior_attempt_seconds'] for r in logs]
        ys=[r['loss'] for r in logs]
        smooth=[statistics.median(ys[max(0,i-10):i+11]) for i in range(len(ys))]
        ax.scatter(xs,ys,s=6,alpha=.23,color=color,label='Logged batch loss')
        ax.plot(xs,smooth,color=color,label='21-point rolling median')
        ax.set(title=label+' training (different loss definitions)',xlabel='Invocation elapsed seconds, including startup',ylabel='Training loss')
        ax.legend(fontsize=8)
    fig.suptitle('E1 short-budget screen: 5 cases, seed 0, no convergence claim')
    fig.savefig(args.output/'comparison.png',dpi=150)
    plt.close(fig)
    config={'command':__import__('sys').argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.det_results,args.flow_results)},
            'dataset_tensor_reads':0}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    (args.output/'summarize_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'eligible':eligible,'best':best,'Q':{n:q(n) for n in names}},indent=2))


if __name__=='__main__':main()
