"""Paired E36 stage-two/confirmation analysis; no optimization or inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E36_persistent_conflict_repair.resources import configure
configure()
import argparse
import json
import sqlite3
import zlib
import numpy as np
from collections import Counter
from experiments.E36_persistent_conflict_repair.io import save,sha

HERE=ROOT/'experiments/E36_persistent_conflict_repair'
NAMES={'reference':'原逻辑','control':'采集对照','chain_repair':'少量邻居耦合平移','pivot_repair':'冲突块对分离方向修正','progress_escape':'按停滞/漂移提前退出'}


def paired(base,other):
    ids=sorted(base);a=[base[i] for i in ids];b=[other[i] for i in ids]
    n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12)
    dq=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)])
    dt=np.array([y['seconds']-x['seconds'] for x,y in zip(a,b)])
    dm=np.array([y['draft_to_final']['mean_center_move']-x['draft_to_final']['mean_center_move'] for x,y in zip(a,b)])
    shards=sorted({r['member'][0] for r in a});lookup={s:k for k,s in enumerate(shards)};g=np.array([lookup[r['member'][0]] for r in a]);rng=np.random.default_rng(20261022)
    sample=rng.integers(0,len(shards),(4000,len(shards)))
    def ci(d,weights):
        numerator=np.bincount(g,weights=weights*d,minlength=len(shards));denominator=np.bincount(g,weights=weights,minlength=len(shards))
        return np.percentile(numerator[sample].sum(1)/denominator[sample].sum(1),[2.5,97.5]).tolist()
    details=[{'index':i,'member':base[i]['member'],'n':base[i]['n'],'baseline_Q':base[i]['metrics']['cost'],'Q':other[i]['metrics']['cost'],
        'delta_Q':float(dq[k]),'delta_seconds':float(dt[k]),'delta_mean_center_move':float(dm[k]),'pivot_calls':other[i].get('pivot_calls',0),
        'weighted_Q_delta_contribution':float(w[k]*dq[k]/w.sum())} for k,i in enumerate(ids)]
    acted=np.array([r.get('pivot_calls',0)>0 for r in b])
    return {'weighted_Q_delta':float(np.average(dq,weights=w)),'weighted_Q_delta_95CI':ci(dq,w),
        'seconds_delta':float(dt.mean()),'seconds_delta_95CI':ci(dt,np.ones(len(ids))),
        'mean_center_move_delta':float(dm.mean()),'mean_center_move_delta_95CI':ci(dm,np.ones(len(ids))),
        'weighted_Q_delta_triggered_contribution':float((w*dq*acted).sum()/w.sum()),
        'weighted_Q_delta_nontriggered_contribution':float((w*dq*(~acted)).sum()/w.sum()),
        'Q_improved_cases_1e_4':int((dq<-1e-4).sum()),'Q_worse_cases_1e_4':int((dq>1e-4).sum()),
        'max_regression':float(dq.max()),'max_improvement':float(dq.min()),
        'worst_cases':sorted(details,key=lambda r:r['delta_Q'],reverse=True)[:8],
        'best_cases':sorted(details,key=lambda r:r['delta_Q'])[:8],
        'largest_weighted_regressions':sorted(details,key=lambda r:r['weighted_Q_delta_contribution'],reverse=True)[:5],
        'shards':len(shards)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='20261022_stage2_paired100');ap.add_argument('--out',default='20261022_stage2_report');args=ap.parse_args()
    folder=HERE/'runs'/args.run;out=HERE/'runs'/args.out;out.mkdir(exist_ok=True)
    raw=json.loads((folder/'summary.json').read_text());config=json.loads((folder/'config.json').read_text());database=folder/'outcomes.sqlite3'
    con=sqlite3.connect('file:'+database.resolve().as_posix()+'?mode=ro',uri=True);records=[json.loads(zlib.decompress(row[0])) for row in con.execute('SELECT payload FROM records')];con.close()
    arms=config['arms'];expected=raw['summaries']['reference']['count'];assert len(records)==len(arms)*expected
    assert all(r['metrics']['is_feasible'] and 'error' not in r for r in records)
    byarm={a:{r['index']:r for r in records if r['arm']==a} for a in arms}
    effects={base:{arm:paired(byarm[base],rr) for arm,rr in byarm.items() if arm!=base} for base in ('reference','control') if base in byarm}
    functions={};acted_cases=[]
    for arm,rr in byarm.items():
        root=[t for r in rr.values() for t in r['stats'].get('telemetry',[]) if t['level']==0]
        pivots=[(i,t) for i,r in rr.items() for t in r['stats']['trace'] if t['phase']=='conflict_pivot']
        chain=[(i,t) for i,r in rr.items() for t in r['stats']['trace'] if t['phase']=='chain_repair']
        functions[arm]={'root_shapes':sum(r['root_shape_calls'] for r in rr.values()),'root_statuses':dict(Counter(t['status'] for t in root)),
            'root_cumulative_mean_center':float(np.mean([r['root_cumulative_center_move'] for r in rr.values()])) if root else None,
            'pivot_calls':len(pivots),'pivot_indices':sorted({i for i,t in pivots}),
            'pivot_retries':sum('retry_after' in t for i,t in pivots),
            'chain_calls':len(chain),'chain_effective_calls':sum(bool(t['moves']) for i,t in chain),
            'chain_moves':sum(len(t['moves']) for i,t in chain),
            'chain_max_move':max([m['block_units'] for i,t in chain for m in t['moves']]+[0.]),
            'escape_indices':sorted(i for i,r in rr.items() if r.get('conflict_escape')),
            'rescue_indices':sorted(i for i,r in rr.items() if r.get('rescue_calls',0)>0)}
        if arm=='pivot_repair':
            for i in functions[arm]['pivot_indices']:
                r=rr[i];base=byarm['reference'][i];root=[t for t in r['stats']['telemetry'] if t['level']==0]
                acted_cases.append({'index':i,'member':r['member'],'n':r['n'],'before_Q':base['metrics']['cost'],'after_Q':r['metrics']['cost'],
                    'before_seconds':base['seconds'],'after_seconds':r['seconds'],'before_root_calls':base['root_shape_calls'],'after_root_calls':r['root_shape_calls'],
                    'before_move':base['draft_to_final']['mean_center_move'],'after_move':r['draft_to_final']['mean_center_move'],
                    'before_metrics':base['metrics'],'after_metrics':r['metrics'],
                    'first_legal_shape_step':next((k+1 for k,t in enumerate(root) if t['hard_legal']),None),
                    'pivots':[t for t in r['stats']['trace'] if t['phase']=='conflict_pivot']})
    resources={r['resource']['pid']:r['resource'] for r in records}
    assert len(resources)<=4 and all(v['max_cpu_execution_slots']==1 and v['torch_intraop']==1 for v in resources.values())
    assert len({tuple(v['affinity']) for v in resources.values()})==len(resources)
    unchanged={name:sha(ROOT/name)==value for name,value in config['sources'].items()};assert all(unchanged.values())
    result={'scope':'same-resource four-process one-thread frozen-model adaptive development comparison; no test/training',
        'summaries':raw['summaries'],'paired':effects,'functions':functions,'pivot_cases':acted_cases,
        'resources':list(resources.values()),'batch_wall_seconds':raw['batch_wall_seconds'],
        'all_sources_unchanged':True,'source_database_sha256':sha(database),'analysis_source_sha256':sha(Path(__file__)),
        'cumulative_elapsed_seconds':raw['cumulative_elapsed_seconds'],'test_read':False,'model_training':False,
        'panel_scope':config.get('panel_scope','fixed adaptive E35 dev100'),
        'nonoverlap_with_discovery':config.get('nonoverlap_with_discovery')}
    save(out/'summary.json',result)
    lines=[f'# E36：{expected}例同资源配对结果','',
        f'共{len(records)}个输出，全部硬合法、错误0。4进程×每进程1数值线程，分别绑定单独CPU槽；复用/生成同一冻结模型draft，求解决策不使用参考几何或Q。主计时为后处理墙钟，进程CPU时间及批量吞吐另列。Q为MIB派生开发中性时间指标，不是比赛分数。',
        f"面板范围：{result['panel_scope']}。本阶段源文件哈希均与运行时一致。批量总墙钟{raw['batch_wall_seconds']:.2f}秒（含初始化/调度/评分/保存），不是单例延迟。",'',
        '| 策略 | 加权Q | 后处理均秒 | P95秒 | CVaR95 Q | 最大Q | 最终平均位移 | 救援例数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in raw['summaries'].items():lines.append(f"| {NAMES[arm]} | {s['weighted_Q']:.6f} | {s['seconds_mean']:.6f} | {s['seconds_p95']:.6f} | {s['cvar95_Q']:.6f} | {s['max_Q']:.6f} | {s['final_mean_center_move']:.4f} | {s['rescue_cases']} |")
    lines+=['','## 配对差与不确定性','', '| 策略对照 | Δ加权Q | 95%区间 | Δ均时秒 | 95%区间 | Q改善/恶化例数（1e-4） |','|---|---:|---|---:|---|---|']
    for base,table in effects.items():
        for arm,e in table.items():lines.append(f"| {arm} vs {base} | {e['weighted_Q_delta']:+.6f} | {e['weighted_Q_delta_95CI']} | {e['seconds_delta']:+.6f} | {e['seconds_delta_95CI']} | {e['Q_improved_cases_1e_4']}/{e['Q_worse_cases_1e_4']} |")
    lines+=['','95%区间为按完整来源分片的4000次配对bootstrap，不是独立测试或多重比较认证。reference只改变同资源线程参数和相对导入，control包含与新策略共同的过程采集；对照两者可识别截止时刻/采集扰动。原两线程阶段不能与本阶段直接作算法耗时对比。','',
        '## 冲突修正的实际作用','', '| index | 块数 | Q前→后 | 后处理秒前→后 | 主流程次数前→后 | 净位移前→后 | 首次合法shape |','|---|---:|---|---|---|---|---:|']
    for r in acted_cases:lines.append(f"| {r['index']} | {r['n']} | {r['before_Q']:.6f}→{r['after_Q']:.6f} | {r['before_seconds']:.4f}→{r['after_seconds']:.4f} | {r['before_root_calls']}→{r['after_root_calls']} | {r['before_move']:.3f}→{r['after_move']:.3f} | {r['first_legal_shape_step']} |")
    lines+=['','只为持续冲突的少量块对改变一个分离选择，并将该关系置为硬分离；其他关系保持弹性。最多两对活跃、四次尝试，若方向导致有向环或与预放置位置直接冲突则跳过。它仍调用原几何子问题，不能写成整个后处理已变成纯平移。','',
        '动作触发与未触发实例对加权Q的贡献分别保存在summary。未触发实例仍可能因采集与墙钟截止改变迭代数，不能将这些微小差异全部归因于关系修正。整体平均与尾部之外，还必须检查逐例显著恶化及新增救援。',
        'CPU时间使用Windows进程CPU计时，存在粒度误差；每例墙钟>两倍CPU时间的数量用于发现严重抢占/停留，不将CPU时间替代实际延迟。第一阶段558秒异常记录未删除，本阶段另记资源和控制，不借更换计时口径抹去异常。','',
        '[精确配对统计/动作/逐例反例](summary.json) · [配对效果图](effects.png)','']
    (out/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(out,result)
    print(json.dumps({'paired_pivot':effects['reference'].get('pivot_repair'),'functions':functions,'pivot_cases':acted_cases,'resources':result['resources']},ensure_ascii=False),flush=True)


def plot(out,result):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=[a for a in result['summaries'] if a not in ('reference','control')];fig,axes=plt.subplots(1,3,figsize=(15,5));y=np.arange(len(arms))
    for ax,key,ci,title in [(axes[0],'weighted_Q_delta','weighted_Q_delta_95CI','Weighted Q change vs matched original'),(axes[1],'seconds_delta','seconds_delta_95CI','Mean postprocessor seconds change')]:
        rr=[result['paired']['reference'][a] for a in arms];v=np.array([r[key] for r in rr]);bounds=np.array([r[ci] for r in rr])
        ax.errorbar(v,y,xerr=np.maximum(0,np.stack((v-bounds[:,0],bounds[:,1]-v))),fmt='o',capsize=4);ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,arms);ax.set_title(title);ax.grid(alpha=.2)
    for a in ['reference']+arms:
        s=result['summaries'][a];axes[2].scatter(s['cvar95_Q'],s['max_Q'],label=a)
    axes[2].set(xlabel='CVaR95 Q',ylabel='Maximum Q',title='Quality tail; lower is better');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'effects.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
