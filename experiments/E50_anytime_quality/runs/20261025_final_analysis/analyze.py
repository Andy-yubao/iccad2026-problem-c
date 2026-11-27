"""Saved-result analysis and Q/time/draft-distance frontier; never invoke a solver."""
import os
for k in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,csv,math
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
SOURCE=HERE.parent/'20261025_frozen100_parallel_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert read(SOURCE/'execution.json')['complete'],'Require complete primary100'
    config=read(SOURCE/'config.json');rows=[read(SOURCE/'cases'/f'{i:04d}.json') for i in config['indices']]
    from datasets.reference_layouts import ReferenceLayouts
    from experiments.E50_anytime_quality.archive import Archive
    from experiments.E50_anytime_quality.backend.frozen.packing import prepare_problem
    from experiments.E50_anytime_quality.selection import choose
    from visualization.layouts import plot_case,geometry_details
    dataset=ReferenceLayouts('dev_monitor500')
    old={r['index']:r for r in read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json')['records']}
    table=[];fronts=[];curves={t:[] for t in(.1,.25,.5,1.,2.,3.,4.,4.6)}
    for r in rows:
        i=r['index'];item=dataset[i];assert list(item['member'])==r['member']==old[i]['member']
        p=prepare_problem(*item['case'].solver_inputs);arc=Archive(p,np.asarray(r['draft']),0.,None)
        cs=r['solver']['candidates'];chosen=next(c for c in cs if c['id']==r['solver']['selected_id']);best=min(cs,key=lambda c:c['metrics']['cost'])
        od=arc.distance(np.asarray(old[i]['positions']));hit=[c for c in cs if c['metrics']['cost']<1.03]
        values=np.array([[c['metrics']['cost'],c['distance']['D'],c['checked_seconds']] for c in cs])
        nondominated=[]
        for j,c in enumerate(cs):
            if not np.any(np.all(values<=values[j]+[1e-9,1e-9,1e-6],axis=1)&np.any(values<values[j]-[1e-9,1e-9,1e-6],axis=1)):nondominated.append(c['id'])
        fronts.append({'index':i,'candidate_ids':nondominated,'coordinates_source':str((SOURCE/'cases'/f'{i:04d}.json').relative_to(ROOT)).replace('\\','/'),
            'scope':'posthoc Q/D/checked-time candidate frontier, not online selection or an optimality bound'})
        for t in curves:
            prefix=[c for c in cs if c['checked_seconds']<=t]
            if not prefix:continue
            baselines=[c for c in prefix if c['baseline_candidate']];base=min(baselines,key=lambda c:c['native_proxy']) if baselines else None
            pick,_=choose(prefix,base,arc.obj.area_scale,arc.draft_wire,arc.obj.nsoft)
            curves[t].append({'index':i,'n':r['n'],'Q':pick['metrics']['cost'],'D':pick['distance']['D']})
        table.append({'index':i,'n':r['n'],'baseline_Q':r['baseline_Q'],'selected_Q':r['selected_metrics']['cost'],
            'Q_delta':r['selected_metrics']['cost']-r['baseline_Q'],'selected_D':chosen['distance']['D'],'baseline_D':od['D'],
            'full_seconds':r['full_seconds'],'offline_candidate_Q_seconds':r['offline_diagnostics_seconds'],
            'first_found_Q_lt1_03_seconds':min((c['checked_seconds'] for c in hit),default=None),
            'min_found_D_for_Q_lt1_03':min((c['distance']['D'] for c in hit),default=None),
            'posthoc_best_Q':best['metrics']['cost'],'selected_candidate_id':chosen['id'],'posthoc_best_id':best['id'],
            'candidate_count':len(cs),'worker_abort_count':len(r['solver']['worker_aborts']),
            'mean_center_local':chosen['distance']['mean_center_local'],'p95_center_local':chosen['distance']['p95_center_local'],
            'shape_change':chosen['distance']['mean_abs_log_aspect_change'],'relation_change':chosen['distance']['near_relation_change_fraction']})
    with (HERE/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    q=np.array([r['selected_Q'] for r in table]);b=np.array([r['baseline_Q'] for r in table]);n=np.array([r['n'] for r in table]);weights=np.exp((n-n.max())/12)
    d=np.array([r['selected_D'] for r in table]);od=np.array([r['baseline_D'] for r in table]);seconds=np.array([r['full_seconds'] for r in table]);groups={}
    for j,r in enumerate(rows):groups.setdefault(r['member'][0].split('/')[0],[]).append(j)
    keys=list(groups);rng=np.random.default_rng(20261025);boot=[]
    for _ in range(2000):
        ids=np.concatenate([groups[keys[k]] for k in rng.integers(len(keys),size=len(keys))]);boot.append(float(np.average((q-b)[ids],weights=weights[ids])))
    replay=[]
    for t,points in curves.items():
        full=len(points)==100
        replay.append({'time_cut_seconds':t,'coverage':len(points),'weighted_Q_full_panel':float(np.average([r['Q'] for r in points],weights=np.exp((np.array([r['n'] for r in points])-120)/12))) if full else None,
            'note':'prefix selection replay from a five-second search; not a new short-budget run; missing cases never dropped from a claimed full-panel score'})
    result={'count':100,'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in rows),
        'weighted_Q':float(np.average(q,weights=weights)),'baseline_weighted_Q':float(np.average(b,weights=weights)),
        'weighted_Q_delta_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
        'excess_Q_reduction_fraction':float((np.average(b,weights=weights)-np.average(q,weights=weights))/(np.average(b,weights=weights)-1)),
        'mean_Q':float(q.mean()),'baseline_mean_Q':float(b.mean()),'p95_Q':float(np.quantile(q,.95)),
        'baseline_p95_Q':float(np.quantile(b,.95)),'max_Q':float(q.max()),'baseline_max_Q':float(b.max()),
        'paired_win_tie_loss':[int(sum(q<b-1e-8)),int(sum(np.abs(q-b)<=1e-8)),int(sum(q>b+1e-8))],
        'regression_indices':[r['index'] for r in table if r['Q_delta']>1e-8],'over1_2_indices':[r['index'] for r in table if r['selected_Q']>1.2],
        'Q_lt1_03':int(sum(q<1.03)),'mean_seconds':float(seconds.mean()),'max_seconds':float(seconds.max()),'over5s':int(sum(seconds>5)),
        'selected_D_median':float(np.median(d)),'selected_D_p95':float(np.quantile(d,.95)),
        'baseline_D_median':float(np.median(od)),'baseline_D_p95':float(np.quantile(od,.95)),
        'closer_to_draft_cases':int(sum(d<od-1e-9)),'candidate_count':sum(r['candidate_count'] for r in table),
        'total_offline_candidate_Q_seconds':sum(r['offline_candidate_Q_seconds'] for r in table),
        'worker_abort_cases':sum(r['worker_abort_count']>0 for r in table),
        'passed':bool((q<=b+1e-8).all() and np.average(q,weights=weights)<1.015 and sum(q>1.2)<=1 and (seconds<=5).all()),
        'scope':'adaptive fixed development100, no sealed test; bootstrap excludes selection bias and run-to-run numerical timing variation',
        'candidate_frontiers':fronts,'prefix_replays':replay,'solver_or_inference_calls':0}
    save(HERE/'analysis.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    sc0=axes[0].scatter(b,q,c=n,s=23,cmap='viridis');fig.colorbar(sc0,ax=axes[0],label='Modules',shrink=.85);limit=max(float(b.max()),float(q.max()),1.2);axes[0].plot([1,limit],[1,limit],'--',color='#87928f')
    axes[0].set(xlabel='Saved ordinary Q',ylabel='E50 selected Q',title='Paired Q: fixed100')
    sc1=axes[1].scatter(d,q,c=seconds,s=23,cmap='plasma');fig.colorbar(sc1,ax=axes[1],label='Seconds',shrink=.85);axes[1].axhline(1.2,color='#ac5d47',linestyle='--');axes[1].set(xlabel='Distance from draft D',ylabel='Selected Q',title='Q vs. distance from draft')
    axes[2].scatter(n,seconds,s=23,color='#267c71');axes[2].axhline(5,color='#ac5d47',linestyle='--');axes[2].set(xlabel='Modules',ylabel='Full solve seconds',title='End-to-end solve time')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(HERE/'quality_distance_runtime.png',dpi=160);plt.close(fig)
    (HERE/'images').mkdir(exist_ok=True)
    examples=list(dict.fromkeys(result['over1_2_indices']+[max(table,key=lambda r:r['selected_Q'])['index'],21]))
    for i in examples:
        r=next(r for r in rows if r['index']==i);chosen=next(c for c in r['solver']['candidates'] if c['id']==r['solver']['selected_id']);c=np.asarray(dataset[i]['case'].solver_inputs[5]).tolist()
        variants={'raw':{'positions':r['draft']},'old':{'positions':old[i]['positions'],'metrics':old[i]['metrics']},'new':{'positions':chosen['positions'],'metrics':chosen['metrics']}}
        for v in variants.values():v['soft_details']=geometry_details(v['positions'],c)
        plot_case({'panel':'E50 frozen100','case_id':f'dev index{i}','constraints':c,'variants':variants,
            'caption':'Frozen E42 draft and identical public inputs. Search/selection never receive true Q.'},HERE/'images'/f'case_{i:04d}.png',
            stages=(('raw','Frozen E42 draft'),('old','Saved ordinary postprocessing'),('new','E50 selected solution')))
    lines=['# E50固定100例实测报告','',f"验收状态：{'通过' if result['passed'] else '未通过'}。本报告来自实际端到端运行，非候选库事后择优。固定开发100经过自适应优化，不是封存测试或官方比赛成绩。",'',
        '| 指标 | E50 | 同例原常规结果 |','|---|---:|---:|',
        f"| 加权Q | {result['weighted_Q']:.8f} | {result['baseline_weighted_Q']:.8f} |",
        f"| 普通均值Q | {q.mean():.8f} | {b.mean():.8f} |",
        f"| P95 Q | {result['p95_Q']:.6f} | {result['baseline_p95_Q']:.6f} |",
        f"| 最差Q | {q.max():.6f} | {b.max():.6f} |",'',
        f"硬合法{result['hard_legal']}/100；逐例胜/平/负{result['paired_win_tie_loss']}（容差1e-8）；Q>1.2案例{result['over1_2_indices']}。加权超额Q下降{result['excess_Q_reduction_fraction']:.1%}。完整均时{seconds.mean():.6f}秒，最大{seconds.max():.6f}秒，超5秒{result['over5s']}例。",'',
        '![质量、距离和完整时间](quality_distance_runtime.png)','',
        f"D中位数{np.median(d):.4f}、P95 {np.quantile(d,.95):.4f}；同原draft重算常规解分别为{np.median(od):.4f}/{np.quantile(od,.95):.4f}，{result['closer_to_draft_cases']}/100例更近。D是固定工程偏好，未用人工评分校准；组成分量见paired_metrics.csv。",'',
        '## 预算与评价边界','',
        f"5秒包含公开输入转换、CPU推理、两条2线程搜索、全部候选独立硬检查、调度/回退/终止及最终解官方评分。冷启动、模型载入、预热/必要重启和磁盘写入单列；全部未选中候选的事后Q诊断另耗时{result['total_offline_candidate_Q_seconds']:.3f}秒，不属于求解入口，亦不反馈搜索。旧普通运行并发口径不同，不将旧计时用于正式提速宣称。",'',
        '原E42权重、Loss、普通后处理和冻结索引保持原样。搜索只接收公开输入和draft；真实Q与参考分母只在父进程评分。新的公开代理不具备任意未知参考分母下的数学不退化保证，逐例不退化由本轮实际记录验证。', '',
        f"worker聚类配对重采样95%加权Q差区间{result['weighted_Q_delta_bootstrap95']}；不包含算法选择偏差或时间截断的重复运行噪声。",'',
        '冻结后另100开发例及五难例三次重复的完整结果见[确认与稳定性](confirmation_stability.md)。主100通过；确认100存在5例小回退，不将主100结论扩展成任意新输入逐例支配。','',
        '## 帕累托记录','',
        'analysis.json保留每例Q/D/核验完成时间的非支配候选ID，坐标只存在原cases中。prefix_replays是5秒轨迹的截断回放，不是新的短预算运行；未满100例覆盖时不生成全100加权Q。算法级前沿与稳定性仍需同配置重复运行及固定确认面板，不能将事后oracle或每例不同算法拼成统一后处理器。','',
        '## 难例草稿与结果','']
    for i in examples:lines.extend([f'![dev {i}: draft / ordinary / E50](images/case_{i:04d}.png)',''])
    (HERE/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in('candidate_frontiers','prefix_replays')})
if __name__=='__main__':main()
