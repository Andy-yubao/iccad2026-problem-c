"""Rebuild the E48 report from saved candidates only; no solving or inference."""
from pathlib import Path
import sys,os,json,csv,hashlib
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.E48_quality_potential.run import load_record
from experiments.E48_quality_potential.solver import Archive
from experiments.E48_quality_potential.backend.frozen.packing import prepare_problem
from datasets.reference_layouts import ReferenceLayouts
from visualization.layouts import plot_case,geometry_details

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    config=read(HERE/'config.json');execution=read(HERE/'execution.json')
    if not execution['complete']:raise RuntimeError('This report requires the completed frozen100; do not mistake partial output for final results')
    rows=[load_record(HERE/'cases'/f'{i:04d}.json') for i in config['indices']]
    dataset=ReferenceLayouts('dev_monitor500')
    baseline={r['index']:r for r in read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json')['records']}
    baseline_d=[]
    for r in rows:
        item=dataset[r['index']];old=baseline[r['index']]
        assert list(item['member'])==r['member']==old['member']
        archive=Archive(prepare_problem(*item['case'].solver_inputs),np.asarray(r['raw_positions']),0.,None)
        baseline_d.append(archive.distance(np.asarray(old['positions'])))
    q=np.array([r['selected_metrics']['cost'] for r in rows]);base=np.array([r['baseline']['Q'] for r in rows]);n=np.array([r['n'] for r in rows]);t=np.array([r['full_solve_seconds'] for r in rows])
    d=np.array([r['solver']['candidates'][r['solver']['selected_id']]['distance']['D'] for r in rows])
    oracle=np.array([min(c['metrics']['cost'] for c in r['solver']['candidates']) for r in rows])
    table=[];target=[];actual_frontier=[]
    for r,old_distance in zip(rows,baseline_d):
        chosen=r['solver']['candidates'][r['solver']['selected_id']];candidates=r['solver']['candidates']
        hits=[c for c in candidates if c['metrics']['cost']<1.03 and not c['shelf_fallback']]
        public_best=candidates[r['solver']['public_quality_best_id']];best=min(candidates,key=lambda c:c['metrics']['cost'])
        table.append({'index':r['index'],'member':':'.join(map(str,r['member'])),'n':r['n'],'baseline_Q':r['baseline']['Q'],
            'selected_Q':chosen['metrics']['cost'],'posthoc_best_Q':best['metrics']['cost'],'public_best_Q':public_best['metrics']['cost'],
            'selected_D':chosen['distance']['D'],'baseline_D':old_distance['D'],'D_delta':chosen['distance']['D']-old_distance['D'],
            'posthoc_best_D':best['distance']['D'],
            'min_found_D_Q_lt1_03':min((c['distance']['D'] for c in hits),default=None),
            'first_found_full_seconds_Q_lt1_03':min((c['score_complete_seconds'] for c in hits),default=None),
            'first_non_shelf_legal_full_seconds':r['inference_seconds']+r['solver']['first_non_shelf_legal_seconds'] if r['solver']['first_non_shelf_legal_seconds'] is not None else None,
            'full_seconds':r['full_solve_seconds'],'candidate_count':len(candidates),
            'selected_mean_center_local':chosen['distance']['mean_center_local'],'selected_p95_center_local':chosen['distance']['p95_center_local'],
            'selected_max_center_over_chip_diagonal':chosen['distance']['max_center_over_chip_diagonal'],
            'selected_mean_abs_log_aspect_change':chosen['distance']['mean_abs_log_aspect_change'],
            'selected_near_relation_change_fraction':chosen['distance']['near_relation_change_fraction'],'selected_source':chosen['source']})
        target.append(bool(hits))
        front=[c for c in candidates if not any(other['metrics']['cost']<=c['metrics']['cost'] and other['distance']['D']<=c['distance']['D'] and
            (other['metrics']['cost']<c['metrics']['cost']-1e-9 or other['distance']['D']<c['distance']['D']-1e-9) for other in candidates)]
        actual_frontier.append({'index':r['index'],'ids':[c['id'] for c in front],
                               'scope':'posthoc Q-D frontier; positions remain once in the original candidate record'})
    with (HERE/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    weights=np.exp((n-n.max())/12);groups={}
    for i,r in enumerate(rows):groups.setdefault(r['member'][0].split('/')[0],[]).append(i)
    rng=np.random.default_rng(20261024);keys=list(groups);boot=[]
    for _ in range(1000):
        ids=np.concatenate([groups[keys[k]] for k in rng.integers(len(keys),size=len(keys))])
        boot.append(float(np.average((q-base)[ids],weights=weights[ids])))
    gap=q-oracle
    result={'count':len(rows),'weighted_Q':float(np.average(q,weights=weights)),'baseline_weighted_Q':float(np.average(base,weights=weights)),
        'paired_weighted_Q_delta':float(np.average(q-base,weights=weights)),
        'paired_weighted_Q_delta_worker_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
        'pair_win_tie_loss':[int(sum(q<base-1e-8)),int(sum(np.abs(q-base)<=1e-8)),int(sum(q>base+1e-8))],
        'selected_Q_lt1_03':int(sum(q<1.03)),'posthoc_any_Q_lt1_03':int(sum(oracle<1.03)),
        'max_selected_Q':float(q.max()),'p95_selected_Q':float(np.quantile(q,.95)),
        'posthoc_max_best_Q':float(oracle.max()),'mean_selection_gap_Q':float(gap.mean()),'max_selection_gap_Q':float(gap.max()),
        'mean_full_seconds':float(t.mean()),'max_full_seconds':float(t.max()),'over5s_count':int(sum(t>5)),
        'selected_D_median':float(np.median(d)),'selected_D_p95':float(np.quantile(d,.95)),
        'baseline_D_median':float(np.median([v['D'] for v in baseline_d])),
        'baseline_D_p95':float(np.quantile([v['D'] for v in baseline_d],.95)),
        'paired_mean_D_delta':float(np.mean(d-np.array([v['D'] for v in baseline_d]))),
        'closer_D_cases':int(sum(d<np.array([v['D'] for v in baseline_d])-1e-9)),
        'posthoc_frontiers':actual_frontier,'target_not_met_indices':[int(r['index']) for r in rows if r['selected_metrics']['cost']>=1.03],
        'scope':'targeted adaptive development100, no unseen test, posthoc oracle not online selection, CI excludes solver run-to-run noise',
        'read_only_report':True,'model_or_solver_calls':0}
    save(HERE/'analysis.json',result)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    axes[0].scatter(base,q,c=d,cmap='viridis',s=23)
    lim=max(1.05,float(max(base.max(),q.max())));axes[0].plot([1,lim],[1,lim],'--',color='#88928d',lw=1)
    axes[0].set(xlabel='旧E42常规后处理 Q',ylabel='E48在线所选 Q',title='同例质量：点低于虚线表示改善')
    axes[1].scatter(d,q,c=n,cmap='plasma',s=23);axes[1].axhline(1.03,color='#b85b45',linestyle='--')
    axes[1].set(xlabel='原draft整体距离 D',ylabel='E48在线所选 Q',title='Q与距离同时保留，D无硬上限')
    axes[2].scatter(n,t,color='#287f75',s=20);axes[2].axhline(5,color='#b85b45',linestyle='--')
    axes[2].set(xlabel='模块数',ylabel='完整求解秒数',title='全部候选、评分检查与调度计时')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(HERE/'quality_distance_runtime.png',dpi=160);plt.close(fig)
    examples=[443,int(rows[int(np.argmax(q))]['index'])];examples=list(dict.fromkeys(examples))
    (HERE/'images').mkdir(exist_ok=True)
    for index in examples:
        r=next(r for r in rows if r['index']==index);c=np.asarray(dataset[index]['case'].solver_inputs[5]).tolist();selected=r['solver']['candidates'][r['solver']['selected_id']]
        variants={'raw':{'positions':r['raw_positions']},'old':{'positions':baseline[index]['positions'],'metrics':baseline[index]['metrics']},
                  'new':{'positions':selected['positions'],'metrics':selected['metrics']}}
        for v in variants.values():v['soft_details']=geometry_details(v['positions'],c)
        plot_case({'panel':'E48 quality-first evaluator','case_id':f'dev index{index}','constraints':c,'variants':variants,
                   'caption':'Same draft and inputs; baseline is an existing saved record; true Q is used only for offline scoring.'},
                  HERE/'images'/f'case_{index:04d}.png',stages=(('raw','Frozen E42 draft'),('old','Saved ordinary postprocessing'),('new','E48 balanced selection')))
    lines=['# E48：5秒draft潜力求解器的固定100例评价','',
        f"完成{len(rows)}/100，主结果为公开输入在线选择，Q<1.03为{result['selected_Q_lt1_03']}/100；候选中事后最低Q达标为{result['posthoc_any_Q_lt1_03']}/100。**没有达到所有Q<1.03，不能称为已经校准的权威draft评分器。** 原E42模型/Loss/普通后处理及封存测试保持不变。",'',
        '| 指标 | E48在线选择 | 旧同例常规结果 |','|---|---:|---:|',
        f"| 加权Q | {result['weighted_Q']:.6f} | {result['baseline_weighted_Q']:.6f} |",
        f"| 最差Q | {result['max_selected_Q']:.6f} | {float(base.max()):.6f} |",
        f"| P95 Q | {result['p95_selected_Q']:.6f} | {float(np.quantile(base,.95)):.6f} |",
        f"| Q<1.03 | {sum(q<1.03)}/100 | {sum(base<1.03)}/100 |",
        '',f"Q逐例胜/平/负{result['pair_win_tie_loss']}。worker聚类配对重采样95%加权Q差区间{result['paired_weighted_Q_delta_worker_bootstrap95']}，仅覆盖成员抽样不确定性，不包含数值随机性或自适应开发选择偏差。旧运行并发/线程口径不同，不比较旧时间作为提速证据。",'',
        f"完整均时{t.mean():.6f}秒、最大{t.max():.6f}秒，超过5秒{sum(t>5)}例。全部候选官方评分/独立检查计入实测时间；初始化/载入/JIT预热单列environment.json，磁盘写入不计完整求解。货架是否被选中、独立硬检查及候选数见summary.json。",'',
        '![质量、距离、完整时间](quality_distance_runtime.png)','','## 距离和候选潜力','',
        f"所选整体距离D中位数{np.median(d):.4f}、P95 {np.quantile(d,.95):.4f}。D及所有组成分量在paired_metrics.csv；它是固定工程偏好，没有用人类反馈校准，不能把某个D值直接解释成主观好坏。每例Q-D事后前沿保留候选ID，完整坐标在原cases中只存一份。",'',
        f"已存常规解用完全相同的原draft、公开面积和D重算：D中位数{result['baseline_D_median']:.4f}、P95 {result['baseline_D_p95']:.4f}；E48在{result['closer_D_cases']}/100例更近，配对平均D变化{result['paired_mean_D_delta']:+.4f}。这是已存布局诊断，未重跑求解或推理。",'',
        f"在线所选Q相对候选事后最好Q的平均差{gap.mean():.6f}、最大差{gap.max():.6f}。这区分公开选解代价偏差与搜索未找到好候选；事后最低Q利用参考分母，只能作为潜力诊断，不能称在线求解成绩。达到Q<1.03时的首次完整时间和最小已找到D在paired_metrics.csv，无达标候选就记空值，不能认定不存在。",'',
        '计算复杂不直接等于draft差：相同时间还包含模块数、约束、数值条件、当前调度和候选评分开销。许多好例早就找到达标候选，剩余时间仍用来搜索更近或更优结果。应比较同内核、同预算下首次好解时间/距离及达标率，不能把总耗时硬映射为唯一草稿分数。','',
        '## 机制与限制','',
        '保护分支、独立状态的多候选分支、永久原draft中心/形状软先验、可撤销关系变化、一次限时关系MIP、刚性分量压缩均共享截止。合法incumbent和远场货架兜底从开始就保留，每次候选接受前后复核固定尺寸、预放置、非负、面积和重叠。公开代价/软距离选择不读取真实Q，observe回调返回值不使用。','',
        '此前v3截断普通轨迹且没有独立分支状态，五例出现大退化，证据保留；v4修正调度后冻结100，不再按100反馈修改本轮搜索参数。首轮目标未达，不能只因候选变多或少数难例改善就升级工程基线。','',
        '100面板=5个已知难例＋其余495中的固定随机95，是明确面向开发诊断的面板。五例记录复用，不重复推理/求解；子记录保存reused_from和源SHA。原参考自由几何未进入搜索，没有读取未暴露测试。','',
        '## 复现与停止恢复','',
        '本报告对应历史软截止入口launch.ps1；完成后不重跑。推荐外部截止入口launch_resident.ps1见实验README，保护版仅通过五例、尚未重跑100，不能覆盖本轮超时。运行目录或实验runs下创建STOP可在案例边界停止。模型、全候选坐标、运行源码快照和图片只留本机；源码/身份/小型汇总可入Git。','',
        '```powershell',"& 'E:\\project\\ICCAD\\2026\\problemC\\experiments\\E48_quality_potential\\launch.ps1' -Stage screen -Resume",'```','',
        '只重建本报告（不求解、不推理）：','', '```powershell',
        "& 'E:\\project\\ICCAD\\2026\\problemC\\.conda\\envs\\iccad2026\\python.exe' 'E:\\project\\ICCAD\\2026\\problemC\\experiments\\E48_quality_potential\\runs\\20261024_screen100_frozen_v4\\analyze_results.py'",'```','']
    (HERE/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    print({k:v for k,v in result.items() if k not in ('posthoc_frontiers','target_not_met_indices')})


if __name__=='__main__':main()
