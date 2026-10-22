"""Post-hoc paired statistics from frozen E36 outputs; never call a solver."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E36_persistent_conflict_repair.resources import configure
configure()
import json
import zlib
import sqlite3
from collections import Counter
import numpy as np
from experiments.E36_persistent_conflict_repair.io import save,sha

HERE=ROOT/'experiments/E36_persistent_conflict_repair';OUT=Path(__file__).resolve().parent
LABELS={'reference':'原逻辑','control':'采集对照','hard_after_legal':'首次合法后硬化','local_repair':'局部固定形状平移','early_rescue':'三轮冲突提前退出','local_then_escape':'先局部平移再退出'}


def comparisons(a,b):
    ids=sorted(a);base=[a[i] for i in ids];other=[b[i] for i in ids]
    n=np.array([r['n'] for r in base]);w=np.exp((n-n.max())/12)
    dq=np.array([r['metrics']['cost']-s['metrics']['cost'] for r,s in zip(other,base)])
    dt=np.array([r['seconds']-s['seconds'] for r,s in zip(other,base)])
    dm=np.array([r['draft_to_final']['mean_center_move']-s['draft_to_final']['mean_center_move'] for r,s in zip(other,base)])
    shard=sorted({r['member'][0] for r in base});g=np.array([shard.index(r['member'][0]) for r in base]);rng=np.random.default_rng(20261022)
    sample=rng.integers(0,len(shard),(4000,len(shard)))
    def interval(d,weights):
        sums=np.bincount(g,weights=d*weights,minlength=len(shard));den=np.bincount(g,weights=weights,minlength=len(shard))
        return np.percentile(sums[sample].sum(1)/den[sample].sum(1),[2.5,97.5]).tolist()
    changes=[{'index':i,'member':a[i]['member'],'n':a[i]['n'],'baseline_Q':a[i]['metrics']['cost'],'Q':b[i]['metrics']['cost'],
        'delta_Q':float(dq[k]),'delta_seconds':float(dt[k]),'delta_mean_center_move':float(dm[k])} for k,i in enumerate(ids)]
    return {'weighted_Q_delta':float(np.average(dq,weights=w)),'weighted_Q_delta_95CI':interval(dq,w),
        'seconds_delta':float(dt.mean()),'seconds_delta_95CI':interval(dt,np.ones(len(ids))),
        'mean_center_move_delta':float(dm.mean()),'mean_center_move_delta_95CI':interval(dm,np.ones(len(ids))),
        'Q_improved_cases_1e_4':int((dq<-1e-4).sum()),'Q_worse_cases_1e_4':int((dq>1e-4).sum()),
        'Q_worst_regression':float(dq.max()),'Q_best_improvement':float(dq.min()),
        'worst_cases':sorted(changes,key=lambda r:r['delta_Q'],reverse=True)[:5],
        'best_cases':sorted(changes,key=lambda r:r['delta_Q'])[:5],'shards':len(shard)}


def main():
    folder=HERE/'runs/20261022_paired100';raw=json.loads((folder/'summary.json').read_text());assert raw['status']=='complete'
    config=json.loads((folder/'config.json').read_text());db=folder/'outcomes.sqlite3'
    con=sqlite3.connect('file:'+db.resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(row[0])) for row in con.execute('SELECT payload FROM records')];con.close()
    assert len(records)==600 and all(r['metrics']['is_feasible'] and 'error' not in r for r in records)
    byarm={a:{r['index']:r for r in records if r['arm']==a} for a in LABELS}
    effects={base:{arm:comparisons(byarm[base],rr) for arm,rr in byarm.items() if arm!=base} for base in ('reference','control')}
    functions={};triggered={};failures={};details={};frozen={name:sha(ROOT/name)==value for name,value in config['sources'].items()}
    assert all(frozen.values())
    for arm,rr in byarm.items():
        shapes=[t for r in rr.values() for t in r['stats'].get('telemetry',[])];root=[t for t in shapes if t['level']==0]
        repair=[(i,t) for i,r in rr.items() for t in r['stats']['trace'] if t['phase']=='local_repair']
        functions[arm]={'root_shapes':sum(r['root_shape_calls'] for r in rr.values()),'all_shapes':len(shapes) if shapes else None,
            'strict_shapes':sum(t['strict_nonoverlap'] for t in root),'legal_root_shapes':sum(t['hard_legal'] for t in root) if root else None,
            'root_cumulative_mean_center':float(np.mean([r['root_cumulative_center_move'] for r in rr.values()])) if root else None,
            'root_solver_statuses':dict(Counter(t['status'] for t in root)),
            'repair_calls':len(repair),'repair_effective_calls':sum(bool(t['moves']) for i,t in repair),
            'repair_direct_seconds':sum(t['seconds'] for i,t in repair),
            'repair_moved_block_occurrences':sum(len(t['moves']) for i,t in repair),
            'repair_max_single_block_shift':max([m['block_units'] for i,t in repair for m in t['moves']]+[0.])}
        triggered[arm]={'repair_indices':sorted({i for i,t in repair}),'escape_indices':sorted(i for i,r in rr.items() if r['conflict_escape']),
            'rescue_indices':sorted(i for i,r in rr.items() if r['robust_refine_calls'])}
        failures[arm]=[{'index':i,'iteration':t['iteration'],'status':t['status']} for i,r in rr.items() for t in r['stats'].get('telemetry',[]) if t['level']==0 and t['status'] not in ('Solved','AlmostSolved','MaxTime')]
        details[arm]={str(i):{'metrics':r['metrics'],'seconds':r['seconds'],'root_shape_calls':r['root_shape_calls'],
            'root_cumulative_center_move':r['root_cumulative_center_move'],'draft_to_final':r['draft_to_final'],
            'repair_trace':[t for t in r['stats']['trace'] if t['phase'] in ('local_repair','conflict_escape')]} for i,r in rr.items() if i in (329,496,368,127,332,310)}
    result={'scope':'same cached frozen-model drafts on adaptive dev100; measured postprocessor runtime; no training/test',
        'summaries':raw['summaries'],'paired':effects,'functions':functions,'triggered':triggered,'strict_failure_events':failures,
        'stress_cases':details,'sources_unchanged':all(frozen.values()),'source_database_sha256':sha(db),
        'upstream_database_sha256':config['source_database_sha256'],'analysis_source_sha256':sha(Path(__file__)),
        'cumulative_numerical_seconds':raw['cumulative_elapsed_seconds'],'new_training':False,'test_read':False}
    save(OUT/'summary.json',result)
    lines=['# E36：持续冲突处理的首轮配对研究','','固定E35的同一100例，模型与draft冻结；CPU总计两线程。五个反例检查后冻结算法，正式六臂600个输出全部硬合法。主计时为后处理实测，不含复用draft的历史推理时间；不可直接与E25原500端到端耗时比较。Q为自适应MIB开发中性时间指标，不是正式比赛成绩。','',
        '| 策略 | 加权Q | 平均后处理秒 | P95秒 | CVaR95 Q | 最大Q | 最终平均位移 | 主流程shape/例 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in raw['summaries'].items():lines.append(f"| {LABELS[arm]} | {s['weighted_Q']:.6f} | {s['seconds_mean']:.6f} | {s['seconds_p95']:.6f} | {s['cvar95_Q']:.6f} | {s['max_Q']:.6f} | {s['mean_final_center_move']:.4f} | {s['root_shape_calls_mean']:.2f} |")
    lines+=['','## 对原逻辑的配对差','', '| 策略 | Δ加权Q与95%区间 | Δ平均秒与95%区间 | Q改善/恶化例数（1e-4） | 最大逐例Q恶化 |','|---|---|---|---|---:|']
    for arm,e in effects['reference'].items():lines.append(f"| {LABELS[arm]} | {e['weighted_Q_delta']:+.6f} {e['weighted_Q_delta_95CI']} | {e['seconds_delta']:+.6f} {e['seconds_delta_95CI']} | {e['Q_improved_cases_1e_4']}/{e['Q_worse_cases_1e_4']} | {e['Q_worst_regression']:.6f} |")
    lines+=['','区间按来源完整分片做4000次配对bootstrap，是本次开发面板的不确定性描述，不是多重比较后的显著性认证。全部策略共享输入；reference保持原求解逻辑，control加入公共过程采集，用于识别采集和墙钟截止扰动。各策略对control的配对差也完整保存在summary。','',
        '## 机制是否实际触发','', '| 策略 | 主流程shape总数 | 硬无重叠shape | 修复调用/有移动调用 | 块移动次数 | 最大小块平移 | 提前退出实例 | 救援实例 |','|---|---:|---:|---|---:|---:|---:|---:|']
    for arm,f in functions.items():lines.append(f"| {LABELS[arm]} | {f['root_shapes']} | {f['strict_shapes']} | {f['repair_calls']}/{f['repair_effective_calls']} | {f['repair_moved_block_occurrences']} | {f['repair_max_single_block_shift']:.4f} | {len(triggered[arm]['escape_indices'])} | {len(triggered[arm]['rescue_indices'])} |")
    lines+=['','## 解释边界','',
        '固定形状局部修复只沿单轴移动少数非预放置块；保持宽高、不引入新的物理重叠，单次单块≤0.5块单位、最多四块、每例最多两次调用。它不保证所有软接触/边界关系不变；候选仍由原硬守卫和内部目标筛选。几何移动少不等于质量更好。',
        '本轮机制只改变主流程；递归打包后的细化保持原控制逻辑。提前退出仍会尝试剩余预算内的末端投影，然后按原条件进入救援；没有把直接返回行保底当作提速。救援率上升可能抵消节省，或改变尾部，需逐例核对。',
        '合法后硬化测试将后续shape的分离约束从有限惩罚改为零弹性；若子问题不可行/无合格返回，则保留合法最佳解并依原流程处理。这与最终输出始终硬合法是不同层次。非负坐标、固定/预放置规格和面积容差从原子问题开始就受约束；边界/分组是正式质量软约束，不能把其惩罚误称为硬合法。',
        '本轮的reference是本实验独立保留的E25原逻辑；仅直接导入路径改变。新策略含冲突识别与记录的真实成本。累计位移只在control及新策略中采集，reference只有最终净位移；不拿历史观测轨迹当本轮精确的累计路径对照。',
        '本100例已参与诊断开发，且五例机制检查是按历史反例选取；这些结果不能宣称独立泛化。对Q/耗时/尾部的联合结论必须同时考虑逐例反例与观测扰动，不能只选有利均值。','',
        '[精确汇总/触发名单/压力样例](summary.json) · [配对差与尾部图](paired_effects.png) · [原运行配置](../20261022_paired100/config.json) · [五例检查](../20261022_gate5/summary.json)','']
    (OUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(result,byarm)
    print(json.dumps({'summaries':raw['summaries'],'functions':functions,'paired':effects['reference'],'triggered':triggered},ensure_ascii=False),flush=True)


def plot(result,byarm):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=[a for a in LABELS if a not in ('reference','control')]
    fig,axes=plt.subplots(1,3,figsize=(15,5.5));y=np.arange(len(arms))
    for ax,key,ci,title in [(axes[0],'weighted_Q_delta','weighted_Q_delta_95CI','Weighted Q change vs original'),(axes[1],'seconds_delta','seconds_delta_95CI','Mean postprocessor seconds change')]:
        data=[result['paired']['reference'][a] for a in arms];v=np.array([e[key] for e in data]);bounds=np.array([e[ci] for e in data])
        ax.errorbar(v,y,xerr=np.maximum(0,np.stack((v-bounds[:,0],bounds[:,1]-v))),fmt='o',capsize=4);ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,arms);ax.set_title(title);ax.grid(alpha=.2)
    for arm in ['reference']+arms:
        s=result['summaries'][arm];axes[2].scatter(s['cvar95_Q'],s['max_Q'],label=arm)
    axes[2].set(xlabel='CVaR95 Q',ylabel='Maximum Q',title='Quality tail (lower is better)');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'paired_effects.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
