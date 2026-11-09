"""Stored-output analysis of radial expansion, no model/optimization calls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from experiments.E37_radial_draft_expansion.resources import configure
configure()
import argparse
import sqlite3
import zlib
import json
import numpy as np
from experiments.E37_radial_draft_expansion.io import save,sha
from experiments.E37_radial_draft_expansion.cases import public_cases
from experiments.E37_radial_draft_expansion.backend.frozen.packing import prepare_problem,_dimensions
from experiments.E37_radial_draft_expansion.diagnostics import geometry
from experiments.E37_radial_draft_expansion.reference_fast_geometry import topology

HERE=ROOT/'experiments/E37_radial_draft_expansion'
LABELS={'reference':'不扩展','radial5_literal':'纯径向5%直接输入','radial5_restore':'径向5%＋恢复预放置','radial5_rebase_restore':'径向5%＋公共平移＋恢复预放置'}


def paired(base,other):
    ids=sorted(base);a=[base[i] for i in ids];b=[other[i] for i in ids];n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12)
    dq=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(a,b)]);dt=np.array([y['seconds']-x['seconds'] for x,y in zip(a,b)])
    dm=np.array([y['final_geometry']['movement_from_original']['center_mean']-x['final_geometry']['movement_from_original']['center_mean'] for x,y in zip(a,b)])
    shards=sorted({r['member'][0] for r in a});lookup={s:k for k,s in enumerate(shards)};g=np.array([lookup[r['member'][0]] for r in a]);rng=np.random.default_rng(20261026);sample=rng.integers(0,len(shards),(4000,len(shards)))
    def interval(values,weights):
        sums=np.bincount(g,weights=values*weights,minlength=len(shards));den=np.bincount(g,weights=weights,minlength=len(shards))
        return np.percentile(sums[sample].sum(1)/den[sample].sum(1),[2.5,97.5]).tolist()
    changes=[{'index':i,'member':base[i]['member'],'n':base[i]['n'],'baseline_Q':base[i]['metrics']['cost'],'Q':other[i]['metrics']['cost'],
        'delta_Q':float(dq[k]),'delta_seconds':float(dt[k]),'delta_final_center_move':float(dm[k]),
        'original_overlap_ratio':base[i]['original_geometry']['overlap_area_pair_sum_over_total'],
        'expanded_overlap_ratio':other[i]['input_geometry']['overlap_area_pair_sum_over_total']} for k,i in enumerate(ids)]
    return {'weighted_Q_delta':float(np.average(dq,weights=w)),'weighted_Q_delta_95CI':interval(dq,w),
        'seconds_delta':float(dt.mean()),'seconds_delta_95CI':interval(dt,np.ones(len(ids))),
        'final_move_delta':float(dm.mean()),'final_move_delta_95CI':interval(dm,np.ones(len(ids))),
        'Q_improved_cases_1e_4':int((dq<-1e-4).sum()),'Q_worse_cases_1e_4':int((dq>1e-4).sum()),
        'max_regression':float(dq.max()),'max_improvement':float(dq.min()),'shards':len(shards),
        'worst_cases':sorted(changes,key=lambda r:r['delta_Q'],reverse=True)[:8],
        'best_cases':sorted(changes,key=lambda r:r['delta_Q'])[:8]}


def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('main','confirmation'),default='main');args=ap.parse_args()
    folder=HERE/'runs'/('20261022_paired100' if args.stage=='main' else '20261022_confirmation500');out=HERE/'runs'/('20261022_report' if args.stage=='main' else '20261022_confirmation_report');out.mkdir(exist_ok=True)
    config=json.loads((folder/'config.json').read_text());raw=json.loads((folder/'summary.json').read_text());assert raw['status']=='complete'
    con=sqlite3.connect('file:'+(folder/'outcomes.sqlite3').resolve().as_posix()+'?mode=ro',uri=True);records=[json.loads(zlib.decompress(row[0])) for row in con.execute('SELECT payload FROM records')];con.close()
    assert all('error' not in r and r['metrics']['is_feasible'] for r in records)
    byarm={arm:{r['index']:r for r in records if r['arm']==arm} for arm in config['arms']};effects={arm:paired(byarm['reference'],rr) for arm,rr in byarm.items() if arm!='reference'}
    cases,metadata,_=public_cases('confirmation' if args.stage=='confirmation' else 'monitor');geometry_tables={};seeds={};resources={r['resource']['pid']:r['resource'] for r in records}
    for arm,rr in byarm.items():
        table={}
        for stage,key in [('original','original_geometry'),('raw_radial','raw_expansion_geometry'),('solver_input','input_geometry'),('final','final_geometry')]:
            gg=[r[key] for r in rr.values()]
            table[stage]={k:mean(gg,k) for k in gg[0] if k!='movement_from_original'}
            table[stage]['bbox_ratio_to_original_mean']=float(np.mean([r[key]['bbox_area']/r['original_geometry']['bbox_area'] for r in rr.values()]))
            table[stage]['wire_ratio_to_original_mean']=float(np.mean([r[key]['wire']/max(r['original_geometry']['wire'],1e-12) for r in rr.values()]))
        geometry_tables[arm]=table
        reconstructed=[]
        for i,r in rr.items():
            assert list(metadata[i]['member'])==r['member'];p=prepare_problem(*cases[i].solver_inputs);actual=np.asarray(r['input_positions'])
            widths,heights=_dimensions(p,actual[:,2]/actual[:,3],True);seed=np.column_stack((actual[:,:2],widths,heights))
            base_actual=np.asarray(byarm['reference'][i]['input_positions']);base_seed=np.column_stack((base_actual[:,:2],widths,heights))
            g=geometry(seed,p,base_actual)
            g['separation_flip_vs_baseline_seed']=float(np.any(topology(seed,p.constraints,array=True)!=topology(base_seed,p.constraints,array=True),axis=1).mean())
            reconstructed.append(g)
        seeds[arm]={'note':'offline reconstruction from frozen STEADY initialization; not extra solver calls',
            'overlap_pairs_mean':mean(reconstructed,'overlap_pairs'),'overlap_ratio_mean':mean(reconstructed,'overlap_area_pair_sum_over_total'),
            'separation_flip_vs_baseline_seed_mean':mean(reconstructed,'separation_flip_vs_baseline_seed')}
    assert len(resources)<=4 and len({tuple(v['affinity']) for v in resources.values()})==len(resources)
    assert all(v['max_cpu_execution_slots']==1 and v['torch_intraop']==1 for v in resources.values())
    assert all(sha(ROOT/name)==value for name,value in config['sources'].items())
    result={'scope':config['panel_scope'],'summaries':raw['summaries'],'paired_vs_reference':effects,'geometry':geometry_tables,'reconstructed_initial_seed':seeds,
        'resources':list(resources.values()),'all_sources_unchanged':True,'source_database_sha256':sha(folder/'outcomes.sqlite3'),
        'analysis_source_sha256':sha(Path(__file__)),'cumulative_elapsed_seconds':raw['cumulative_elapsed_seconds'],'no_new_solves':True,'model_training':False,'test_read':False}
    save(out/'summary.json',result)
    if args.stage=='main':
        base=raw['summaries']['reference'];eligible=[]
        for arm in ('radial5_restore','radial5_rebase_restore'):
            s=raw['summaries'][arm];effect=effects[arm]
            if s['weighted_Q']<=base['weighted_Q'] and s['seconds_mean']<=base['seconds_mean'] and s['cvar95_Q']<=base['cvar95_Q'] and s['max_Q']<=base['max_Q']+1e-8 and effect['max_regression']<=.25:eligible.append(arm)
        selected=min(eligible,key=lambda arm:(raw['summaries'][arm]['weighted_Q'],raw['summaries'][arm]['seconds_mean'])) if eligible else None
        save(HERE/'runs/selection.json',{'eligible':bool(eligible),'arm':selected,'eligible_arms':eligible,
            'rule':'aligned arms only: weighted Q, mean time, CVaR95, max Q jointly nonworse; no paired Q regression >0.25; select lowest weighted Q then time',
            'reason':'Proceed with one frozen candidate' if selected else 'No aligned variant met joint objectives; do not expand to500',
            'source_summary_sha256':sha(out/'summary.json')})
    lines=['# E37：位置径向拉开5%的结果','',
        f"固定{len(byarm['reference'])}例，模型/draft/尺寸冻结，{len(records)}个最终输出全部硬合法。4进程×每进程1线程；实测耗时包含变换与后处理，几何诊断/计分延后。Q为自适应开发中性时间指标。",'',
        '| 策略 | 加权Q | 平均秒 | P95秒 | CVaR95 Q | 最大Q | 救援例数 | 首轮PrimalInfeasible |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in raw['summaries'].items():lines.append(f"| {LABELS[arm]} | {s['weighted_Q']:.6f} | {s['seconds_mean']:.6f} | {s['seconds_p95']:.6f} | {s['cvar95_Q']:.6f} | {s['max_Q']:.6f} | {s['rescue_cases']} | {s['first_PrimalInfeasible']} |")
    lines+=['','## 展开本身与入口对齐的作用','', '| 策略的有效输入 | 重叠对/例 | 重叠面积比（成对和） | 负坐标块/例 | 预放置偏移块/例 | 原近接触保留率 | 分组违反代理 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,table in geometry_tables.items():
        g=table['solver_input'];ret=g['original_near_contact_retention'];ret=f'{ret:.4f}' if ret is not None else 'NA'
        lines.append(f"| {LABELS[arm]} | {g['overlap_pairs']:.2f} | {g['overlap_area_pair_sum_over_total']:.5f} | {g['negative_coordinate_blocks']:.2f} | {g['preplaced_offset_blocks']:.2f} | {ret} | {g['grouping_proxy_violations']:.2f} |")
    lines+=['','接触定义为原mode0分离方向上的边缘间隙绝对值≤0.02个较小块边长，且切向重叠>0.02块单位；用于观察近组装关系，不等于官方精确接触。非法draft的分组连通还可能来自重叠，因此这些是几何代理，不是可行解质量结论。原始径向结果和后处理实际输入分开记录；恢复预放置后不再保证所有块对重叠单调下降。','',
        '## 配对差','', '| 策略 | Δ加权Q（95%分片区间） | Δ平均秒（95%分片区间） | Q改善/恶化例数（1e-4） | 最大逐例回归 |','|---|---|---|---|---:|']
    for arm,e in effects.items():lines.append(f"| {LABELS[arm]} | {e['weighted_Q_delta']:+.6f} {e['weighted_Q_delta_95CI']} | {e['seconds_delta']:+.6f} {e['seconds_delta_95CI']} | {e['Q_improved_cases_1e_4']}/{e['Q_worse_cases_1e_4']} | {e['max_regression']:.6f} |")
    lines+=['','按完整来源分片进行4000次配对bootstrap，描述当前开发面板不确定性，不代表封存测试泛化或多重比较认证。首轮预放置负坐标可使原子问题等式与非负约束直接冲突；不能把原始输入分支的失败一概归因于展开几何无效。','',
        '## 是否更尊重原draft','', '| 策略 | 最终相对原draft平均位移 | 平均绝对log宽高比改变 | 全树shape调用/例 | 原生shape累计平均位移/例 |','|---|---:|---:|---:|---:|']
    for arm,s in raw['summaries'].items():
        rr=list(byarm[arm].values());lines.append(f"| {LABELS[arm]} | {s['final_mean_center_move']:.4f} | {s['final_mean_log_AR_change']:.4f} | {s['all_shape_calls_mean']:.2f} | {np.mean([r['all_shape_cumulative_center'] for r in rr]):.4f} |")
    lines+=['','最终位移全部以同一原始模型draft为基准，避免换成展开后input而人为缩小调整幅度。累计位移是每次shape输入→原始输出的平均中心距离之和，包含递归，不拼接救援重启的跳跃。尺寸在展开阶段逐位不变；后处理仍可按原算法改变形状。','',
        '原始径向变换对中心向量乘1.05，理论上保留原mode0分离选择；但原求解器初始化会按公开面积/MIB规则重建宽高，且入口对齐会恢复预放置坐标，因此实际初始关系可能改变。summary保存按固定初始化规则离线重建的冲突和关系变化，无新增求解。','',
        '[精确统计/反例/初始化重建](summary.json) · [配对与几何图](effects.png)','']
    if args.stage=='main':
        lines+=['','## 结论与停止决定','',
            '没有对齐策略同时满足加权Q、均时、CVaR95、最大Q和单例回归上限，因此不扩大500例，不替换E25，也不继续扫描扩展比例。纯展开降低成对重叠面积和26.4%，却不能据此推断求解更容易。恢复预放置后，均时差区间跨零，不能声称加速；加权Q点估计与本面板分片区间均偏向恶化。公共平移臂的最差5%均值下降约5.1%，但加权Q和严重逐例回归未通过要求。','',
            '纯展开后外包框平均增大9.07%，公开线长代理增大3.33%；mode0关系翻转平均仅0.00033%，原近接触保留率却仅0.5%。这说明相对方向和紧贴拼装是不同特征。近接触阈值为0.02个块尺度，比例对阈值敏感；不能把该保留率解读为有用信息损失比例，也不能仅凭相关观察断定Q恶化的单一原因。后处理可重建接触，但最终位移与形状调整并未变小。','',
            '恢复预放置臂的加权HPWL gap为0.003692→0.002954、加权面积gap为-0.000898→-0.001788；其加权分组违反计数由0.26894升至0.44206，边界由0.25090升至0.27799。几何gap变好仍可被软约束代价抵消，不能只观察重叠和线长。各项均按同一模块数指数权重汇总，不能将这些单项均值直接相加重构非线性Q。','',
            '按模块数作描述性切片，公共平移臂在<60块的40例平均ΔQ=-0.01668，在90–109块的20例却为+0.02667；加权Q恶化与较大实例更差相符。此为事后解释，不据此训练门控或选择性报告。','',
            '后续假设应是仅在冲突区域定向移动、保留组内近邻与公开锚点，并允许干预回滚。本轮没有据此新增策略或求解。模型保持冻结，测试未读取。']
        example_path=out/'examples.json'
        if example_path.exists():
            examples=json.loads(example_path.read_text(encoding='utf-8'))
            lines+=['','## 同时保留成功例与反例','',
                '| 开发索引 | 策略 | Q原→新 | 秒原→新 | 图 |','|---|---|---|---|---|']
            for e in examples['figures']:
                lines.append(f"| {e['index']} | {e['arm']} | {e['reference_Q']:.6f}→{e['variant_Q']:.6f} | {e['reference_seconds']:.3f}→{e['variant_seconds']:.3f} | [布局]({e['file']}) |")
            lines+=['','例496原有分组断连改善且免去长救援，例261却从分组全连通变为G3三个分量，并增加边界违反。两者共同说明统一扩展可能改变求解路径；现有数据不支持把它无条件应用到所有draft。','',
                '[最终官方分项、模块数切片及源码/尺寸/权重/E25原结果复核](examples.json)。图由共享visualization.layouts.plot_case生成，全部来自已保存输出，没有新增求解。']
    (out/'analysis.md').write_text('\n'.join(lines),encoding='utf-8');plot(out,result)
    print(json.dumps({'summaries':raw['summaries'],'paired':effects,'geometry':geometry_tables,'initial_seed':seeds},ensure_ascii=False),flush=True)


def plot(out,result):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=[a for a in result['summaries'] if a!='reference'];y=np.arange(len(arms));fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,ci,title in [(axes[0],'weighted_Q_delta','weighted_Q_delta_95CI','Weighted Q change vs original input'),(axes[1],'seconds_delta','seconds_delta_95CI','Preprocessing + postprocess seconds change')]:
        ee=[result['paired_vs_reference'][a] for a in arms];v=np.array([e[key] for e in ee]);bounds=np.array([e[ci] for e in ee])
        ax.errorbar(v,y,xerr=np.maximum(0,np.stack((v-bounds[:,0],bounds[:,1]-v))),fmt='o',capsize=4);ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,arms);ax.set_title(title);ax.grid(alpha=.2)
    for a in result['summaries']:
        g=result['geometry'][a]['solver_input'];axes[2].scatter(g['overlap_area_pair_sum_over_total'],g['original_near_contact_retention'],label=a)
    axes[2].set(xlabel='Pairwise overlap area / public area',ylabel='Original near-contact retention',title='Overlap reduction and assembly separation');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'effects.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
