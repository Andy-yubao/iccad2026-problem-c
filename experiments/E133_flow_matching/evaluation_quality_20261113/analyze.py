"""Summarize completed primary evaluation and pair only existing baseline results."""
import json
from pathlib import Path
import numpy as np
from experiments.E133_flow_matching.control import OWNER,ROOT,read,sha,sources
from experiments.E133_flow_matching.io import save

HERE=Path(__file__).resolve().parent
BASE=ROOT/'experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500'

def stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])

def main():
    out=HERE/'dev500';fm=read(out/'results.json');baseline=read(BASE/'results.json')
    assert fm['status']=='complete' and fm['count']==500
    assert read(out/'state.json')['status']=='complete'
    fcases=[read(out/f'cases/{i:04d}.json') for i in range(500)]
    bcases=[read(BASE/f'cases/{i:04d}.json') for i in range(500)]
    for f,b in zip(fcases,bcases):
        assert f['index']==b['index'] and f['member']==b['member'] and f['n']==b['n']
        assert f['metrics']['hpwl_baseline']==b['metrics']['hpwl_baseline']
        assert f['metrics']['bbox_area_baseline']==b['metrics']['bbox_area_baseline']
    frozen=read(OWNER/'evaluation/frozen_model.json')
    assert sources()==frozen['sources'] and sha(ROOT/frozen['checkpoint'])==frozen['checkpoint_sha256']
    assert sha(OWNER/'evaluation/budget.json')==sha(ROOT/'experiments/E119_global_initial_generator/variants/public_priority_v1/evaluation/budget.json')
    history=read(OWNER/'runs/fm_seed0/progress.json')['history']
    best=next(h for h in history if h['epoch']==fm['best_epoch'])
    training=read(OWNER/'runs/fm_seed0/results.json')
    qdelta=np.array([f['metrics']['cost']-b['metrics']['cost'] for f,b in zip(fcases,bcases)])
    paired=dict(improved=int((qdelta< -1e-10).sum()),worse=int((qdelta>1e-10).sum()),tied=int((np.abs(qdelta)<=1e-10).sum()),
        mean_delta=float(qdelta.mean()),median_delta=float(np.median(qdelta)),p95_delta=float(np.quantile(qdelta,.95)))
    old_geometry=[json.loads(line) for line in (ROOT/'output/draft_quality_20261112/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    old_provenance=read(ROOT/'output/draft_quality_20261112/provenance.json')
    assert old_provenance['sources']['scoring\\draft_geometry.py']==sha(ROOT/'scoring/draft_geometry.py')
    old_geometry={r['index']:r for r in old_geometry if r['model']=='E119' and r['stage']=='draft'}
    assert set(old_geometry)==set(range(500))
    geometry_names=['duplicate_coverage','covered_module_p95','covered_module_max','whitespace_bbox',
        'group_contact_bottleneck','boundary_gap_local','fixed_dimension_relative_mean','preplaced_position_local_mean',
        'bbox_reference','hpwl_reference','move_global_mean','shape_log_change_mean']
    geometry={}
    for name in geometry_names:
        old=[old_geometry[i].get(name,np.nan) for i in range(500)]
        new=[r['draft_quality'].get(name,np.nan) for r in fcases]
        if all(np.isfinite(old)) and all(np.isfinite(new)):
            d=np.array(new)-np.array(old)
            geometry[name]=dict(E119=stats(old),E133=stats(new),improved=int((d< -1e-10).sum()),worse=int((d>1e-10).sum()))
    soft={}
    for key in ('boundary_violations','grouping_violations','total_soft_violations','violations_relative','area_gap','hpwl_gap'):
        soft[key]=dict(E119=stats([r['metrics'][key] for r in bcases]),E133=stats([r['metrics'][key] for r in fcases]),
            E119_affected=sum(r['metrics'][key]>0 for r in bcases),E133_affected=sum(r['metrics'][key]>0 for r in fcases))
    buckets={}
    for lo,hi in ((0,60),(60,90),(90,110),(110,121)):
        indices=[i for i,r in enumerate(fcases) if lo<=r['n']<hi]
        buckets[f'{lo}-{hi-1}']=dict(count=len(indices),E119_mean_Q=float(np.mean([bcases[i]['metrics']['cost'] for i in indices])),
            E133_mean_Q=float(np.mean([fcases[i]['metrics']['cost'] for i in indices])),
            E133_mean_seconds=float(np.mean([fcases[i]['complete_seconds'] for i in indices])))
    step=read(out/'five_train_step_sensitivity.json')
    sensitivity={k:dict(mean_endpoint_MSE=float(np.mean(v['standardized_endpoint_MSE'])),
        per_case_endpoint_MSE=v['standardized_endpoint_MSE'],batch_seconds=v['batch_seconds']) for k,v in step['steps'].items()}
    timing={name:stats([r[name] for r in fcases]) for name in ('draft_seconds','complete_seconds','repair_seconds')}
    worst=sorted(fcases,key=lambda r:(-r['metrics']['cost'],r['index']))[:10]
    summary=dict(status='complete',training=training,best_monitor=best['dev_monitor'],best_train_monitor=best['train_monitor'],
        initial_dev_FM_loss=history[0]['dev_monitor']['total'],FM_loss_reduction=1-fm['own_selection_loss']/history[0]['dev_monitor']['total'],
        shape_velocity_loss_fraction=best['dev_monitor']['shape']/3/best['dev_monitor']['total'],
        E119=baseline,E133=fm,paired_Q=paired,draft_geometry=geometry,final_soft=soft,buckets=buckets,timing=timing,
        same_five_train_step_sensitivity=sensitivity,worst_indices=[r['index'] for r in worst],
        original_source_and_checkpoint_unchanged=True,frozen_before_Q=True,test_read=False)
    save(HERE/'summary.json',summary)
    save(HERE/'verification.json',dict(checkpoint_sha256=frozen['checkpoint_sha256'],trained_sources_match=True,
        same500_members_order_module_counts_and_scoring_denominators=True,postprocessor_budget_identity=True,
        adapter_sha256=sha(HERE/'evaluate.py'),analysis_sha256=sha(Path(__file__)),
        baseline_result_sha256=sha(BASE/'results.json'),geometry_source_sha256=sha(ROOT/'scoring/draft_geometry.py'),
        historical_geometry_sha256=sha(ROOT/'output/draft_quality_20261112/cases.jsonl'),test_read=False))
    better=fm['mixed_Q']<baseline['mixed_Q']
    lines=['# E133 Flow Matching质量评价（2026-11-13）','',
        '**结论：'+('本次固定dev500的最终混合Q低于E119历史基线，质量更好。' if better else '本次固定dev500质量明显弱于E119历史基线，最终混合Q更高。')+'** 草稿、后处理后结果与耗时分别报告；本次单seed、单生成样本，不构成架构优劣或FM方法普遍有效/无效的因果结论。','',
        '## 范围与完成情况','',
        '训练247轮/123500步后正常早停，唯一FM Loss-best为第239轮，权重SHA256 `'+frozen['checkpoint_sha256']+'`。本次无训练、无参数修改、无test、无新增图片、无多候选选优。原自动评价在构造冻结记录时重复传入test_read，0例完成；原现场保留。本次用户明确要求评价后，以独立输出适配器修正记录并完成评价，原训练/评价源码与原8h预算记录均保持。此次只读评价单独登记1800秒上限，不冒充原8h内完成。','',
        '固定原dev500逐例核对成员、顺序、模块数、评分分母；Euler8、每例唯一固定噪声，原E98.depth_budget.v1后处理、原始面积、无MIB及独立硬检查。4/16步仅原五训练成员诊断，不据此更换主配置。','',
        '## 最终质量与完整耗时','',
        table(['指标（Q越低越好）','E119历史best55','E133 FM best239'],[
            ['加权Q',f"{baseline['weighted_Q']:.6f}",f"{fm['weighted_Q']:.6f}"],
            ['平均Q',f"{baseline['mean_Q']:.6f}",f"{fm['mean_Q']:.6f}"],
            ['混合Q',f"{baseline['mixed_Q']:.6f}",f"{fm['mixed_Q']:.6f}"],
            ['后处理后硬合法',str(baseline['hard_legal'])+'/500',str(fm['hard_legal'])+'/500'],
            ['草稿硬合法',str(baseline['raw_hard_legal'])+'/500',str(fm['raw_hard_legal'])+'/500'],
            ['合法行回退',baseline['fallbacks'],fm['fallbacks']],
            ['完整均时ms',f"{baseline['mean_complete_seconds']*1000:.1f}",f"{fm['mean_complete_seconds']*1000:.1f}"],
            ['完整P95 ms',f"{baseline['p95_complete_seconds']*1000:.1f}",f"{fm['p95_complete_seconds']*1000:.1f}"],
            ['完整最大ms',f"{baseline['max_complete_seconds']*1000:.1f}",f"{fm['max_complete_seconds']*1000:.1f}"],
            ['超过250ms例数',baseline['over250ms'],fm['over250ms']],
            ['Q P95',f"{baseline['p95_Q']:.6f}",f"{fm['p95_Q']:.6f}"],
            ['最大Q',f"{baseline['max_Q']:.6f}",f"{fm['max_Q']:.6f}"]]),'',
        f"同例配对：FM较好{paired['improved']}例、较差{paired['worse']}例、持平{paired['tied']}例；平均Q差{paired['mean_delta']:+.6f}。逐例完整时间包含输入复制、特征、噪声生成、传输、8次速度调用、解码、全部后处理/回退和独立硬检查；加载/预热/评分/几何诊断另列。历史耗时受运行环境影响，不能作严格等环境测速。",'',
        '## 真实草稿几何','',
        '以下等权平均到实例；模块位置/接触/边界误差按模块尺度归一。草稿未用公开规格覆盖F/P。非法草稿的较小外框或线长不自动代表更优。','']
    names=[('duplicate_coverage','重复覆盖率'),('covered_module_p95','每例模块覆盖P95'),('whitespace_bbox','真实留白/外框'),
        ('group_contact_bottleneck','组接触瓶颈/模块尺度'),('boundary_gap_local','边界缺口/模块尺度'),
        ('fixed_dimension_relative_mean','F/P尺寸相对误差'),('preplaced_position_local_mean','P左下角位置误差/模块尺度')]
    lines += [table(['指标','E119草稿','FM草稿','FM改善/恶化例数'],[
        [label,f"{100*geometry[k]['E119']['mean']:.3f}%",f"{100*geometry[k]['E133']['mean']:.3f}%",
        f"{geometry[k]['improved']}/{geometry[k]['worse']}"] for k,label in names if k in geometry]),'',
        '## FM训练与积分诊断','',
        f"固定dev速度MSE {history[0]['dev_monitor']['total']:.6f}→{fm['own_selection_loss']:.6f}，下降{100*summary['FM_loss_reduction']:.1f}%；best训练监测MSE {best['train_monitor']['total']:.6f}，验证MSE {best['dev_monitor']['total']:.6f}。形状速度通道占剩余总速度误差{100*summary['shape_velocity_loss_fraction']:.1f}%。这是插值路径速度预测，不等于自由积分后的终点误差；正式Euler8的dev500平均标准化终点MSE为{fm['mean_endpoint_MSE']:.6f}。",'',
        table(['五个原训练成员，同一噪声','4步','8步（主配置）','16步'],[
            ['平均标准化终点MSE',*[f"{sensitivity[str(k)]['mean_endpoint_MSE']:.6f}" for k in (4,8,16)]]]),'',
        '五例步数变化只能提示离散误差或状态敏感性，不能支持整套泛化或改步数后Q一定更好。没有按Q选择步数、checkpoint或采样种子。训练日志仍未观察到train平台，不把正常dev早停称为充分拟合。','',
        '## 退化集中在哪里','',
        f"固定规格条件被有效利用：F/P尺寸相对误差均值为{100*geometry['fixed_dimension_relative_mean']['E133']['mean']:.3f}%（E119为{100*geometry['fixed_dimension_relative_mean']['E119']['mean']:.3f}%），但这没有转化为好的全局布局。FM草稿总线长/reference为{geometry['hpwl_reference']['E133']['mean']:.4f}（E119为{geometry['hpwl_reference']['E119']['mean']:.4f}）；后处理后线长gap均值为{100*soft['hpwl_gap']['E133']['mean']:.2f}%（E119为{100*soft['hpwl_gap']['E119']['mean']:.2f}%）。最终有分组软违反的例数{soft['grouping_violations']['E133_affected']}，对照为{soft['grouping_violations']['E119_affected']}；有贴边软违反的例数{soft['boundary_violations']['E133_affected']}，对照为{soft['boundary_violations']['E119_affected']}。这些是组织/连接质量退化的直接证据，不是数值溢出：全部500例有限，ratio截断为{fm['ratio_clipped']}个模块，无合法行回退。",'',
        f"五例中4→8→16步的终点MSE逐步上升，尚无增加步数即可改善的正面证据；这并不说明4步整套Q更好，也不足以区分目标分布、速度误差积累或单参考多模态歧义的原因。建议下一轮首先诊断位置/拓扑组织与FM目标的关系，而非直接增加积分步数。本次不实施后续实验。",'',
        '## 限制与下一步判断','',
        '- E133训练16k、920643参数、纯FM Loss；E119训练36k、712275参数、原混合Loss，训练轮数和推理调用次数也不同。因此这里是实用质量参照，不能隔离FM的因果收益。',
        '- 当前每题单reference、单seed、单样本；不能据此声称学到了多优解分布。开发集长期重复使用，本次不是未见测试确认。',
        '- 先根据草稿退化项与五例积分敏感性区分模型目标和积分误差；不得仅凭FM Loss或重叠变化替代最终Q判断。不自动追加新Loss训练、reflow、seed、步数Q扫描或测试。','',
        '## 坏例与原始证据','',
        table(['index','N','FM最终Q','E119最终Q','完整ms','行回退'],[
            [r['index'],r['n'],f"{r['metrics']['cost']:.6f}",f"{bcases[r['index']]['metrics']['cost']:.6f}",
            f"{r['complete_seconds']*1000:.1f}",r['fallback']] for r in worst]),'',
        '- 逐例草稿/最终坐标、状态、硬检查、评分与完整后处理trace：`dev500/cases/`。',
        '- 汇总及分N/soft/耗时分项：`summary.json`；原件与身份检查：`verification.json`。',
        '- 五训练例4/8/16步轨迹：`dev500/five_train_step_sensitivity.json`。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(E133=fm,paired_Q=paired,draft_geometry={k:{'E119':v['E119']['mean'],'E133':v['E133']['mean']} for k,v in geometry.items()},sensitivity=sensitivity),ensure_ascii=False))
if __name__=='__main__':main()
