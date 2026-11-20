"""Summarize fixed-weight unsupervised-layer diagnostics and retain all cases."""
import csv
import json
from pathlib import Path
import numpy as np
from .audit_unsupervised import OUT,PRIOR,RUN,ROOT,OWNER,LAYERS,UNSUP,read,save,sha,check,persistent

def main():
    plan,manifest=check()
    rows=[read(OUT/f'cases/{i:04d}.json') for i in range(500)]
    assert [r['member'] for r in rows]==manifest['members']
    rng=np.random.default_rng(20261115)
    groups=sorted({m[0] for m in manifest['members']})
    ids=np.array([groups.index(m[0]) for m in manifest['members']]);counts=np.bincount(ids)
    draws=rng.integers(0,len(groups),(2000,len(groups)))
    def paired(delta):
        sums=np.bincount(ids,weights=delta,minlength=len(groups))
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        return dict(mean=float(np.mean(delta)),ci95=np.quantile(boot,[.025,.975]).tolist())
    motion=[]
    for k,layer in enumerate(range(4,13)):
        items=[r['motion'][k] for r in rows]
        means={key:float(np.mean([item[key] for item in items])) for key in items[0] if key!='layer'}
        motion.append(dict(layer=layer,**means,
            full_side_exchanges_total=sum(item['full_side_exchanges'] for item in items),
            full_side_exchange_cases=sum(item['full_side_exchanges']>0 for item in items),
            local_full_side_exchanges_total=sum(item['local_full_side_exchanges'] for item in items)))
    all_edges=[dict(case=r['index'],**edge) for r in rows for edge in r['group_edges']]
    bad_edges=[edge for edge in all_edges if edge['initial_bad']]
    boundaries=[dict(case=r['index'],**b) for r in rows for b in r['boundary']]
    bad_b=[b for b in boundaries if b['initial_bad']]
    blocked_b=[b for b in bad_b if b['initial_bad_blocked']]
    def events(items,key):
        return dict(total=len(items),persistent_repaired=sum(item[key] is not None for item in items),
            remained=sum(item[key] is None for item in items),
            first_persistent_layer={str(layer):sum(item[key]==layer for item in items) for layer in LAYERS})
    group=dict(all_fixed_edges=len(all_edges),initial_bad_edges=len(bad_edges),
        initial_bad_cases=len({edge['case'] for edge in bad_edges}),
        all_clear=events(bad_edges,'all_clear_layer'),
        clear_and_contact_halved=events(bad_edges,'clear_and_contact_halved_layer'),
        original_blockers_clear=events(bad_edges,'original_clear_layer'),
        any_new_blocker_by_S12=sum(edge['new_blocker_counts'][-1]>0 for edge in bad_edges),
        initial_bad_still_blocked_fraction=[float(np.mean([edge['blocker_counts'][k]>0 for edge in bad_edges])) for k in range(10)],
        initial_bad_contact_mean=[float(np.mean([edge['contact'][k] for edge in bad_edges])) for k in range(10)])
    bsummary=dict(all_required_sides=len(boundaries),initial_bad=len(bad_b),initial_bad_blocked=len(blocked_b))
    for name,items in [('bad',bad_b),('bad_blocked',blocked_b)]:
        bsummary[name]=dict(gap_mean=[float(np.mean([b['gaps'][k] for b in items])) for k in range(10)],
            shadow_fraction=[float(np.mean([b['shadows'][k]>0 for b in items])) for k in range(10)],
            repairs={str(threshold):events([dict(b,repair=b['repair_layers'][str(threshold)] ) for b in items],'repair') for threshold in (.01,.02,.05)})
    void=np.array([[v['bbox_fraction'] for v in r['void']] for r in rows])
    cohort=np.lexsort((np.arange(500),-void[:,0]))[:100]
    void_summary=dict(mean=void.mean(0).tolist(),
        reference_mean=float(np.mean([r['reference_void']['bbox_fraction'] for r in rows])),
        high_S3_indices=cohort.tolist(),high_S3_mean=void[cohort].mean(0).tolist(),
        high_S3_reference_mean=float(np.mean([rows[i]['reference_void']['bbox_fraction'] for i in cohort])),
        paired_S12_minus_S3=paired(void[:,-1]-void[:,0]),
        high_S3_reduction20pct_halfpp_persistent=sum(persistent((void[i]<=.8*void[i,0])&(void[i]<=void[i,0]-.005)) is not None for i in cohort))
    void_summary['high_S3_persistent_reduction_layer']={str(layer):sum(persistent((void[i]<=.8*void[i,0])&(void[i]<=void[i,0]-.005))==layer for i in cohort) for layer in LAYERS}
    geometry={key:[float(np.mean([r['geometry'][k][key] for r in rows])) for k in range(10)] for key in
        ('bbox_over_public_area','whitespace_bbox','duplicate_coverage','group_contact_bottleneck','group_foreign_bbox_fraction','boundary_gap_local','boundary_outside_fraction')}
    windows=[]
    for j,anchor in enumerate((3,6,9)):
        values=[r['windows'][j] for r in rows]
        large=sum(v['large_excursion_modules'] for v in values);undone=sum(v['undone_modules'] for v in values)
        win=dict(anchor=anchor,end=anchor+3,loss_sacrifice_cases=sum(v['loss_sacrifice'] for v in values),
            large_excursion_modules=large,undone_modules=undone,undone_fraction=undone/max(large,1),
            full_exchange_events_unsupervised=sum(e['layer'] in (anchor+1,anchor+2) for r in rows for e in r['exchanges']),
            fixed_barriers_persistently_cleared_unsupervised=sum(e['all_clear_layer'] in (anchor+1,anchor+2) for e in bad_edges),
            blocked_boundary_persistently_repaired_unsupervised=sum(b['repair_layers']['0.01'] in (anchor+1,anchor+2) for b in blocked_b))
        windows.append(win)
    # Event-level persistence and direction of rare order reversals, fixed S3 pairs.
    event_stats=dict(total=sum(len(r['exchanges']) for r in rows),
        cases_any=sum(bool(r['exchanges']) for r in rows),
        unsupervised=sum(e['layer'] in UNSUP for r in rows for e in r['exchanges']),
        supervised=sum(e['layer'] not in UNSUP for r in rows for e in r['exchanges']),
        mean_S3_S12_local_order_reversal=float(np.mean([r['cumulative']['local_final_order_reversal_rate'] for r in rows])))
    old=np.load(PRIOR/'losses.npy');names=manifest['loss_names']
    ablations={}
    if (OUT/'interventions.json').exists():
        weights=np.zeros(13);weights[[3,6,9]]=.2/3;weights[12]=.8
        modes=dict(read(OUT/'interventions.json')['modes'])
        if (OUT/'rebound_interventions.json').exists():modes.update(read(OUT/'rebound_interventions.json')['modes'])
        for name,mode in modes.items():
            new=np.load(OUT/f'{name}_losses.npy')
            delta=new[:,12,0]-old[:,12,0]
            ablations[name]=dict(mean_final_loss=float(new[:,12,0].mean()),paired_delta=paired(delta),
                improved=int((delta<0).sum()),worsened=int((delta>0).sum()),unchanged=int((delta==0).sum()),
                weighted_objective=float((new[:,:,0]@weights).mean()),
                supervised_means={str(layer):float(new[:,layer,0].mean()) for layer in (3,6,9,12)},
                component_means={key:float(new[:,12,j].mean()) for j,key in enumerate(names)})
        prior_skip=read(RUN/'skip_layers_01236912_v1/results.json')
        assert prior_skip['verification']['checkpoint_sha256']==manifest['checkpoint_sha256']
        value=prior_skip['comparison']['S12']
        ablations['skip_all_prior']=dict(mean_final_loss=value['skipped'],paired_delta=dict(mean=value['delta'],ci95=value['delta_paired_source_bootstrap95']),
            improved=value['improved'],worsened=value['worsened'],unchanged=value['unchanged'],
            scope='Reused earlier fixed-weight all-six-block skip; same panel/weights, original source gate verified')
    result=dict(layers=LAYERS,motion=motion,exchanges=event_stats,group=group,boundary=bsummary,void=void_summary,
        geometry=geometry,windows=windows,ablations=ablations,
        case_count=500,source_shards=len(groups),checkpoint_sha256=manifest['checkpoint_sha256'])
    if (OUT/'followup.json').exists():
        followup=read(OUT/'followup.json')
        result['followup_summary']=dict(passage={key:{name:value[name] for name in ('count','cases','crossing_unsupervised','completion_unsupervised')} for key,value in followup['passage_sensitivity'].items()},
            overlap_order_crossings=followup['overlap_order_crossings'],
            clear_and_group_path_halved_count=followup['clear_and_group_path_halved_count'],
            reference_center_line_controls=followup['reference_center_line_controls'],
            maximum_empty_rectangle_checks=followup['maximum_empty_rectangle_checks'])
    save(OUT/'summary.json',result)
    with (OUT/'per_case_layers.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(['case','layer','largest_empty_bbox_fraction','bbox_over_public_area','whitespace_bbox','fixed_initial_barriers_remaining','initial_bad_boundary_gap_mean'])
        for r in rows:
            initial=[e for e in r['group_edges'] if e['initial_bad']];bad=[b for b in r['boundary'] if b['initial_bad']]
            for k,layer in enumerate(LAYERS):writer.writerow([r['index'],layer,r['void'][k]['bbox_fraction'],r['geometry'][k]['bbox_over_public_area'],r['geometry'][k]['whitespace_bbox'],sum(e['blocker_counts'][k]>0 for e in initial),np.mean([b['gaps'][k] for b in bad]) if bad else ''])
    write_report(result)
    examples(rows,result)
    plot(result)
    check()
    save(OUT/'verification.json',dict(status='passed',cases=500,source_shards=len(groups),
        original_weights_and_sources_unchanged=True,original_positions_sha256=sha(PRIOR/'positions.npy'),
        observations=read(OUT/'observations.json'),interventions=read(OUT/'interventions.json') if ablations else None,
        followup_passed=(OUT/'followup.json').exists(),rebound_interventions=read(OUT/'rebound_interventions.json') if (OUT/'rebound_interventions.json').exists() else None,
        training=False,postprocessing_calls=0,Q_calls=0,test_read=False,plan_sha256=sha(OUT/'plan.json'),
        report_source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status='complete',cases=500,ablations=len(ablations),
        complete_side_exchanges=event_stats['total'],report=str(OUT/'report.md')),ensure_ascii=False),flush=True)

def write_report(s):
    text=['# E134 每三层基线：非监督层结构审计','',
        'best63，同一dev500。只审计S3之后；原权重、原Loss、原后处理不变。换侧指标是离散输出的量化代理，不宣称观测到了网络内部连续移动路径。','',
        '## 逐层移动和换位','',
        '|到达层|去平移后平均位移/典型边长|大位移≥0.5模块占比|明确局部轴次序翻转|完成完整换侧事件数|有完整换侧题数|',
        '|---|---:|---:|---:|---:|---:|']
    for v in s['motion']:text.append(f"|S{v['layer']}|{v['translation_removed_move_mean']:.4f}|{100*v['large_move05']:.3f}%|{100*v['local_clear_order_reversal_rate']:.3f}%|{v['full_side_exchanges_total']}|{v['full_side_exchange_cases']}/500|")
    text+=['','典型边长定义为sqrt(模块公开面积)。完整换侧允许跨越多层，表中计入完成层；并不意味着中心次序在完成层才发生翻转。','',
        '## 固定S3阻隔追踪','',
        f"固定7896条S3同组接触MST边，初始存在中心连线阻隔且接触距离>0.05的边共{s['group']['initial_bad_edges']}条，涉及{s['group']['initial_bad_cases']}题。中心连线受阻是几何代理，不等于所有路径受阻。",'',
        '|持续到S12的结果|边数|占初始问题边|','|---|---:|---:|']
    for label,key in [('中心连线不再受任何异组模块阻隔','all_clear'),('解除阻隔且原边接触距离至少减半','clear_and_contact_halved'),('原阻隔模块退出，不要求无新阻隔','original_blockers_clear')]:
        v=s['group'][key];text.append(f"|{label}|{v['persistent_repaired']}/{v['total']}|{100*v['persistent_repaired']/v['total']:.2f}%|")
    if 'followup_summary' in s:
        n=s['followup_summary']['clear_and_group_path_halved_count'];text.append(f'|解除阻隔且允许替换同组链路后的连接瓶颈至少减半|{n}/1170|{100*n/1170:.2f}%|')
    text+=['','固定初始问题边的平均直接接触距离从0.21555到0.25168；原边可以被另一条同组链路替换，因此不能把直接距离变大都视为分组变差。','',
        '## 固定S3贴边要求追踪','',
        f"共有{s['boundary']['all_required_sides']}条模块-边要求，其中S3边距>0.1典型边长的有{s['boundary']['initial_bad']}条，且有外向遮挡的有{s['boundary']['initial_bad_blocked']}条。以下都是近边阈值，不等于硬合法。",'',
        '|固定S3问题集合|S12边距≤1%典型边长|≤2%|≤5%|S3→S12平均边距|','|---|---:|---:|---:|---:|']
    for key,label in [('bad','全部初始大边距'),('bad_blocked','初始大边距且受遮挡')]:
        v=s['boundary'][key];numbers=[v['repairs'][str(t)]['persistent_repaired'] for t in (.01,.02,.05)];total=v['repairs']['0.01']['total']
        text.append('|'+label+'|'+'|'.join(f'{n}/{total} ({100*n/total:.2f}%)' for n in numbers)+f"|{v['gap_mean'][0]:.5f}→{v['gap_mean'][-1]:.5f}|")
    text+=['',
        '## 空隙与包围框','',
        '|层|最大空矩形/外框|S3空隙最大100题|外框/总面积|精确空白率|重叠覆盖率|',
        '|---|---:|---:|---:|---:|---:|']
    for k,layer in enumerate(LAYERS):text.append(f"|S{layer}|{100*s['void']['mean'][k]:.3f}%|{100*s['void']['high_S3_mean'][k]:.3f}%|{s['geometry']['bbox_over_public_area'][k]:.4f}|{100*s['geometry']['whitespace_bbox'][k]:.3f}%|{100*s['geometry']['duplicate_coverage'][k]:.3f}%|")
    text+=['','## 固定权重干预','',
        'skip仅删除指定几何块。hold保留所有隐状态计算，但让后续层继续使用上一监督层的状态反馈，直到下一监督输出才接受新状态；它同时改变state embedding与几何偏置，不是只删除视觉穿越事件。所有干预均会使轨迹偏离训练分布。','',
        '|干预|S12 Loss|相对原模型增加|95%配对分片区间|改善/变差/相同题数|',
        '|---|---:|---:|---|---:|']
    for name,v in s['ablations'].items():text.append(f"|{name}|{v['mean_final_loss']:.6f}|{v['paired_delta']['mean']:+.6f}|{v['paired_delta']['ci95']}|{v['improved']}/{v['worsened']}/{v['unchanged']}|")
    text+=['','## 暂时退步与后续改善','',
        '|窗口|中间Loss高于起点但终点改善题数|中间大位移模块|其中终点收回至少一半位移|非监督层持续解除初始阻隔数|',
        '|---|---:|---:|---:|---:|']
    for v in s['windows']:text.append(f"|S{v['anchor']}→S{v['end']}|{v['loss_sacrifice_cases']}/500|{v['large_excursion_modules']}|{v['undone_modules']} ({100*v['undone_fraction']:.2f}%)|{v['fixed_barriers_persistently_cleared_unsupervised']}|")
    if 'followup_summary' in s:
        text+=['','## 次级敏感性检查','',
            '完整换侧的间隙阈值取0.005、0.02、0.05时均为2次、2题。两次中心次序翻转均发生在S5，分别到S6、S12才完成反向分离。',
            '更宽松的“仍在重叠时交换中心次序”明显更多，不能将完整换侧稀少解释为没有局部交错。','',
            '|到达层|重叠状态次序交换事件数，5%边长裕量|涉及题数|涉及模块占比，逐题均值|',
            '|---|---:|---:|---:|']
        for v in s['followup_summary']['overlap_order_crossings']['0.05']:text.append(f"|S{v['layer']}|{v['total']}|{v['cases']}/500|{100*v['mean_module_fraction']:.3f}%|")
        text+=['','最大空矩形已通过3个小布局的穷举对照和80个实际布局的无占据交叉检查；结果见followup.json。']
    text+=['',
        '全部数值：summary.json；原始事件：cases/；逐题逐层表：per_case_layers.csv；定义：plan.json。',
        '本面板已用于原选模。结果只适用于当前冻结模型；不等于少层重训的结果，不建立最终Q或后处理可修复性的因果结论。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

def examples(rows,s):
    from visualization.layouts import plot_case
    positions=np.load(PRIOR/'positions.npy',mmap_mode='r');flags=np.load(PRIOR/'flags.npy');ref=np.load(PRIOR/'reference.npy')
    initial=np.array([sum(e['initial_bad'] for e in r['group_edges']) for r in rows])
    remained=np.array([sum(e['initial_bad'] and e['all_clear_layer'] is None for e in r['group_edges']) for r in rows])
    cleared=np.array([sum(e['initial_bad'] and e['all_clear_layer'] is not None for e in r['group_edges']) for r in rows])
    choices=[]
    eligible=np.flatnonzero(remained>0)
    if len(eligible):choices.append(('persistent_barrier',int(min(eligible,key=lambda i:(abs(initial[i]-np.median(initial[eligible])),i)))))
    choices.append(('most_barriers_cleared',int(np.argmax(cleared))))
    eligible=[i for i in s['void']['high_S3_indices'] if rows[i]['void'][-1]['bbox_fraction']>=.8*rows[i]['void'][0]['bbox_fraction']]
    if eligible:choices.append(('persistent_void',int(max(eligible,key=lambda i:(rows[i]['void'][0]['bbox_fraction'],-i)))))
    for i in (171,191):choices.append((f'rare_side_exchange_{i}',i))
    selected=[]
    for name,index in choices:
        n=rows[index]['n'];stages=[('reference','Reference')]+[(f'S{k}',f'S{k}') for k in (3,4,5,6,12)]
        variants=dict(reference=dict(positions=ref[index,:n].tolist()))
        variants.update({f'S{k}':dict(positions=positions[index,k,:n].tolist()) for k in (3,4,5,6,12)})
        plot_case(dict(title=f'dev {index:04d} | {name} | actual saved states',constraints=flags[index,:n].tolist(),variants=variants),OUT/f'{name}.png',stages=stages)
        selected.append(dict(name=name,index=index,initial_bad_edges=int(initial[index]),persistent_bad_edges=int(remained[index]),cleared_bad_edges=int(cleared[index])))
    save(OUT/'example_selection.json',selected)

def plot(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.array(LAYERS)
    ax=axes[0,0];m=s['motion']
    ax.plot(x[1:],[v['large_move05']*100 for v in m],'o-',label='位移≥0.5典型边长')
    ax.plot(x[1:],[v['local_clear_order_reversal_rate']*100 for v in m],'s-',label='局部明确轴次序翻转')
    ax.set(title='S3之后以多大动作改布局',ylabel='模块占比 / 模块对轴比例 %');ax.legend()
    ax=axes[0,1]
    ax.plot(x,np.array(s['group']['initial_bad_still_blocked_fraction'])*100,'o-',label='S3阻隔边仍被阻隔')
    ax.plot(x,np.array(s['boundary']['bad_blocked']['shadow_fraction'])*100,'s-',label='S3差且受阻贴边项仍有遮挡')
    ax.set(title='固定初始问题的存续率',ylabel='%');ax.legend()
    ax=axes[1,0]
    ax.plot(x,np.array(s['void']['mean'])*100,'o-',label='全部500题')
    ax.plot(x,np.array(s['void']['high_S3_mean'])*100,'s-',label='固定S3最大空隙100题')
    ax.axhline(s['void']['reference_mean']*100,ls=':',color='gray',label='Reference均值')
    ax.set(title='最大空矩形 / 当前布局外框',ylabel='%');ax.legend()
    ax=axes[1,1]
    ab=s['ablations'];names=[n for n in ('original','skip45','skip78','skip1011','hold4','hold7','hold10','hold_rebounds') if n in ab]
    ax.bar(np.arange(len(names)),[ab[n]['mean_final_loss'] for n in names],color=['#2563eb' if n=='original' else '#d97706' if n.startswith('skip') else '#16a34a' for n in names])
    labels={'original':'原模型','skip45':'跳过4/5','skip78':'跳过7/8','skip1011':'跳过10/11','hold4':'固定S4反馈','hold7':'固定S7反馈','hold10':'固定S10反馈','hold_rebounds':'固定S4/7/10反馈'}
    ax.set_xticks(np.arange(len(names)),[labels[n] for n in names],rotation=35,ha='right',fontsize=9);ax.set(title='删层 vs 只固定反弹输出的反馈',ylabel='末层完整Loss，越低越好')
    for ax in axes.ravel():
        ax.grid(alpha=.2);ax.set_axisbelow(True)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.set_xlabel('输出层');ax.set_xticks(x)
        for layer in (3,6,9,12):ax.axvline(layer,color='gray',ls='--',alpha=.2)
    fig.suptitle('E134 best63 · 同dev500 · 非监督层结构审计',fontsize=16)
    fig.tight_layout();fig.savefig(OUT/'audit_curves.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
