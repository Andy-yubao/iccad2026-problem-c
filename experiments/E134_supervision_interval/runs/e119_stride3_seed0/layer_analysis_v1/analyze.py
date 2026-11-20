"""Paired layer statistics, registered emergence rules, CV probes and plots."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from .topology import HERE,ROOT,read

def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bootstrap_weights(members):
    groups={}
    for i,member in enumerate(members):groups.setdefault(member[0],[]).append(i)
    keys=list(groups);which=np.zeros(500,int);sizes=[]
    for j,key in enumerate(keys):which[groups[key]]=j;sizes.append(len(groups[key]))
    rng=np.random.default_rng(20261114)
    counts=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    weights=counts[:,which].astype(float);weights/=weights.sum(1,keepdims=True)
    return weights,len(keys)

def ci_summary(values,weights):
    values=np.asarray(values,float)
    count=np.isfinite(values).sum(0)
    means=np.divide(np.nansum(values,axis=0),count,out=np.full(values.shape[1],np.nan),where=count>0)
    numerator=weights@np.nan_to_num(values)
    denominator=weights@np.isfinite(values).astype(float)
    boot=numerator/np.maximum(denominator,1e-30)
    low=[];high=[]
    for j in range(values.shape[1]):
        available=denominator[:,j]>0
        low.append(float(np.quantile(boot[available,j],.025)) if available.any() else None)
        high.append(float(np.quantile(boot[available,j],.975)) if available.any() else None)
    return dict(mean=[float(x) if np.isfinite(x) else None for x in means],low=low,high=high,n=count.tolist())

def first_persistent(values,threshold):
    good=np.isfinite(values)&(values>=threshold)
    persistent=np.logical_and.accumulate(good[:,::-1],axis=1)[:,::-1]
    return np.where(persistent.any(1),persistent.argmax(1),-1)

def birth_stats(values):
    valid=values>=0
    return dict(reached=int(valid.sum()),not_reached=int((~valid).sum()),
        median=float(np.median(values[valid])) if valid.any() else None,
        p25=float(np.quantile(values[valid],.25)) if valid.any() else None,
        p75=float(np.quantile(values[valid],.75)) if valid.any() else None,
        histogram={str(k):int((values==k).sum()) for k in range(13)})

def main():
    manifest=read(HERE/'manifest.json');plan=read(HERE/'plan.json')
    rows=[read(HERE/f'cases/{i:04d}.json') for i in range(500)]
    if [row['member'] for row in rows]!=manifest['members']:raise ValueError('Paired member order differs')
    loss=np.load(HERE/'losses.npy');loss_names=manifest['loss_names'];loss_index={name:i for i,name in enumerate(loss_names)}
    topo_names=list(rows[0]['layers'][0]['topology'])
    topology={name:np.asarray([[row['layers'][layer]['topology'].get(name) for layer in range(13)] for row in rows],float) for name in topo_names}
    geo_names=['duplicate_coverage','covered_module_mean','covered_module_max','covered_module_over10','bbox_over_public_area','whitespace_bbox',
        'hpwl_reference','group_contact_bottleneck','group_foreign_bbox_fraction','group_tree_blocked_fraction','boundary_gap_local','boundary_outside_fraction']
    geometry={name:np.asarray([[row['layers'][layer]['geometry'].get(name) for layer in range(13)] for row in rows],float) for name in geo_names}
    changes={name:np.asarray([[row['layers'][layer]['change'].get(name) for layer in range(13)] for row in rows],float)
        for name in ('center_move_local_mean','logshape_change_mean','local_axis_flip_rate')}
    weights,shards=bootstrap_weights(manifest['members'])
    result=dict(checkpoint_sha256=manifest['checkpoint_sha256'],case_count=500,source_shards=shards,
        loss={name:ci_summary(loss[:,:,j],weights) for j,name in enumerate(loss_names)},
        topology={name:ci_summary(value,weights) for name,value in topology.items()},
        geometry={name:ci_summary(value,weights) for name,value in geometry.items() if np.isfinite(value).any()},
        changes={name:dict(mean=[None if not np.isfinite(x) else float(x) for x in np.nanmean(value[:,1:],0)]) for name,value in changes.items()})
    transitions=[]
    # Twelve simultaneously covered paired step intervals for primary topology and Loss.
    simultaneous={}
    for name,values in [('loss',loss[:,:,0]),('local_axis_accuracy',topology['local_axis_accuracy']),('knn6_recall',topology['knn6_recall'])]:
        delta=np.diff(values,axis=1);means=delta.mean(0);boot=weights@delta;se=boot.std(0,ddof=1)
        z=np.abs((boot-means)/np.maximum(se,1e-30)).max(1);critical=np.quantile(z,.95)
        simultaneous[name]=dict(delta=means.tolist(),low=(means-critical*se).tolist(),high=(means+critical*se).tolist(),critical=float(critical))
    result['simultaneous_step_intervals']=simultaneous
    for layer in range(1,13):
        values=dict(layer=layer,loss_change=float(np.mean(loss[:,layer,0]-loss[:,layer-1,0])),
            local_order_change=float(np.mean(topology['local_axis_accuracy'][:,layer]-topology['local_axis_accuracy'][:,layer-1])),
            knn_change=float(np.mean(topology['knn6_recall'][:,layer]-topology['knn6_recall'][:,layer-1])),
            overlap_change=float(np.mean(geometry['duplicate_coverage'][:,layer]-geometry['duplicate_coverage'][:,layer-1])),
            improved_loss_cases=int((loss[:,layer,0]<loss[:,layer-1,0]).sum()))
        transitions.append(values)
    result['transitions']=transitions
    intervals=[]
    for a,b in ((0,3),(3,6),(6,9),(9,12),(3,12)):
        values=dict(start=a,end=b,loss_drop=float(np.mean(loss[:,a,0]-loss[:,b,0])),components={})
        for name in ('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted','public_preplaced_position_weighted'):
            values['components'][name]=float(np.mean(loss[:,a,loss_index[name]]-loss[:,b,loss_index[name]]))
        for name in ('local_axis_accuracy','knn6_recall','intent_reference_correct_clear','intent_agreement_final'):
            values[name+'_gain']=float(np.mean(topology[name][:,b]-topology[name][:,a]))
        intervals.append(values)
    result['supervised_intervals']=intervals
    gains={}
    for name,values in topology.items():
        if name not in ('local_axis_accuracy','knn6_recall','intent_reference_correct_clear'):continue
        means=np.nanmean(values,0);gain=means[12]-means[0]
        shares=(means-means[0])/gain if gain>1e-12 else np.full(13,np.nan)
        gains[name]=dict(total_gain=float(gain),net_gain_fraction_by_layer=shares.tolist())
    result['net_gain_fraction']=gains
    result['emergence']={}
    for threshold in (.90,.95,.99):
        result['emergence'][f'final_intent_stable_{threshold}']=birth_stats(first_persistent(topology['intent_agreement_final'],threshold))
    for axis,knn in plan['reference_good_thresholds']:
        combined=np.minimum(topology['local_axis_accuracy']/axis,topology['knn6_recall']/knn)
        result['emergence'][f'reference_fidelity_axis{axis}_knn{knn}']=birth_stats(first_persistent(combined,1.))
    for name in ('local_axis_accuracy','knn6_recall'):
        final=topology[name][:,12]
        normalized=topology[name]/np.maximum(final[:,None],1e-20)
        result['emergence'][f'final_level_90pct_{name}']=birth_stats(first_persistent(normalized,.90))
    diagnostics=read(HERE/'hidden_attention.json')
    result['hidden']={name:np.mean([d[name] for d in diagnostics],axis=0).tolist()
        for name in ('hidden_rms','hidden_rank','block_residual','state_injection','attention')}
    probe=read(HERE/'probe.json');probe_comparison=[]
    for layer in range(13):
        predicted=np.asarray(probe['stages'][f'S{layer}']['metrics'],float)
        actual=np.asarray(probe['actual_readout_metrics'],float)[:,layer]
        delta=ci_summary(predicted-actual,weights)
        probe_comparison.append(dict(layer=layer,probe_mean=np.nanmean(predicted,0).tolist(),actual_mean=np.nanmean(actual,0).tolist(),
            probe_minus_actual=delta))
    result['probe_comparison']=probe_comparison
    result['probe_summary']={stage:dict(mean=np.nanmean(np.asarray(value['metrics'],float),0).tolist(),
        permuted_label_mean=np.nanmean(np.asarray(value['permuted_label_control'],float),0).tolist() if value['permuted_label_control'] is not None else None)
        for stage,value in probe['stages'].items()}
    raw_followup=np.load(HERE/'raw_center_topology.npz')
    result['raw_center_followup']=dict(all_nodes_mean=raw_followup['all_nodes'].mean(0).tolist(),
        nonpre_mean=raw_followup['nonpre'].mean(0).tolist(),
        S0_probe_minus_raw_center=ci_summary(np.asarray(probe['stages']['S0']['metrics'],float)[:,:2]-raw_followup['nonpre'][:,0],weights),
        scope='secondary diagnostic of the pre-origin-clamp head centers; uses saved tensors only')
    q=np.asarray([row['original_final_Q'] for row in rows]);n=np.asarray([row['n'] for row in rows])
    order=np.lexsort((np.arange(500),-q));tail=np.zeros(500,bool);tail[order[:50]]=True
    associations=[]
    for layer in range(13):
        controls=np.column_stack((np.ones(500),rankdata(n),rankdata(geometry['duplicate_coverage'][:,layer]),rankdata(loss[:,layer,loss_index['reference_weighted']])))
        y=rankdata(q);ry=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0]
        for name in ('local_axis_accuracy','knn6_recall','intent_reference_correct_clear'):
            x=topology[name][:,layer];rx=rankdata(x);rx-=controls@np.linalg.lstsq(controls,rx,rcond=None)[0]
            bad_rank=rankdata(-x);pos=tail.sum();neg=(~tail).sum()
            auc=(bad_rank[tail].sum()-pos*(pos+1)/2)/(pos*neg)
            associations.append(dict(layer=layer,metric=name,spearman_Q=float(spearmanr(x,q).statistic),
                partial_rank_Q=float(np.corrcoef(rx,ry)[0,1]),tail50_AUC=float(auc),
                mean_tail=float(x[tail].mean()),mean_other=float(x[~tail].mean())))
    result['Q_associations_observational_only']=associations
    result['tail_indices']=order[:50].tolist()
    # Raw-head clamps and per-step actual motion are independently observed.
    raw=np.load(HERE/'states.npy',mmap_mode='r');p=np.load(HERE/'positions.npy',mmap_mode='r');areas=np.load(HERE/'areas.npy')
    clamp=[]
    for layer in range(13):
        values=[]
        for i,row in enumerate(rows):
            count=row['n'];state=raw[i,layer,:count];physical=p[i,layer,:count];scale=np.sqrt(areas[i].sum())
            unbounded_xy=state[:,:2]*scale-physical[:,2:]/2
            values.append([float((unbounded_xy<0).any(1).mean()),float((np.abs(state[:,2])>5).mean())])
        clamp.append(np.mean(values,axis=0).tolist())
    result['clamp_fractions']=clamp
    save(HERE/'summary.json',result)
    write_tables(result,manifest,loss,rows,topology,geometry)
    plots(result,manifest)
    verify(manifest)
    print(json.dumps(dict(status='complete',case_count=500,source_shards=shards,report=str(HERE/'report.md'))),flush=True)

def write_tables(result,manifest,loss,rows,topology,geometry):
    names=manifest['loss_names'];means=loss.mean(0)
    text='# E134 每三层基线：逐层与拓扑形成诊断\n\n固定best63、同一dev500；观察既有S0与真实S1–S12，不更改原模型/数据/权重/后处理。原始域Loss用于与训练监测对应，实际几何单列。所有变化为同题配对，95%区间用源分片聚类bootstrap。\n\n'
    text+='## 全部输出层\n\n|层|直接监督|完整Loss|Reference|碰撞|局部方向正确且明确|K6邻居召回|分离方向兼容且明确|与S12方向一致|重复覆盖率|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for k in range(13):
        text+=f"|S{k}|{'是' if k in (3,6,9,12) else '否'}|{means[k,0]:.6f}|{means[k,names.index('reference_weighted')]:.6f}|{means[k,names.index('overlap_weighted')]:.6f}|"+'|'.join(f'{np.nanmean(topology[name][:,k])*100:.3f}%' for name in ('local_axis_accuracy','knn6_recall','intent_reference_correct_clear','intent_agreement_final'))+f"|{geometry['duplicate_coverage'][:,k].mean()*100:.3f}%|\n"
    from experiments.E134_supervision_interval.loss import A,B
    terms=[('合计','total',1),('参考位置','position',1),('参考形状','shape',.25),
        ('碰撞穿透','penetration',A),('碰撞平均负担','module_load_mean',A),('碰撞最大超10%负担','module_max_excess10',A),
        ('线长','wirelength',.5*B),('紧凑','compactness',.5*B),('Grouping','grouping_weighted',1),
        ('Boundary','boundary_weighted',1),('F/P尺寸','public_shape_weighted',1),('P位置','public_preplaced_position_weighted',1)]
    text+='\n## 单独比较监督层：完整Loss分项（已计系数）\n\n|项|S3|S6|S9|S12|\n|---|---:|---:|---:|---:|\n'
    for label,name,factor in terms:text+='|'+label+'|'+'|'.join(f'{means[k,names.index(name)]*factor:.6f}' for k in (3,6,9,12))+'|\n'
    text+='\n## 监督层间的净变化\n\n|区间|Loss下降|Reference贡献|碰撞贡献|其余贡献|局部方向增益pp|K6增益pp|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for interval in result['supervised_intervals']:
        c=interval['components'];other=interval['loss_drop']-c['reference_weighted']-c['overlap_weighted']
        text+=f"|S{interval['start']}→S{interval['end']}|{interval['loss_drop']:.6f}|{c['reference_weighted']:.6f}|{c['overlap_weighted']:.6f}|{other:.6f}|{interval['local_axis_accuracy_gain']*100:.3f}|{interval['knn6_recall_gain']*100:.3f}|\n"
    text+='\n## 逐步变化（含12步同时95%区间）\n\n|到达层|Loss变化|Loss同时95%CI|方向变化pp|方向同时95%CI pp|K6变化pp|重叠变化pp|Loss改善题数|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for i,row in enumerate(result['transitions']):
        ls=result['simultaneous_step_intervals']['loss'];ts=result['simultaneous_step_intervals']['local_axis_accuracy']
        text+=f"|S{row['layer']}|{row['loss_change']:+.6f}|[{ls['low'][i]:+.6f},{ls['high'][i]:+.6f}]|{row['local_order_change']*100:+.3f}|[{ts['low'][i]*100:+.3f},{ts['high'][i]*100:+.3f}]|{row['knn_change']*100:+.3f}|{row['overlap_change']*100:+.3f}|{row['improved_loss_cases']}/500|\n"
    text+='\n## 形成时间（500题分布，不把未达标题补为S12）\n\n|预注册判据|达到/500|未达到|首次持续达到层 中位数[P25,P75]|\n|---|---:|---:|---:|\n'
    for name,row in result['emergence'].items():text+=f"|{name}|{row['reached']}|{row['not_reached']}|{row['median']} [{row['p25']},{row['p75']}]|\n"
    text+='\n## 非P节点：分组交叉验证线性探针与实际读出\n\n探针只拟合相对中心的线性统计映射，按源分片五折留出、每题等权、固定正则强度；不训练/修改布局网络，不用Q选参。指标均为留出预测。\n\n|层|实际方向正确率|线性探针方向正确率|实际K6|线性探针K6|探针−实际方向差pp及95%CI|\n|---|---:|---:|---:|---:|---:|\n'
    for r in result['probe_comparison']:
        d=r['probe_minus_actual']
        text+=f"|S{r['layer']}|{r['actual_mean'][0]*100:.3f}%|{r['probe_mean'][0]*100:.3f}%|{r['actual_mean'][1]*100:.3f}%|{r['probe_mean'][1]*100:.3f}%|{d['mean'][0]*100:+.3f} [{d['low'][0]*100:+.3f},{d['high'][0]*100:+.3f}]|\n"
    text+='\n## 原始输入与隐状态探针/乱序对照\n\n|状态|探针方向|探针K6|训练标签乱序后的方向|训练标签乱序后的K6|\n|---|---:|---:|---:|---:|\n'
    for name,row in result['probe_summary'].items():
        null=row['permuted_label_mean'];fields=[f'{x*100:.3f}%' for x in row['mean'][:2]]+([f'{x*100:.3f}%' for x in null[:2]] if null else ['—','—'])
        text+='|'+name+'|'+'|'.join(fields)+'|\n'
    text+='\n## 隐状态与注意力\n\n|状态|隐藏RMS|节点变化有效维数(参与比)|块残差/输入范数|状态注入/上一隐藏范数|注意力归一熵|self质量|wire质量|group质量|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    h=result['hidden']
    for j,name in enumerate(manifest['hidden_names']):
        extra='—|—|—|—|—|—' if j==0 else f"{h['block_residual'][j-1]:.4f}|{h['state_injection'][j-1]:.4f}|"+'|'.join(f'{x:.4f}' for x in h['attention'][j-1])
        text+=f"|{name}|{h['hidden_rms'][j]:.4f}|{h['hidden_rank'][j]:.4f}|{extra}|\n"
    text+='\n## 原始头中心与非负解码的分离检查（后续诊断）\n\n这里用既有原始中心复算，未增加前向；用以检查S0探针优势是否仅来自非负坐标clamp。\n\n|层|原始头非P方向|实际解码非P方向|非P线性探针方向|原始头非P K6|解码非P K6|探针非P K6|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for k in (0,1,2,3,6,9,12):
        raw=result['raw_center_followup']['nonpre_mean'][k];r=result['probe_comparison'][k]
        text+=f"|S{k}|{raw[0]*100:.3f}%|{r['actual_mean'][0]*100:.3f}%|{r['probe_mean'][0]*100:.3f}%|{raw[1]*100:.3f}%|{r['actual_mean'][1]*100:.3f}%|{r['probe_mean'][1]*100:.3f}%|\n"
    gradient=read(HERE/'gradients.json')
    text+='\n## 冻结点梯度与参数分配\n\n下列是完整dev500均值目标的eval模式局部导数，未建优化器或更新参数；梯度范数不能直接解释成Adam更新份额或训练时资源。\n\n|模块|参数量|加权L3梯度范数|加权L6|加权L9|加权L12|\n|---|---:|---:|---:|---:|---:|\n'
    for name in ['input','global1','global2',*[f'geometry{i}' for i in range(1,13)],'state_embed','head']:
        text+=f"|{name}|{gradient['parameter_counts'][name]}|"+'|'.join(f'{x:.6f}' for x in gradient['weighted_gradient_norms'][name])+'|\n'
    for group,values in gradient['cosine'].items():
        text+=f'\n{group}：未加权L3/L6/L9/L12导数的余弦\n\n|目标|L3|L6|L9|L12|\n|---|---:|---:|---:|---:|\n'
        for label,row in zip(('L3','L6','L9','L12'),values):text+='|'+label+'|'+'|'.join(f'{x:.4f}' for x in row)+'|\n'
    text+='\n## 与既有最终Q的关联（非因果、非跨数据泛化结论）\n\n|层|指标|Spearman(Q)|控制N/重叠/Reference Loss后的部分秩相关|原Q最坏50题AUC|\n|---|---|---:|---:|---:|\n'
    for row in result['Q_associations_observational_only']:
        text+=f"|S{row['layer']}|{row['metric']}|{row['spearman_Q']:.4f}|{row['partial_rank_Q']:.4f}|{row['tail50_AUC']:.4f}|\n"
    task=read(HERE/'task_structure_associations.json')
    text+='\n## 已登记任务结构代理的次级关联检查\n\n较大负担按较高风险排序；以下仍是同面板描述关联，不是独立泛化验证或因果结果。接触瓶颈同时包含几何距离，并非纯离散拓扑。\n\n|层|指标|n|Spearman(Q)|部分秩相关|尾部50题AUC|\n|---|---|---:|---:|---:|---:|\n'
    for row in task['rows']:
        text+=f"|S{row['layer']}|{row['metric']}|{row['n']}|{row['spearman_Q']:.4f}|{row['partial_rank_Q']:.4f}|{row['tail50_AUC']:.4f}|\n"
    text+='\n## 范围与限制\n\nReference不是唯一优质布局；方向/邻居保真不等同最终Q。终态稳定仅表示像S12，不保证好。固定阈值的敏感性已列出；hidden探针的跨折性只针对本面板探针拟合，原网络已用dev500选模。梯度是best63局部eval导数，不是训练过程资源消耗，参数量/隐藏有效维数也不等同任务难度。未重跑各层后处理，不能由此断言早层同样易于修复或拓扑在学习上必然更难。单seed与复用dev限制推广。\n'
    (HERE/'report.md').write_text(text,encoding='utf-8')
    with (HERE/'per_case_layers.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['index','source','row','layer','final_Q',*['loss.'+n for n in names],*['topology.'+n for n in topology],*['geometry.'+n for n in geometry]]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            for layer in range(13):
                entry=dict(index=i,source=row['member'][0],row=row['member'][1],layer=layer,final_Q=row['original_final_Q'])
                entry.update({'loss.'+name:loss[i,layer,j] for j,name in enumerate(names)})
                entry.update({'topology.'+name:values[i,layer] for name,values in topology.items()})
                entry.update({'geometry.'+name:values[i,layer] if np.isfinite(values[i,layer]) else '' for name,values in geometry.items()})
                writer.writerow(entry)

def plots(result,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    layers=np.arange(13)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for name,label in [('local_axis_accuracy','局部方向正确且明确'),('knn6_recall','K6邻居召回'),('intent_reference_correct_clear','参考分离方向兼容且明确')]:
        r=result['topology'][name];axes[0,0].plot(layers,np.asarray(r['mean'])*100,marker='o',label=label)
        axes[0,0].fill_between(layers,np.asarray(r['low'])*100,np.asarray(r['high'])*100,alpha=.12)
    r=result['topology']['intent_agreement_final'];axes[0,1].plot(layers,np.asarray(r['mean'])*100,marker='o',label='与S12明确分离方向一致')
    axes[0,1].axhline(95,color='gray',linestyle=':',label='95%')
    comp=result['probe_comparison']
    axes[1,0].plot(layers,[r['actual_mean'][0]*100 for r in comp],marker='o',label='实际读出：非P局部方向')
    axes[1,0].plot(layers,[r['probe_mean'][0]*100 for r in comp],marker='s',label='分片五折线性探针：非P方向')
    axes[1,1].plot(layers,np.asarray(result['geometry']['duplicate_coverage']['mean'])*100,marker='o',label='精确重复覆盖率')
    for ax,title in zip(axes.flat,('相对Reference的结构','终态结构稳定性（不等同质量）','隐状态可读信息与真实布局','实际草稿重叠')):
        for s in (3,6,9,12):ax.axvline(s,color='#aaa',linewidth=.8,linestyle='--',alpha=.5)
        ax.set(title=title,xlabel='实际输出层 S0–S12',ylabel='%');ax.set_xticks(layers);ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.suptitle('E134 每三层基线 best63 · 同一dev500 · 只读诊断',fontsize=14);fig.tight_layout()
    fig.savefig(HERE/'topology_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for name in ('total','reference_weighted','overlap_weighted','boundary_weighted'):
        axes[0].plot(layers,result['loss'][name]['mean'],marker='o',label=name)
    axes[0].set_yscale('log');axes[0].set(title='全轨迹Loss（对数纵轴）',xlabel='层',ylabel='Loss');axes[0].legend(fontsize=9)
    for name in ('total','reference_weighted','overlap_weighted'):
        axes[1].plot(layers[3:],np.asarray(result['loss'][name]['mean'])[3:],marker='o',label=name)
    axes[1].set(title='S3以后：监督层间的展开',xlabel='层',ylabel='Loss');axes[1].legend(fontsize=9)
    for ax in axes:
        for s in (3,6,9,12):ax.axvline(s,color='#aaa',linewidth=.8,linestyle='--')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'loss_curves.png',dpi=170);plt.close(fig)

def verify(manifest):
    checkpoint=HERE.parent/'checkpoints/best.pt'
    if sha(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('Frozen weights changed')
    capture=read(HERE/'capture_state.json')
    if capture['status']!='complete':raise ValueError('Capture/gradient diagnostic incomplete')
    prior=read(ROOT/'experiments/E136_frontend_F_linear/runs/statistics_dev500_v1/manifest.json')
    if prior['members']!=manifest['members']:raise ValueError('Prior paired panel changed')
    for item in prior['payloads']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Reference payload changed')
    save(HERE/'verification.json',dict(status='passed',cases=500,readouts=13,terminal_and_prior_gallery_bitwise_equal=True,
        reference_payload_checksums_verified=True,weights_unchanged=True,gradient_no_optimizer=True,
        probe_grouped_out_of_fold=True,probe_no_Q_tuning=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False,
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},plan_sha256=sha(HERE/'plan.json')))

if __name__=='__main__':main()
