"""Read saved E152 v2 statistics; capture requested real readouts only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from shapely.geometry import box
from shapely.ops import unary_union
from datasets.random_reference import RandomReference
from experiments.E152_s2_local_repair.io import sources
from experiments.E152_s2_local_repair.evaluation.run import Predictor, public_graph
from experiments.E152_s2_local_repair.batches import decode
from experiments.E152_s2_local_repair.data import load_rows
from experiments.E152_s2_local_repair.resident import ResidentRows
from experiments.E152_s2_local_repair.base_loss import base_per_case
from experiments.E152_s2_local_repair.loss import drift_terms
from visualization.style import normalize
from visualization.layouts import plot_case

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
ROOT = RUNS.parents[2]
ARMS = ('e152_s2_ref', 'e152_s2_no_ref')
SEED = 20261118
LABELS = ['L0 初始布局'] + [f'L{d} · S{(d-1)//4} 第{((d-1)%4)//2+1}次 · 子层{(d-1)%2+1}' for d in range(1,13)]
for d in (4,8,12):
    LABELS[d] = f'L{d} · S{d//4-1}完成'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,p)
def bbox(p):
    p=np.asarray(p,dtype=np.float64); return float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
def grouping(p, flags):
    polygons=[box(x,y,x+w,y+h) for x,y,w,h in p]; bad=[]
    for group in sorted(set(int(v) for v in flags[:,3] if v>0)):
        ids=np.flatnonzero(flags[:,3]==group).tolist()
        union=unary_union([polygons[i] for i in ids])
        components=len(union.geoms) if union.geom_type=='MultiPolygon' else 1
        if components>1: bad.append(dict(group=group,components=components,modules=ids))
    return sum(v['components']-1 for v in bad),bad
def describe(rows,key):
    metrics=[r[key] for r in rows]; ratios=np.array([m['bbox_area']/m['bbox_area_baseline'] for m in metrics])
    return dict(count=len(rows),hard_legal=sum(m['is_feasible'] for m in metrics),
        soft_satisfied_cases=sum(m['total_soft_violations']==0 for m in metrics),
        violations={k:dict(total=sum(m[k] for m in metrics),cases=sum(m[k]>0 for m in metrics),max=max(m[k] for m in metrics))
                    for k in ('grouping_violations','boundary_violations','total_soft_violations','fixed_violations','preplaced_violations')},
        soft_violation_rate_micro=sum(m['total_soft_violations'] for m in metrics)/sum(m['max_possible_violations'] for m in metrics),
        mean_per_case_soft_violation_rate=float(np.mean([m['violations_relative'] for m in metrics])),
        bbox_reference_ratio=dict(mean=float(ratios.mean()),p95=float(np.quantile(ratios,.95)),maximum=float(ratios.max()),
             counts_strictly_above={str(t):int((ratios>t).sum()) for t in (1.,1.05,1.1,1.2)},
             above1p2_indices=np.flatnonzero(ratios>1.2).tolist(),
             top10_indices=sorted(range(500),key=lambda i:(-ratios[i],i))[:10]))
def transition(a,b,areas,flags):
    """Conservative geometric diagnostics, not a proof of topological equivalence."""
    a,b=np.asarray(a),np.asarray(b); n=len(a); upper=np.triu(np.ones((n,n),bool),1)
    tolerance=.01*np.minimum(np.sqrt(areas)[:,None],np.sqrt(areas)[None,:])
    def orders(p):
        lo=p[:,:2];hi=lo+p[:,2:]
        return np.stack([np.where(hi[:,None,k]+tolerance<lo[None,:,k],-1,
                         np.where(lo[:,None,k]>hi[None,:,k]+tolerance,1,0)) for k in (0,1)],-1)
    oa,ob=orders(a),orders(b); flipped=(oa*ob==-1)&upper[...,None]
    pairs=np.argwhere(flipped.any(-1))
    def contacts(p):
        lo=p[:,:2];hi=lo+p[:,2:]
        signed=np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:])
        close=np.zeros((n,n),bool)
        for axis in (0,1):
            gap=np.minimum(np.abs(hi[:,None,axis]-lo[None,:,axis]),np.abs(lo[:,None,axis]-hi[None,:,axis]))
            close|=(gap<=tolerance)&(signed[...,1-axis]>tolerance)
        return close & upper & (flags[:,3,None]==flags[None,:,3]) & (flags[:,3,None]>0)
    ca,cb=contacts(a),contacts(b)
    return dict(complete_side_swap_pairs=len(pairs),axis_reversals=int(flipped.sum()),
        swap_pairs=[dict(modules=[int(i),int(j)],axes=[axis for k,axis in enumerate(('x','y')) if flipped[i,j,k]],same_group=bool(flags[i,3]>0 and flags[i,3]==flags[j,3])) for i,j in pairs],
        near_group_contact_edges_before=int(ca.sum()),near_group_contact_edges_after=int(cb.sum()),
        near_group_contact_edges_added=np.argwhere(cb & ~ca).tolist(),near_group_contact_edges_removed=np.argwhere(ca & ~cb).tolist())

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    initial_sources=sources(); dataset=RandomReference('dev_monitor500')
    members=[dataset.records[i]['member'] for i in dataset.indices]
    models={}; extremes=set(); provenance={}
    for arm in ARMS:
        run=RUNS/f'{arm}_seed0'; folder=run/'evaluation_no_mib_dev500'; frozen=read(run/'frozen.json')
        checkpoint=ROOT/frozen['checkpoint']; digest=sha(checkpoint); result=read(folder/'results.json')
        assert digest==frozen['checkpoint_sha256']==result['checkpoint_sha256']
        assert result['status']=='complete' and result['count']==500
        config=read(run/'config.json')['identity']; assert config['revision']=='E152.s2_local_repair.v2.seed0'
        assert config['sources']==initial_sources
        progress=read(run/'progress.json'); best=next(h for h in progress['history'] if h['epoch']==frozen['best_epoch'])
        losses=best['dev_monitor']['per_case_total']; assert len(losses)==500
        assert read(run/'panels.json')['dev_monitor_members']==members
        rows=[read(folder/'cases'/f'{i:04d}.json') for i in range(500)]
        assert all(r['index']==i and r['member']==members[i] and r['checkpoint_sha256']==digest for i,r in enumerate(rows))
        worst_loss=sorted(range(500),key=lambda i:(-losses[i],i))[:5]
        worst_q=sorted(range(500),key=lambda i:(-rows[i]['metrics']['cost'],i))[:5]
        extremes.update(worst_loss+worst_q)
        models[arm]=dict(run=run,rows=rows,frozen=frozen,checkpoint=checkpoint,digest=digest,losses=losses,
                          worst_loss=worst_loss,worst_q=worst_q,result=result)
        provenance[arm]=dict(checkpoint_sha256=digest,best_epoch=frozen['best_epoch'],
            result_sha256=sha(folder/'results.json'),progress_sha256=sha(run/'progress.json'),
            case_sha256={f'{i:04d}':sha(folder/'cases'/f'{i:04d}.json') for i in range(500)})
    ordinary=random.Random(SEED).sample([i for i in range(500) if i not in extremes],20)
    selections={arm:dict(ordinary=ordinary,worst_loss=m['worst_loss'],worst_Q=m['worst_q']) for arm,m in models.items()}
    save(HERE/'selection.json',dict(seed=SEED,models=selections,normal_rule='Same random20 for both models; exclude union of both own worst5 Loss and worst5 Q sets',
        loss_rule='Frozen own-Loss-best saved dev500 per_case_total descending; index tie-break; A/B Loss definitions differ',
        Q_rule='Existing final postprocessed per-case Q descending; index tie-break',duplicates_preserved_between_extreme_categories=True))

    all_cases=[]; reference_areas=[]
    for i in range(500):
        item=dataset[i]; assert item['member']==members[i]
        ref=bbox(item['supervision']['reference_xywh']);reference_areas.append(ref)
        for arm,m in models.items():
            row=m['rows'][i];raw,final=row['raw_metrics'],row['metrics']
            for metrics,key in ((raw,'draft_positions'),(final,'positions')):
                assert abs(metrics['bbox_area_baseline']-ref)<=max(1e-6,ref*1e-7)
                assert abs(metrics['bbox_area']-bbox(row[key]))<=max(1e-6,metrics['bbox_area']*1e-10)
                assert metrics['total_soft_violations']==metrics['grouping_violations']+metrics['boundary_violations']
                assert metrics['mib_violations']==0
            all_cases.append(dict(arm=arm,index=i,member=members[i],n=row['n'],own_loss=m['losses'][i],
                Q=final['cost'],draft_Q_is_hard_penalized=raw['cost'],reference_bbox_area=ref,
                draft_bbox_ratio=raw['bbox_area']/ref,final_bbox_ratio=final['bbox_area']/ref,
                post_over_draft_bbox=final['bbox_area']/raw['bbox_area'],
                draft_grouping=raw['grouping_violations'],final_grouping=final['grouping_violations'],
                draft_boundary=raw['boundary_violations'],final_boundary=final['boundary_violations'],
                draft_soft_total=raw['total_soft_violations'],final_soft_total=final['total_soft_violations'],
                max_possible_soft=final['max_possible_violations'],hard_legal=final['is_feasible'],
                fallback=row['fallback'],complete_ms=row['complete_seconds']*1000))
        if (i+1)%100==0:
            save(HERE/'state.json',dict(status='statistics',completed=i+1,total=500));print(f'Saved-metric checks {i+1}/500',flush=True)
    summary={arm:dict(best_epoch=m['frozen']['best_epoch'],mixed_Q=m['result']['mixed_Q'],draft=describe(m['rows'],'raw_metrics'),
                           final=describe(m['rows'],'metrics')) for arm,m in models.items()}
    fa=[r['metrics']['total_soft_violations']>0 for r in models[ARMS[0]]['rows']]
    fb=[r['metrics']['total_soft_violations']>0 for r in models[ARMS[1]]['rows']]
    paired=dict(both_satisfy=sum(not a and not b for a,b in zip(fa,fb)),A_satisfies_B_violates=sum(not a and b for a,b in zip(fa,fb)),
                B_satisfies_A_violates=sum(a and not b for a,b in zip(fa,fb)),both_violate=sum(a and b for a,b in zip(fa,fb)))
    save(HERE/'statistics.json',dict(status='complete',scope='Original fixed dev500; E152 v2 frozen own-Loss best',models=summary,paired=paired,
        definitions=dict(soft='Grouping plus boundary only; original areas, no MIB; F/P remain hard constraints',
            grouping='Sum of polygon union connected components minus one per group; draft overlaps may hide disconnection',
            boundary='One violation per constrained block if any required side/corner fails; not one per edge',
            abnormal_area='Strictly final tight bbox area / actual reference tight bbox area >1.2; saved baseline verified against actual reference for all500'),
        source_results_reused=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
    save(HERE/'cases.json',dict(rows=all_cases))
    save(HERE/'provenance.json',dict(models=provenance,collector_sha256=sha(Path(__file__)),frozen_sources=initial_sources))
    report=['# E152 v2：软约束、外框面积与逐阶段展示','',
        '原固定 dev500。A 保留 reference（best10），B 去掉 reference（best8）。复用已完成的原后处理与 Q，不训练、不重跑后处理、不读 test。','',
        '| 指标 | A 保留 reference | B 无 reference |','|---|---:|---:|']
    for stage,label in (('draft','L12草稿'),('final','后处理')):
        for key,text in (('grouping_violations','分组断连'),('boundary_violations','贴边违规'),('total_soft_violations','软违规合计')):
            v=[summary[a][stage]['violations'][key] for a in ARMS]
            report.append(f'| {label}：{text}（次数 / 涉及题数） | '+ ' | '.join(f'{r["total"]} / {r["cases"]}' for r in v)+' |')
        for key,text in (('soft_satisfied_cases','软约束全部满足题数'),('hard_legal','硬合法题数')):
            report.append(f'| {label}：{text} | '+' | '.join(str(summary[a][stage][key]) for a in ARMS)+' |')
        for threshold in ('1.05','1.1','1.2'):
            report.append(f'| {label}：外框/reference > {threshold} 题数 | '+' | '.join(str(summary[a][stage]['bbox_reference_ratio']['counts_strictly_above'][threshold]) for a in ARMS)+' |')
        report.append(f'| {label}：最大外框/reference | '+' | '.join(f'{summary[a][stage]["bbox_reference_ratio"]["maximum"]:.6f}' for a in ARMS)+' |')
    report+=['| 最终混合Q（越低越好） | '+' | '.join(f'{summary[a]["mixed_Q"]:.6f}' for a in ARMS)+' |','',
        '软约束只统计分组与贴边。分组数为各组不连通分量数减一的和；贴边按违规模块计数，角点任意要求未满足仅计一次。F/P 是硬约束，不能混入软违规。草稿的重叠可能让分组并集看似连通，因此草稿分组数不能单独判断拓扑好坏。','',
        '外框采用所有模块的紧包围盒（max(x+w)−min(x))×(max(y+h)−min(y))，分母为该题实际 reference 的同一定义面积；已逐题核对与保存的评分分母一致。严格 >1.2 才计异常。两组均无该类异常，因此本轮无法用此阈值定位后处理失败，也不能把较小的面积变化直接归因于后处理器。','',
        '后处理后两组都满足软约束：'+str(paired['both_satisfy'])+'题；仅A满足：'+str(paired['A_satisfies_B_violates'])+'题；仅B满足：'+str(paired['B_satisfies_A_violates'])+'题；两组都有违规：'+str(paired['both_violate'])+'题。','',
        '## 面积比例最大的题（便于后处理排查）','',
        '| 模型 | dev索引 | 草稿/reference | 后处理/reference | 分组违规 | 贴边违规 | 最终Q |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for i in summary[arm]['final']['bbox_reference_ratio']['top10_indices']:
            row=next(r for r in all_cases if r['arm']==arm and r['index']==i)
            report.append(f'| {"A" if arm==ARMS[0] else "B"} | {i:04d} | {row["draft_bbox_ratio"]:.6f} | {row["final_bbox_ratio"]:.6f} | {row["final_grouping"]} | {row["final_boundary"]} | {row["Q"]:.6f} |')
    report+=['','## 展示选取与阶段含义','',
        '两组共用普通20例（seed20261118，排除两组极端例并集），再各加本组Loss最差5例和最终Q最差5例。极端类别若重合仍分别保留，因此各30个展示条目。Loss按各组最佳点保存的自身训练目标逐题排序；A/B目标不同，不能直接比较这两个Loss总值。','',
        'L0 是初始布局；功能单元 S0 调用两次覆盖 L1–L4，S1 覆盖 L5–L8，S2 覆盖 L9–L12。重点比较 L0→L4 与 L4→L8；滑块仍保留全部13个真实读出。E152只训练S2，故两组L0–L8应逐位相同，不能将S0/S1已有的改动归因于本轮微调。','',
        '可视化数据、选择与逐阶段几何诊断将在同目录生成。完整500题统计见 statistics.json；逐题数值见 cases.json。','']
    (HERE/'report.md').write_text('\n'.join(report),encoding='utf-8')
    print('Statistics complete: '+json.dumps(summary,ensure_ascii=False),flush=True)

    captured_models={};topology={}
    for arm,m in models.items():
        folder=m['run']/'visualization_normal20_loss5_q5'
        assert not (folder/'data.json').exists(), 'Preserve an already completed gallery'
        entries=[dict(category=name,rank=j+1,index=i) for name,ids in [('普通20例',ordinary),('Loss最差5例',m['worst_loss']),('Q最差5例',m['worst_q'])] for j,i in enumerate(ids)]
        selected=sorted({v['index'] for v in entries})
        save(folder/'selection.json',dict(**selections[arm],entries=entries,checkpoint_sha256=m['digest'],seed=SEED))
        rows=load_rows('dev_monitor500',selected);lookup={i:j for j,i in enumerate(selected)}
        pack=ResidentRows(rows,'cpu',max_nodes=120);captured={}
        with torch.inference_mode():
            predictor=Predictor(m['checkpoint']);before={k:v.detach().cpu().clone() for k,v in predictor.model.state_dict().items()}
            for j,i in enumerate(selected):
                item=dataset[i];row=m['rows'][i];n=row['n'];flags=np.asarray(item['solver_inputs'][5],dtype=int);areas=np.asarray(item['solver_inputs'][1])
                g={key:value.cuda() for key,value in public_graph(item['solver_inputs']).items()}
                states=predictor.model(g)
                assert states.shape==(13,1,120,3)
                layers=(decode(states,g)[:,0,:n]*g['scale'][0]).cpu().numpy().astype(float)
                np.testing.assert_array_equal(layers[-1],np.asarray(row['draft_positions']))
                assert hashlib.sha256(layers[8].tobytes()).hexdigest()==row['L8_sha256']
                batch=pack.batch([lookup[i]]);cpu_states=states.cpu();losses=[]
                for depth,state in enumerate(cpu_states):
                    v={key:float(value[0]) for key,value in base_per_case(state,batch).items()}
                    v.update(directly_supervised=depth==12,supervision_weight=1. if depth==12 else 0.)
                    if depth==12:
                        center,ratio=drift_terms(cpu_states[8],state,batch)
                        v['drift_weighted']=.2*float((center+.25*ratio)[0])
                        v['objective_total']=v['total']-(v['reference_weighted'] if arm==ARMS[1] else 0)+v['drift_weighted']
                        v['objective_name']='公开任务 + 0.2改动'+(' + reference' if arm==ARMS[0] else '')
                    else:v['objective_name']='共同完整Loss诊断（冻结）'
                    count,_=grouping(layers[depth],flags);v['grouping_violations']=count
                    losses.append(v)
                transitions={f'S{k}':transition(layers[4*k],layers[4*k+4],areas,flags) for k in range(3)}
                ref_count,_=grouping(item['supervision']['reference_xywh'],flags);final_count,bad=grouping(row['positions'],flags)
                assert final_count==row['metrics']['grouping_violations']
                note='；'.join(f'{key}严格完整换侧 {value["complete_side_swap_pairs"]} 对，同组近接触边新增 {len(value["near_group_contact_edges_added"])} / 丢失 {len(value["near_group_contact_edges_removed"])}' for key,value in transitions.items())
                captured[i]=dict(dev_index=i,member=row['member'],n=n,constraints=flags.tolist(),reference=item['supervision']['reference_xywh'].tolist(),
                    layers=layers.tolist(),layer_ids=list(range(13)),layer_labels=LABELS,layer_losses=losses,final=row['positions'],
                    topology_changes=transitions,grouping_summary=dict(reference=ref_count,final=final_count,final_bad_groups=bad),
                    objective_note='L0→L4=S0；L4→L8=S1；L8→L12=S2。'+note+'。严格换侧仅是保守几何证据，0次不能证明拓扑未变；近接触容差为模块局部尺度的1%。L12目标含0.2改动惩罚，分项表完整Loss为共同诊断。',
                    metrics=dict(Q=row['metrics']['cost'],hard_legal=bool(row['metrics']['is_feasible']),complete_ms=row['complete_seconds']*1000),
                    area_ratios=dict(draft=row['raw_metrics']['bbox_area']/reference_areas[i],final=row['metrics']['bbox_area']/reference_areas[i]))
                normalize(captured[i])
                save(folder/'state.json',dict(status='capturing',completed=j+1,unique_count=len(selected),display_count=30))
                print(f'{arm} captured {j+1}/{len(selected)} dev{i:04d}',flush=True)
            assert all(torch.equal(v.cpu(),before[k]) for k,v in predictor.model.state_dict().items())
        cases=[]
        for entry in entries:
            i=entry['index'];case=copy.deepcopy(captured[i]);case.update(group=entry['category'],
                label=f'{entry["category"]} #{entry["rank"]} · dev{i:04d} · own Loss {m["losses"][i]:.6f} · Q {m["rows"][i]["metrics"]["cost"]:.6f}')
            cases.append(case)
        assert len(cases)==30 and sha(m['checkpoint'])==m['digest']
        save(folder/'data.json',dict(title=f'E152 {"A 保留reference" if arm==ARMS[0] else "B 无reference"} · best{m["frozen"]["best_epoch"]}',
            note='普通20例 + Loss最差5例 + Q最差5例。L0初始布局；S0完成=L4；S1完成=L8；S2完成=L12。两组L0–L8被冻结且相同。左右Reference/原后处理固定，滑块切换全部真实层。',
            identity=dict(arm=arm,checkpoint_sha256=m['digest'],best_epoch=m['frozen']['best_epoch'],actual_layer_ids=list(range(13)),
                terminal_drafts_bitwise_equal=True,L8_hashes_exact=True,postprocessing_reused=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False),cases=cases))
        for tag,i in (('normal',ordinary[0]),('worst_Q',m['worst_q'][0])):
            c=normalize(captured[i]);variants=dict(reference=dict(positions=c['reference']),final=dict(positions=c['final']))
            stages=[('reference','Reference')]
            for depth in (0,4,8,12):
                name=f'L{depth}';variants[name]=dict(positions=c['layers'][depth]);stages.append((name,LABELS[depth]))
            stages.append(('final','Postprocessed'))
            plot_case(dict(title=f'{arm} · dev{i:04d} · stage comparison',constraints=c['constraints'],axis_limits=c['bounds'],variants=variants),
                      folder/f'preview_{tag}.png',stages=stages)
        save(folder/'state.json',dict(status='complete',display_count=30,unique_count=len(selected),category_counts=dict(ordinary=20,worst_loss=5,worst_Q=5),
            actual_layers=13,model_unchanged=True,checkpoint_unchanged=True,terminal_drafts_bitwise_equal=True,L8_hashes_exact=True,
            training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
        captured_models[arm]=captured
        topology[arm]={category:{f'S{k}':dict(cases=len(ids),cases_with_complete_side_swap=sum(captured[i]['topology_changes'][f'S{k}']['complete_side_swap_pairs']>0 for i in ids),
            complete_side_swap_pairs=sum(captured[i]['topology_changes'][f'S{k}']['complete_side_swap_pairs'] for i in ids),
            group_near_edges_added=sum(len(captured[i]['topology_changes'][f'S{k}']['near_group_contact_edges_added']) for i in ids),
            group_near_edges_removed=sum(len(captured[i]['topology_changes'][f'S{k}']['near_group_contact_edges_removed']) for i in ids)) for k in range(3)}
            for category,ids in [('ordinary20',ordinary),('all_selected_unique',selected)]}
        del predictor,before,pack,rows,states,cpu_states,batch;gc.collect();torch.cuda.empty_cache()
    common=set(captured_models[ARMS[0]])&set(captured_models[ARMS[1]])
    for i in common:
        np.testing.assert_array_equal(np.asarray(captured_models[ARMS[0]][i]['layers'][:9]),np.asarray(captured_models[ARMS[1]][i]['layers'][:9]))
    assert sources()==initial_sources
    save(HERE/'topology_summary.json',dict(models=topology,matched_cases_checked=len(common),L0_through_L8_bitwise_equal=True,
        warning='Selected cases only; strict separated-pair reversal and near-contact diagnostics are not an exhaustive topology test',
        complete_swap_definition='Pair completely separated along x or y at input and output, order reversed; separation margin > 1% of min module sqrt(public area) at both endpoints',
        near_contact_definition='Same-group edge distance <=1% min sqrt(public area), orthogonal overlap > same tolerance; excludes corner-only contact'))
    with (HERE/'report.md').open('a',encoding='utf-8') as out:
        out.write('\n两组共有 '+str(len(common))+' 个展示成员已核验 L0–L8 逐位一致。所有末层均与原评价草稿逐位一致；后处理坐标与Q直接复用。\n\n')
        out.write('普通20例的保守换侧诊断（同一对模块在阶段输入/输出均完全分离且x或y先后关系反转，含1%局部尺度余量）：\n\n| 阶段 | A发生换侧题数 / 对数 | B发生换侧题数 / 对数 |\n|---|---:|---:|\n')
        for k in range(3):
            out.write(f'| S{k} (L{4*k}→L{4*k+4}) | '+' | '.join(f'{topology[a]["ordinary20"][f"S{k}"]["cases_with_complete_side_swap"]} / {topology[a]["ordinary20"][f"S{k}"]["complete_side_swap_pairs"]}' for a in ARMS)+' |\n')
        out.write('\n换侧只是一种明确几何变化，0不能证明拓扑未改变；同组近接触边新增/丢失列表另见每例 topology_changes 和 topology_summary.json。上述选例不能代表全体dev500。\n')
    save(HERE/'state.json',dict(status='complete',models=list(ARMS),statistics_cases_per_model=500,gallery_entries_per_model=30,
        matched_prefix_cases=len(common),L0_through_L8_bitwise_equal=True,frozen_sources_unchanged=True,training=False,postprocessing_calls=0,Q_calls=0,test_read=False))
    print('COMPLETE: statistics, two30-entry real13-layer galleries and prefix identity checks',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(HERE/'state.json',dict(status='error',error=repr(error),traceback=traceback.format_exc()));raise
