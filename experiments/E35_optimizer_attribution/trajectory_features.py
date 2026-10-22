"""Offline measurements of saved E35 trajectories. No model or optimizer calls."""
from .resources import configure
RESOURCE = configure()
from pathlib import Path
from collections import Counter
import json
import sqlite3
import zlib
import time
import subprocess
import numpy as np
from scipy.stats import rankdata
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from .cases import prepare_cache
from .io import HERE, ROOT, save, sha
from .backend.frozen.packing import prepare_problem
from .backend.frozen.feasible import Objective
from .backend.frozen.solver import hard_valid
from .fast_geometry import topology, contact_tree, reduce_edges
from scoring.evaluation import evaluate_case

OUT = HERE / 'runs/20261022_trajectory_features'
SOURCE = HERE / 'runs/20261022_attribution100'


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a): return None
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                    np.percentile(a, [0, 25, 50, 75, 95, 100]).tolist()), mean=float(a.mean()))


def centers(x): return x[:, :2] + x[:, 2:] / 2


def move(a, b, units):
    d = centers(b) - centers(a)
    distances = np.linalg.norm(d, axis=1) / units
    shape = np.abs(np.log((b[:, 2] / b[:, 3]) / (a[:, 2] / a[:, 3])))
    # Removing one common translation is a diagnostic, never a legal output change.
    centered = np.linalg.norm(d - np.median(d, axis=0), axis=1) / units
    return {'center_mean': float(distances.mean()), 'center_p95': float(np.percentile(distances, 95)),
            'center_max': float(distances.max()), 'center_detranslated_mean': float(centered.mean()),
            'moved_gt_0_1_fraction': float((distances > .1).mean()),
            'moved_gt_0_5_fraction': float((distances > .5).mean()),
            'moved_gt_1_fraction': float((distances > 1).mean()),
            'log_AR_mean': float(shape.mean()), 'log_AR_p95': float(np.percentile(shape, 95)),
            'AR_factor_ge_2_fraction': float((shape >= np.log(2)).mean())}


def overlap_features(x, units):
    n = len(x); i, j = np.triu_indices(n, 1)
    extent = np.minimum(x[i, :2]+x[i, 2:], x[j, :2]+x[j, 2:]) - np.maximum(x[i, :2], x[j, :2])
    bad = (extent > 1e-6).all(1); pairs = np.column_stack((i[bad], j[bad]))
    degree = np.bincount(pairs.ravel(), minlength=n)
    parent = np.arange(n)
    def root(k):
        while parent[k] != k: k = parent[k]
        return k
    for a, b in pairs: parent[root(a)] = root(b)
    sizes = Counter(root(k) for k in range(n))
    depth = np.min(extent[bad], axis=1)/np.minimum(units[i[bad]], units[j[bad]])
    return {'overlap_pairs': int(bad.sum()), 'overlap_pair_density': float(bad.mean()),
            'overlap_pair_area_over_total': float(np.maximum(extent, 0).prod(1).sum()/np.sum(units**2)),
            'overlap_largest_component_fraction': max(sizes.values())/n,
            'overlap_max_degree_fraction': float(degree.max()/max(1, n-1)),
            'overlap_penetration_p95': float(np.percentile(depth, 95)) if len(depth) else 0.,
            'overlap_participating_fraction': float((degree>0).mean())}


def input_features(x, p, units):
    c = p.constraints; n = p.n; lo=x[:, :2].min(0); hi=(x[:, :2]+x[:, 2:]).max(0)
    distances = np.column_stack((x[:, 0]-lo[0], hi[0]-x[:, 0]-x[:, 2], hi[1]-x[:, 1]-x[:, 3], x[:, 1]-lo[1]))
    mask = (c[:, 4, None] & np.array([1, 2, 4, 8])) != 0
    requested = mask.any(1)
    gap = np.where(mask, np.maximum(0, distances)/units[:, None], 0).max(1)
    group_ids = [np.flatnonzero(c[:, 3]==g) for g in np.unique(c[:, 3]) if g>0]
    mst_gap = 0.; max_group = 0
    for ids in group_ids:
        max_group = max(max_group, len(ids))
        r=x[ids]; d=np.maximum(0, np.maximum(r[:, None, :2]-r[None, :, :2]-r[None, :, 2:], r[None, :, :2]-r[:, None, :2]-r[:, None, 2:]))
        # Rectangular set-distance lower bound. Intersections have zero gap;
        # this does not certify legal shared-edge contact in an illegal draft.
        w=np.linalg.norm(d, axis=2)/np.minimum(units[ids, None], units[None, ids])
        if len(ids)>1:
            w = w+1e-12; np.fill_diagonal(w, 0)
            mst_gap += float(minimum_spanning_tree(csr_matrix(w)).sum())-(len(ids)-1)*1e-12
    i,j=np.tril_indices(n,-1); delta=centers(x)[i]-centers(x)[j]; extent=(x[i,2:]+x[j,2:])/2
    clearance=(np.abs(delta)-extent)/extent
    ambiguous=np.abs(clearance[:,0]-clearance[:,1])<.1
    degree=p.adjacency.sum(1)+p.pin_weight
    total=max(float(degree.sum()),1e-12)
    tags=np.column_stack((c[:,2]>0,c[:,3]>0,c[:,4]>0)).sum(1)
    return {**overlap_features(x,units), 'n':n,
            'fixed_shape_fraction':float((p.fixed|p.preplaced).mean()), 'preplaced_fraction':float(p.preplaced.mean()),
            'area_cv':float(np.std(units**2)/np.mean(units**2)),
            'draft_log_AR_p95':float(np.percentile(np.abs(np.log(x[:,2]/x[:,3])),95)),
            'boundary_fraction':float(requested.mean()), 'corner_demand_fraction':float((mask.sum(1)>1).mean()),
            'boundary_gap_mean':float(gap[requested].mean()) if requested.any() else 0.,
            'boundary_gap_max':float(gap.max()), 'group_fraction':float((c[:,3]>0).mean()),
            'group_max_size_fraction':max_group/n, 'group_gap_MST_per_block':mst_gap/n,
            'mib_fraction':float((c[:,2]>0).mean()), 'multiple_soft_tags_fraction':float((tags>1).mean()),
            'topology_ambiguous_pair_fraction':float(ambiguous.mean()),
            'wire_pair_density':float(np.count_nonzero(np.triu(p.adjacency,1))/max(1,n*(n-1)/2)),
            'wire_max_degree_weight_share':float(degree.max()/total)}


def guard_reasons(x, p):
    locked=p.fixed|p.preplaced;free=~locked; reasons=[]
    if (x[:, :2]<0).any(): reasons.append('negative_coordinate')
    if (np.abs(x[locked,2:]-p.specs[locked,2:])>1e-8).any(): reasons.append('locked_shape')
    if (np.abs(x[p.preplaced]-p.specs[p.preplaced])>1e-8).any(): reasons.append('preplaced_location')
    if (np.abs(x[free,2:].prod(1)-p.areas[free])/p.areas[free]>.01).any(): reasons.append('area_tolerance')
    if overlap_features(x,np.sqrt(p.areas))['overlap_pairs']: reasons.append('overlap')
    assert bool(reasons) == (not hard_valid(x,p))
    return reasons


def conflicts(x, p, units, full, reduced):
    i,j=np.triu_indices(p.n,1)
    o=np.minimum(x[i,:2]+x[i,2:],x[j,:2]+x[j,2:])-np.maximum(x[i,:2],x[j,:2])
    bad=(o>1e-6).all(1);depth=np.minimum(o[:,0],o[:,1]);norm=depth/np.minimum(units[i],units[j])
    pairs=[[int(a),int(b)] for a,b in zip(i[bad],j[bad])]
    result={'pairs':pairs,'max_penetration_absolute':float(depth[bad].max()) if bad.any() else 0.,
            'max_penetration_block_units':float(norm[bad].max()) if bad.any() else 0.,
            'overlap_pairs_touching_preplaced':int((bad&(p.preplaced[i]|p.preplaced[j])).sum()),
            'overlap_pairs_touching_locked_shape':int((bad&((p.fixed|p.preplaced)[i]|(p.fixed|p.preplaced)[j])).sum())}
    for name,edges in [('full',full),('reduced',reduced)]:
        edges=np.asarray(edges,dtype=int).reshape(-1,3);a,b,axis=edges.T
        # Geometrically required elastic slack lower bound, not saved primal slack.
        slack=np.maximum(0,x[a,axis]+x[a,axis+2]-x[b,axis])
        result[name+'_precedence_violated_edges']=int((slack>1e-6).sum())
        result[name+'_precedence_max_required_slack_block_units']=float((slack/np.minimum(units[a],units[b])).max()) if len(edges) else 0.
    return result


def partial_rho(x, y, ns):
    # Rank residualization controls continuous module count plus prescribed
    # budget tiers; descriptive association only, no significance/causal claim.
    ns=np.asarray(ns); control=np.column_stack((np.ones(len(ns)),rankdata(ns),ns>=60,ns>=90,ns>=110))
    def residual(a):
        a=rankdata(a);return a-control@np.linalg.lstsq(control,a,rcond=None)[0]
    a,b=residual(x),residual(y)
    return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-10 else None


def main():
    started=time.perf_counter(); OUT.mkdir(exist_ok=True)
    design='''# 保存轨迹的调整幅度与难例诊断

本轮只读取E35既有100例full臂，离线重建几何、关系和计分；不调用模型/优化器，不增加求解，不改变E25，不读取测试。单数值进程、两CPU执行槽。源数据库只读，新结果独立保存。

事前问题：大改动是否集中于首轮？求解调用次数是否被模块数和预算混淆？重叠图连通性、穿透深度、边界距离、组间隙、固定规格及多类约束耦合能否解释剩余难度？多次调整是否伴随关系翻转、往返和收益饱和？先分别测量几何负担、数值难度、拓扑选择、预算停止及最终残差，不预设某一指标能解释全部。

中心距离除以每块公开有效面积的平方根；统计包含全部块，并另给出去除整体平移的诊断。形状变化用abs(log(新宽高比/旧宽高比))，达到log(2)表示宽高比变化至少2倍。形状函数原始输入→输出、标准化后的当前搜索状态、最佳解的改变分别计算。主流程和递归救援分别计数，累计幅度不能解释为最终距离。行布局是独立保底，不拿它到首次最佳解的巨幅变化冒充draft修复幅度。

输入特征与全程反馈特征分开；重叠面积为成对交叠面积之和而非并集。组MST间隙是矩形集合距离下界，draft的重叠也会使其为零，不表示分组已合法满足。关系使用实际主流程配置重建，相邻主流程shape输入比较，避免将末端投影或递归重启混入。相关系数只作本100例探索性描述，另控制模块数及分段预算；不能证明原因或普遍预测能力。小调用次数可能是单轮耗时大或预算耗尽；小位移可能跨越精确接触边界，二者均不能直接解释成容易/无效。

尚缺且不能事后恢复：锥规划逐行松弛/对偶值、原始残差和条件数、矩阵分解详细计时、反事实拓扑可行性。不得从保存布局伪造这些量。
'''
    (OUT/'design.md').write_text(design,encoding='utf-8')
    source=SOURCE/'outcomes.sqlite3';con=sqlite3.connect('file:'+source.resolve().as_posix()+'?mode=ro',uri=True)
    records=[json.loads(zlib.decompress(r[0])) for r in con.execute("SELECT payload FROM records WHERE arm='full' ORDER BY idx")];con.close()
    assert len(records)==100
    cases=prepare_cache()['rows'];scores=json.loads((HERE/'runs/20261022_attribution_report/layout_scores_local.json').read_text())
    case_rows=[];step_rows=[];modules={};statuses=Counter();failures=Counter();all_stats=[];selections={}
    for ordinal,r in enumerate(records):
        idx=r['index'];case=cases[idx]['case'];p=prepare_problem(*case.solver_inputs);obj=Objective(p)
        units=np.sqrt(p.areas.copy());locked=p.fixed|p.preplaced;units[locked]=np.sqrt(p.specs[locked,2:].prod(1))
        layouts={k:np.asarray(v,dtype=float) for k,v in r['layouts'].items()};events=r['events']
        initial=next(e for e in events if e['kind']=='initial_state' and e['depth']==1)
        draft=layouts[initial['draft']];seed=layouts[initial['state']];final=np.asarray(r['positions']);scorecache=scores[str(idx)]
        def score(key):
            if key not in scorecache:scorecache[key]=evaluate_case(case,layouts[key],0.)['metrics']
            return scorecache[key]
        f=input_features(draft,p,units)
        shapes=[e for e in events if e['kind']=='call' and e['module']=='shape_solve']
        root=[e for e in shapes if e['depth']==1]
        config=next(e for e in events if e['kind']=='configuration' and e['depth']==1)
        # Non-shape transitions (if any) are explicitly bridged in path measurement.
        current=seed;path=0.;log_path=0.;bridges=0.;previous_topology=None;previous_tree=None;first=None;local=[];pair_counts=Counter()
        for e in shapes:
            stats=e['stats'];statuses[stats['status']]+=1;all_stats.append(stats)
            if e['depth']!=1: continue
            following=[v for v in events[e['id']+1:] if v.get('parent')==e['parent']]
            acceptance=next(v for v in following if v['kind']=='acceptance' and v['module']=='shape')
            guards=[v for v in following if v['kind']=='call' and v['module']=='hard_guard' and v['id']<acceptance['id']]
            a=layouts[e['input']]
            raw=layouts.get(e.get('output'))
            row={'index':idx,'n':p.n,'step':len(local)+1,'event_id':e['id'],'status':stats['status'],
                 'legal':acceptance['hard_legal'],'updated_best':acceptance['updated'],
                 'proxy_gain':acceptance['before_proxy']-acceptance['after_proxy'],
                 **{k:stats[k] for k in ('iterations','variables','constraints','native_seconds','assembly_seconds','setup_seconds')},
                 'wall_seconds':e['seconds_excluding_observer']}
            full=topology(a,p.constraints,mode=config.get('mode',0),array=True)
            reduced=reduce_edges(full,p.n)
            # Full arm has mode0/group_order=False and no overrides (frozen defaults).
            tree=set(contact_tree(a,p,full))
            row['topology_flip_fraction']=float(np.any(full!=previous_topology,axis=1).mean()) if previous_topology is not None else None
            row['contact_tree_replacement_fraction']=len(tree^previous_tree)/max(1,len(tree|previous_tree)) if previous_tree is not None else None
            previous_topology=full;previous_tree=tree
            prior_reduce=[v for v in events[:e['id']] if v['kind']=='call' and v['module']=='edge_reduction' and v['parent']==e['parent']][-1]
            row['reduced_edges']=prior_reduce['output_count']
            assert len(reduced)==row['reduced_edges']
            if raw is not None:
                row['raw_move']=move(a,raw,units)
                # Each returned proposal is made the current state after cleanup,
                # even if it was rejected as an incumbent.
                state=layouts[acceptance['clean']] if acceptance['clean'] else raw
                row['state_move']=move(a,state,units)
                row['state_conflicts']=conflicts(state,p,units,full,reduced)
                pair_counts.update(map(tuple,row['state_conflicts']['pairs']))
                row['state_log_AR_max']=float(np.abs(np.log(state[:,2]/state[:,3])).max())
                row['state_bbox_over_total_area']=float(np.prod((state[:,:2]+state[:,2:]).max(0)-state[:,:2].min(0))/sum(units**2))
                row['input_hard_legal']=hard_valid(a,p)
                row['input_proxy']=obj(a);row['state_proxy']=obj(state)
                bridges+=move(current,a,units)['center_mean'];path+=row['state_move']['center_mean'];log_path+=row['state_move']['log_AR_mean'];current=state
                ratio=raw[:,2:]/a[:,2:]
                row['dimension_step_bound_fraction']=float(((ratio<=.40001)|(ratio>=2.49999)).any(1)[~locked].mean()) if (~locked).any() else 0.
                relative_area=raw[:,2:].prod(1)/p.areas
                row['area_tolerance_edge_fraction']=float((np.abs(relative_area[~locked]-1)>.0099).mean()) if (~locked).any() else 0.
                rejected=[] if acceptance['hard_legal'] else guard_reasons(layouts[guards[-1]['input']],p)
                row['guard_reasons']=rejected;failures.update(rejected)
                row['guard_conflicts']=conflicts(layouts[guards[-1]['input']],p,units,full,reduced)
                if acceptance['hard_legal']:
                    row['candidate_Q']=score(acceptance['clean'])['cost']
                    if first is None:first=(row['step'],acceptance['clean'])
                if acceptance['updated'] and acceptance['before']!=initial['row']:
                    row['incumbent_Q_gain']=score(acceptance['before'])['cost']-score(acceptance['after'])['cost']
                else: row['incumbent_Q_gain']=None
            local.append(row);step_rows.append(row)
        net=move(seed,current,units);end=move(draft,final,units)
        calls=[e for e in events if e['kind']=='call']
        for e in calls:
            if e['module'] not in ('shape_solve','fixed_shape_projection','certified_polish','packed_seed'):continue
            m=modules.setdefault(e['module'],{'calls':0,'paired_calls':0,'per_call_moves':[],'per_case_cumulative_moves':{},'per_call_log_AR':[]})
            m['calls']+=1
            if e.get('input') and e.get('output'):
                d=move(layouts[e['input']],layouts[e['output']],units);m['paired_calls']+=1;m['per_call_moves'].append(d['center_mean']);m['per_call_log_AR'].append(d['log_AR_mean']);m['per_case_cumulative_moves'][idx]=m['per_case_cumulative_moves'].get(idx,0)+d['center_mean']
        changes=[s['topology_flip_fraction'] for s in local if s['topology_flip_fraction'] is not None]
        tree_changes=[s['contact_tree_replacement_fraction'] for s in local if s['contact_tree_replacement_fraction'] is not None]
        finite=[s for s in local if 'state_move' in s]
        after=[s for s in finite if first and s['step']>first[0]]
        total_time=sum(s['wall_seconds'] for s in local)
        no_update=sum(s['wall_seconds'] for s in local if not s['updated_best'])
        row={'index':idx,'member':r['member'],'n':p.n,'draft':f,'draft_to_final':end,
             'root_shape_calls':len(root),'all_shape_calls':len(shapes),'recursive_shape_calls':len(shapes)-len(root),
             'root_legal_shapes':sum(s['legal'] for s in local),'root_shape_best_updates':sum(s['updated_best'] for s in local),
             'root_shape_seconds':total_time,'all_shape_seconds':sum(e['seconds_excluding_observer'] for e in shapes),
             'all_native_iterations':sum(e['stats']['iterations'] for e in shapes),'root_native_iterations':sum(s['iterations'] for s in local),
             'mean_native_iterations_per_call':float(np.mean([s['iterations'] for s in local])),
             'root_first_move':finite[0]['state_move'] if finite else None,
             'root_shape_cumulative_center':path,'root_nonshape_bridge_center':bridges,
             'root_shape_cumulative_log_AR':log_path,'root_last_shape_net':net,
             'root_path_to_net_ratio':(path+bridges)/net['center_mean'] if net['center_mean']>1e-12 else None,
             'first_legal_shape_step':first[0] if first else None,
             'after_first_legal_calls':len(after),'after_first_legal_center_path':sum(s['state_move']['center_mean'] for s in after),
             'after_first_legal_Q_gain':score(first[1])['cost']-r['metrics']['cost'] if first else None,
             'root_no_best_update_time_fraction':no_update/max(total_time,1e-12),
             'root_topology_flip_fraction_mean':float(np.mean(changes)) if changes else 0.,
             'root_contact_tree_replacement_mean':float(np.mean(tree_changes)) if tree_changes else 0.,
             'root_dimension_bound_fraction_mean':float(np.mean([s['dimension_step_bound_fraction'] for s in finite])) if finite else None,
             'root_area_edge_fraction_mean':float(np.mean([s['area_tolerance_edge_fraction'] for s in finite])) if finite else None,
             'most_persistent_overlap_pairs':[{'pair':list(pair),'steps':count} for pair,count in pair_counts.most_common(5)],
             'root_max_penetration_block_units':max(s['state_conflicts']['max_penetration_block_units'] for s in finite) if finite else None,
             'max_variables':max(e['stats']['variables'] for e in shapes),'max_constraints':max(e['stats']['constraints'] for e in shapes),
             'rescue':any(e['kind']=='gate' and e['module']=='robust_seeds' and e['triggered'] for e in events),
             'late_extension':any(e['kind']=='gate' and e['module']=='late_extension' and e['triggered'] for e in events),
             'plateau':any(e['kind']=='gate' and e['module']=='plateau' and e['triggered'] for e in events),
             'budget_exit':any(e['kind']=='gate' and e['module']=='budget_exit' and e['triggered'] and e['depth']==1 for e in events),
             'metrics':r['metrics'],'seconds':r['seconds']}
        assert row['root_legal_shapes']<=row['root_shape_calls']<=row['all_shape_calls']
        assert row['root_path_to_net_ratio'] is None or row['root_path_to_net_ratio']>=1-1e-8
        case_rows.append(row);selections[idx]={'record':r,'initial':initial,'local':local,'first':first}
        if (ordinal+1)%25==0:print('offline cases',ordinal+1,'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={key:describe([r[key] for r in case_rows if r[key] is not None]) for key in (
        'root_shape_calls','all_shape_calls','all_native_iterations','mean_native_iterations_per_call','root_shape_cumulative_center',
        'root_path_to_net_ratio','first_legal_shape_step','root_no_best_update_time_fraction','root_topology_flip_fraction_mean',
        'root_contact_tree_replacement_mean','root_area_edge_fraction_mean','after_first_legal_Q_gain')}
    summary['draft_to_final']={k:describe([r['draft_to_final'][k] for r in case_rows]) for k in case_rows[0]['draft_to_final']}
    summary['first_shape']={k:describe([r['root_first_move'][k] for r in case_rows if r['root_first_move']]) for k in case_rows[0]['root_first_move']}
    for m in modules.values():
        m['per_call_mean_center']=describe(m.pop('per_call_moves'));m['per_call_mean_log_AR']=describe(m.pop('per_call_log_AR'));m['per_case_cumulative_center']=describe(list(m.pop('per_case_cumulative_moves').values()))
    targets={'calls':'all_shape_calls','path':'root_shape_cumulative_center','time':'all_shape_seconds','inner_iterations_per_call':'mean_native_iterations_per_call'}
    associations={};ns=[r['n'] for r in case_rows]
    for label,target in targets.items():
        y=[r[target] for r in case_rows];table=[]
        for feature in case_rows[0]['draft']:
            x=[r['draft'][feature] for r in case_rows]
            if np.std(x)<1e-12:continue
            table.append({'feature':feature,'spearman':float(np.corrcoef(rankdata(x),rankdata(y))[0,1]),'size_budget_adjusted_rho':partial_rho(x,y,ns) if feature!='n' else None})
        associations[label]=sorted(table,key=lambda t:abs(t['size_budget_adjusted_rho'] or 0),reverse=True)
    tiers=[]
    for lo,hi in [(0,59),(60,89),(90,109),(110,999)]:
        rr=[r for r in case_rows if lo<=r['n']<=hi]
        tiers.append({'n_range':[lo,hi],'count':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('all_shape_calls','all_shape_seconds','mean_native_iterations_per_call','root_shape_cumulative_center')},'rescue_cases':sum(r['rescue'] for r in rr)})
    total_time=sum(s['seconds'] for s in all_stats)
    timings={k:sum(s[k] for s in all_stats) for k in ('assembly_seconds','setup_seconds','native_seconds')}
    timings['function_seconds']=total_time
    rank_fields=('all_shape_calls','root_shape_cumulative_center','all_shape_seconds')
    ranks={k:[{v:r[v] for v in ('index','member','n',k,'rescue','late_extension','first_legal_shape_step')} for r in sorted(case_rows,key=lambda r:r[k],reverse=True)[:8]] for k in rank_fields}
    report={'scope':'offline saved adaptive dev100; observational, no new inference/solve/train/test',
            'summary':summary,'modules':modules,'status_counts_all_shapes':dict(statuses),'root_guard_reasons':dict(failures),
            'tiers':tiers,'input_associations':associations,'rankings':ranks,'timings_all_shapes':timings,
            'root_first_legal_missing':[r['index'] for r in case_rows if r['first_legal_shape_step'] is None],
            'rescue_cases':[r['index'] for r in case_rows if r['rescue']],
            'counts':{'cases':len(case_rows),'all_shapes':len(all_stats),'root_shapes':len(step_rows),'returned_shapes':sum(m['paired_calls'] for k,m in modules.items() if k=='shape_solve')},
            'source_database_sha256':sha(source),'analysis_source_sha256':sha(Path(__file__)),
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'resource':RESOURCE,'seconds':time.perf_counter()-started,'new_solves':0,'test_read':False}
    no_rescue=[r for r in case_rows if not r['rescue']]
    report['associations_without_rescue']={feature:{target:partial_rho([r['draft'][feature] for r in no_rescue],[r[target] for r in no_rescue],[r['n'] for r in no_rescue]) for target in ('all_shape_calls','root_shape_cumulative_center','mean_native_iterations_per_call')} for feature in ('overlap_penetration_p95','overlap_pair_area_over_total','preplaced_fraction')}
    report['remaining_final_soft']={k:{'cases_with_violation':sum(r['metrics'][k]>0 for r in case_rows),'sum':sum(r['metrics'][k] for r in case_rows)} for k in ('boundary_violations','grouping_violations','mib_violations')}
    rejected=[s for s in step_rows if 'state_conflicts' in s and not s['legal']]
    report['rejected_geometry']={'calls':len(rejected),'max_penetration_block_units':describe([s['guard_conflicts']['max_penetration_block_units'] for s in rejected]),
        'calls_max_depth_below_1e_5_absolute':sum(s['guard_conflicts']['max_penetration_absolute']<1e-5 for s in rejected),
        'calls_max_depth_above_1e_3_block_units':sum(s['guard_conflicts']['max_penetration_block_units']>1e-3 for s in rejected),
        'status_counts':dict(Counter(s['status'] for s in rejected))}
    save(OUT/'summary.json',report);save(OUT/'cases.json',case_rows);save(OUT/'steps.json',step_rows)
    write_report(report,case_rows,step_rows)
    plot(report,case_rows,step_rows,selections,cases)
    print(json.dumps({'counts':report['counts'],'summary':summary,'tiers':tiers,'ranks':ranks},ensure_ascii=False),flush=True)


def write_report(report,rows,steps):
    s=report['summary']
    lines=['# E25求解轨迹：调整幅度、求解负担与难例','','只分析已完成E35的同一100例观测轨迹；新增求解0，CPU总计两线程。Q为开发中性时间指标。完整指标定义及事前假设见[design](design.md)。','',
           '## 目前缺什么：证据与假设','',
           '更明确的缺口是持续冲突的识别和搜索路线切换，而不是统一增加形状迭代次数。index496的[9,38]重叠在主流程42次输出中持续存在；末轮仍穿透约0.707个块单位，其中38是预放置块。末端步幅已接近零，却仍未可行。index329的[35,60]重叠出现23次，其中60是预放置块；主流程27轮均不合法，末轮外包框/总面积从首轮1.16升到4.68，最极端长宽比约exp(7.257)=1418。前者是带冲突的停滞，后者是带冲突的形状/位置漂移。两者都由打包救援得到最终硬合法解。',
           '这支持优先观测“哪个约束持续冲突、进展是否有效、何时换分离关系/接触树或重启”。它没有证明具体关系不可行，更没有证明某种ML/RL策略一定有效；零弹性可行性及策略收益仍需受控实验。保留当前合法性守卫和救援。',
           '模型侧至少要区分两项任务：减少正常样例的几何修复量；提供或选择更合适的分离/接触关系以避免少数实例停滞。单看坐标误差或重叠面积，无法区分这两种缺口。输入特征与过程反馈应分别用于事前判断和动态控制。','',
           '## 两层迭代不是同一种次数','',
           '控制器每次shape调用都会重新构建并求解一个凸二阶锥规划（带半正定二次目标）。块间左右/上下分离选择与组内接触树在锥规划外启发式确定；锥规划内部调整所有位置和可变宽高，并保留面积、固定规格及预放置约束。每次求解内部还执行多步数值迭代。它不是全局混合整数规划，也不证明找到了最佳拓扑。',
           '面积下界以二阶锥编码，上界用当前形状附近的保守线性约束；宽高单步限制为输入的0.4～2.5倍，线长使用迭代重加权二次近似，边界/接触/弹性分离用松弛惩罚。返回Solved仅说明该子问题达到了求解标准，不保证清理后的几何通过外部硬守卫。','',
           '| 量 | 中位数 | P95 | 最小～最大 |','|---|---:|---:|---:|']
    for k,label in [('root_shape_calls','主流程shape次数'),('all_shape_calls','含递归shape次数'),('all_native_iterations','每例内部数值迭代总数'),('mean_native_iterations_per_call','主流程每次内部迭代数'),('root_shape_cumulative_center','主流程累计平均中心位移/块单位'),('root_path_to_net_ratio','搜索路径长度/最终净位移'),('first_legal_shape_step','首次硬合法shape序号（有者）')]:
        a=s[k];lines.append(f"| {label} | {a['median']:.4g} | {a['p95']:.4g} | {a['min']:.4g}～{a['max']:.4g} |")
    lines+=['','## 单步与最终幅度','',
            '| 量（每例先对全部块统计） | 首轮中位数 | 首轮P95 | draft→最终中位数 | draft→最终P95 |','|---|---:|---:|---:|---:|']
    for k,label in [('center_mean','平均中心位移/块边长'),('center_p95','块中心位移P95/块边长'),('center_detranslated_mean','去整体平移后的平均位移'),('moved_gt_0_5_fraction','移动超过0.5块边长的块比例'),('log_AR_mean','平均绝对log宽高比变化'),('AR_factor_ge_2_fraction','宽高比变化至少2倍的块比例')]:
        a=s['first_shape'][k];b=s['draft_to_final'][k];lines.append(f"| {label} | {a['median']:.4g} | {a['p95']:.4g} | {b['median']:.4g} | {b['p95']:.4g} |")
    lines+=['','主流程累计值按每次实际当前状态的变化累加，首次替换行保底不在内。路径/净位移比还纳入非shape中间转换；跨递归重启不拼接成一条虚假的连续路径。形状函数原始输出与标准化后状态分别保留在steps.json。','',
            '## 模块数和预算必须控制','', '| 模块数 | 例数 | 平均shape次数（含递归） | shape总秒/例 | 每次内部迭代 | 累计位移 | 救援例数 |','|---|---:|---:|---:|---:|---:|---:|']
    for t in report['tiers']:lines.append(f"| {t['n_range']} | {t['count']} | {t['all_shape_calls']:.2f} | {t['all_shape_seconds']:.4f} | {t['mean_native_iterations_per_call']:.2f} | {t['root_shape_cumulative_center']:.3f} | {t['rescue_cases']} |")
    lines+=['','大实例预算分段更大，同时单次规划更贵；次数不等于难度。下面相关性控制了模块数秩和60/90/110分段预算，仅作探索性相关，不作因果或显著性结论。','',
            '| 目标 | 输入特征 | 原始Spearman | 控制规模/预算后的相关 |','|---|---|---:|---:|']
    for target,table in report['input_associations'].items():
        for t in table[:5]:lines.append(f"| {target} | {t['feature']} | {t['spearman']:+.3f} | {t['size_budget_adjusted_rho']:+.3f} |")
    lines+=['','## 调整最大/调用最多的实例','', '| 排名口径 | index | 模块数 | 对应值 | 救援 | 首次合法shape |','|---|---:|---:|---:|---|---|']
    for key,rr in report['rankings'].items():
        for r in rr[:5]:lines.append(f"| {key} | {r['index']} | {r['n']} | {r[key]:.4g} | {r['rescue']} | {r['first_legal_shape_step']} |")
    lines+=['','## 数值、守卫与收益','',f"全部shape状态：{report['status_counts_all_shapes']}。主流程清理后被硬守卫拒绝的原因（可重叠计数）：{report['root_guard_reasons']}。",
            f"174次有返回布局但被拒绝的主流程shape均有重叠；{report['rejected_geometry']['calls_max_depth_above_1e_3_block_units']}次最大穿透超过0.001块单位，{report['rejected_geometry']['calls_max_depth_below_1e_5_absolute']}次低于1e-5原始长度。不能将所有拒绝归为浮点误差。统计采用实际送入守卫的清理后候选，另保存被拒绝后继续搜索的原始状态，二者不混用。",
            f"全部shape累积计时：{report['timings_all_shapes']}。墙钟、子计时及求解器原生计时不应强行相减成精确剩余开销。",
            f"主流程未得到硬合法shape的实例：{report['root_first_legal_missing']}；救援实例：{report['rescue_cases']}。最终100例均硬合法。",
            '没有更新最佳解的时间不等于浪费：被拒绝的候选仍可成为当前搜索状态；细微变化也会决定精确接触。首次合法之后到最终Q改善包含投影/救援/认证，不能全部归功于后续shape。','',
            '控制规模/预算后，重叠穿透深度与累计位移的相关约+0.699，而与调用次数仅约+0.223；预放置比例与每次内部数值迭代约+0.540。这些相关在去掉两个救援例后分别约+0.701、+0.218、+0.528。说明“要搬多远”“求解几轮”“每轮数值多难”需要分开描述。',
            f"最终仍有边界违反的实例{report['remaining_final_soft']['boundary_violations']['cases_with_violation']}例、分组违反{report['remaining_final_soft']['grouping_violations']['cases_with_violation']}例，MIB违反0例；前两类可以重叠。最常见的剩余质量方向是边界/连通接触，而非本面板上的MIB形状一致性。",'',
            '## 后续观察重点','',
            '1. 几何与约束负担：重叠冲突图的连通块/中心块/穿透深度；各边界请求距离与角落需求；分组连通分量、候选桥接间隙和接触长度；MIB/分组/边界/固定规格的耦合。',
            '2. 离散关系选择：每轮左右/上下分离翻转率、组接触树替换率、重复拓扑、分离图长链/环、固定障碍冲突。现轨迹可重建主流程翻转率；精确反事实可行性仍需另设实验。',
            '3. 数值与硬合法：规划规模、装配/初始化/原生求解耗时、状态、迭代数；宽高步限和面积容差贴边比例。以后应增加原始可行性/对偶残差、松弛量、活跃约束和对偶权重，逐项定位究竟哪个约束迫使解迁移。',
            '4. 搜索有效性：首次可行时间、最佳解更新时间、净位移/累计路径、折返、各步代理与Q分项收益、末端接触稳定性、预算退出/平台退出、救援前后关系变化。可行草稿的Q才可与最终Q作质量比较，非法draft的罚分不能当巨大质量收益。',
            '5. 模型缺口：位置、形状、分离关系、接触选择、约束优先级及预算/救援决策分开评估。观测决定下一轮受控实验；不能由“移动大”直接推出模型应监督最终坐标，也不能由相关性直接选RL奖励。','',
            '[逐例指标](cases.json) · [逐次shape指标](steps.json) · [汇总/输入相关性](summary.json) · [幅度和轨迹图](trajectory_diagnostics.png) · [持续冲突/漂移图](conflict_diagnostics.png)',
            '[329布局](case_329.png) · [496布局](case_496.png) · [127布局](case_127.png)','']
    (OUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(report,rows,steps,selections,cases):
    import os
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization.layouts import plot_case,geometry_details
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    n=np.array([r['n'] for r in rows]);path=np.array([r['root_shape_cumulative_center'] for r in rows]);count=np.array([r['all_shape_calls'] for r in rows])
    sc=axes[0,0].scatter(count,path,c=n,cmap='viridis',s=32);fig.colorbar(sc,ax=axes[0,0],label='Blocks');axes[0,0].set(xlabel='Shape calls including rescue',ylabel='Root cumulative mean movement / block length')
    for r in sorted(rows,key=lambda r:r['root_shape_cumulative_center'],reverse=True)[:3]:axes[0,0].annotate(str(r['index']),(r['all_shape_calls'],r['root_shape_cumulative_center']))
    axes[0,1].scatter(n,[r['all_shape_seconds'] for r in rows],c=count,cmap='plasma');axes[0,1].set(xlabel='Blocks',ylabel='All shape wall seconds')
    for k in range(1,13):
        a=[s['state_move']['center_mean'] for s in steps if s['step']==k and 'state_move' in s]
        if not a:continue
        axes[1,0].boxplot([a],positions=[k],widths=.55,showfliers=False);axes[1,0].text(k,.96,str(len(a)),transform=axes[1,0].get_xaxis_transform(),ha='center',fontsize=8)
    axes[1,0].set(xlabel='Root shape step (labels = observed cases)',ylabel='Mean center movement / block length',yscale='log')
    selected=[r['index'] for r in sorted(rows,key=lambda r:r['root_shape_cumulative_center'],reverse=True)[:2]]
    selected+=report['rescue_cases'];selected=list(dict.fromkeys(selected))
    for idx in selected:
        a=[s for s in steps if s['index']==idx and 'state_move' in s]
        axes[1,1].plot([s['step'] for s in a],[s['state_move']['center_mean'] for s in a],marker='.',label=f'index {idx}')
    axes[1,1].set(xlabel='Root shape step',ylabel='Mean center movement / block length',yscale='log');axes[1,1].legend()
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('Saved dev100 trajectories; no new solves; root vs recursive calls kept separate')
    fig.tight_layout();fig.savefig(OUT/'trajectory_diagnostics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for idx in selected:
        a=[s for s in steps if s['index']==idx and 'state_conflicts' in s]
        xx=[s['step'] for s in a]
        axes[0].plot(xx,[s['state_conflicts']['max_penetration_block_units'] for s in a],label=f'index {idx}')
        axes[1].plot(xx,[s['state_bbox_over_total_area'] for s in a])
        axes[2].plot(xx,[np.exp(s['state_log_AR_max']) for s in a])
    axes[0].set_ylabel('Maximum overlap penetration / block length');axes[0].legend()
    axes[1].set_ylabel('Current bbox / public effective total area');axes[2].set(ylabel='Maximum aspect ratio or reciprocal',yscale='log')
    for ax in axes:ax.set_xlabel('Root shape step');ax.grid(alpha=.2)
    fig.suptitle('Persisting conflicts: stagnation (496), drift (329), ordinary repair (127)')
    fig.tight_layout();fig.savefig(OUT/'conflict_diagnostics.png',dpi=170);plt.close(fig)
    for idx in selected[:4]:
        item=selections[idx];r=item['record'];case=cases[idx]['case'];initial=item['initial'];first=item['first'];variants={}
        keys=[('draft',initial['draft']),('first',first[1] if first else None)]
        for tag,key in keys:
            if key:
                pos=r['layouts'][key];metrics=evaluate_case(case,pos,0.)['metrics'];variants[tag]={'positions':pos,'metrics':metrics,'soft_details':geometry_details(pos,case.solver_inputs[5])}
            else:variants[tag]={'positions':None,'unavailable':'No hard-legal root shape proposal'}
        variants['final']={'positions':r['positions'],'metrics':r['metrics'],'soft_details':geometry_details(r['positions'],case.solver_inputs[5])}
        plot_case({'panel':f'E35 stored trajectory index {idx}','case_id':str(r['member']),'constraints':np.asarray(case.solver_inputs[5]).tolist(),'variants':variants},OUT/f'case_{idx}.png',stages=(('draft','Draft (may be illegal)'),('first','First hard-legal root shape'),('final','Final full solver')))


if __name__=='__main__': main()
