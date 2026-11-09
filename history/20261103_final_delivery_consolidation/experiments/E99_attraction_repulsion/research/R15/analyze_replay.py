"""Inspect every saved module/operation; no new trajectory or planner solve."""
import hashlib,json
from pathlib import Path
import numpy as np
from .replay import OUT,cached,read,save
from .operator import variants


def main():
    if (OUT/'analysis.json').exists():raise FileExistsError('Saved analysis protected')
    results=read(OUT/'results.json');assert results['status']=='complete';summary={}
    for arm in variants():
        counts=dict(module_operations=0,own_step_bound_active=0,not_own_limited=0,suppressed_by_other_module=0,
            shape_operations=0,shape_clipped=0,active_pairs_ever=0,repeated_overlap_toggle_pairs=0,movable_end_modules=0,path_over_twice_displacement=0)
        limiter_mass=[];alphas=[];stress_a=0.;stress_r=0.;raw_shape=0.;applied_shape=0.;quartiles=[[] for _ in range(4)]
        for row in results['records']:
            if row['arm']!=arm:continue
            draft,inputs,_=cached(row['index']);c=np.asarray(inputs[5]);movable=c[:,1]==0;soft=~(c[:,:2]!=0).any(1)
            area=draft[:,2:].prod(1);mass=area/np.median(area);local=draft[:,2:].min(1)
            with np.load(OUT/'trajectories'/arm/f"{row['index']:04d}.npz") as a:
                m=a['module_log'];r=a['round_log'];x=a['history']
            raw=np.linalg.norm(m[:,:,6:8],axis=2);budget=m[:,:,8]
            eps=256*np.finfo(float).eps*np.maximum(1,np.abs(draft).max(1))
            active=(raw>eps[None])&movable[None]
            own=raw>budget+eps[None];not_own=active&~own
            counts['module_operations']+=int(active.sum());counts['own_step_bound_active']+=int((active&own).sum())
            counts['not_own_limited']+=int(not_own.sum());counts['suppressed_by_other_module']+=int((not_own&(r[:,4,None]<.999)).sum())
            counts['shape_operations']+=int(soft.sum())*len(m);counts['shape_clipped']+=int((np.abs(m[:,:,15]-m[:,:,16])[:,soft]>1e-12).sum())
            raw_shape+=float(np.abs(m[:,:,15])[:,soft].sum());applied_shape+=float(np.abs(m[:,:,16])[:,soft].sum())
            stress_a+=float(np.abs(m[:,:,13])[:,soft].sum());stress_r+=float(np.abs(m[:,:,14])[:,soft].sum())
            for t in range(len(r)):
                lim=int(r[t,6])
                if lim>=0 and r[t,4]<.999:limiter_mass.append(float(mass[lim]))
                alphas.append(float(r[t,4]))
            i,j=np.triu_indices(len(draft),1);lo=x[:,:,:2];hi=lo+x[:,:,2:]
            overlap=(np.minimum(hi[:,i],hi[:,j])-np.maximum(lo[:,i],lo[:,j])>1e-8).all(2)
            toggles=(overlap[1:]!=overlap[:-1]).sum(0);ever=overlap.any(0)
            counts['active_pairs_ever']+=int(ever.sum());counts['repeated_overlap_toggle_pairs']+=int((toggles>=4).sum())
            delta=np.linalg.norm(x[-1,:,:2]+x[-1,:,2:]/2-draft[:,:2]-draft[:,2:]/2,axis=1)
            counts['movable_end_modules']+=int(movable.sum());counts['path_over_twice_displacement']+=int((movable&(m[-1,:,12]>2*delta+1e-8)).sum())
            bins=np.digitize(area,np.quantile(area,[.25,.5,.75]),right=True)
            for q in range(4):
                ids=movable&(bins==q)
                if ids.any():quartiles[q].append(dict(mean_mass=float(mass[ids].mean()),mean_path_fraction=float((m[-1,ids,12]/local[ids]).mean()),
                    mean_displacement_fraction=float((delta[ids]/local[ids]).mean()),mean_attractive_force=float(np.linalg.norm(m[:,ids,1:3],axis=2).mean())))
        counts.update(own_bound_fraction=counts['own_step_bound_active']/max(1,counts['module_operations']),
            other_module_suppression_fraction=counts['suppressed_by_other_module']/max(1,counts['not_own_limited']),
            shape_clipped_fraction=counts['shape_clipped']/max(1,counts['shape_operations']),
            repeated_toggle_fraction=counts['repeated_overlap_toggle_pairs']/max(1,counts['active_pairs_ever']))
        summary[arm]=dict(counts=counts,common_scale_median=float(np.median(alphas)),
            limiter_mass_median=float(np.median(limiter_mass)) if limiter_mass else None,
            attraction_shape_stress_fraction=stress_a/max(stress_a+stress_r,1e-30),raw_shape_absolute_sum=raw_shape,applied_shape_absolute_sum=applied_shape,
            mass_quartiles=[dict(quartile=q+1,**{k:float(np.mean([r[k] for r in items])) for k in items[0]}) if items else dict(quartile=q+1) for q,items in enumerate(quartiles)])
    save(OUT/'analysis.json',dict(status='complete',source_results_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),summary=summary,
        no_new_trajectory=True,no_new_planner_solve=True,test_read=False))
    lines=['# 训练来源500例：逐模块与逐轮机制审计','', '四个预先登记的控制，每个500例、20轮；原完整后处理器冻结，本阶段没有后处理求解。以下只是草稿机制证据，不能替代最终Q。','',
        '| 方案 | 有害整图冻结例数 | 草稿交面积均值 | 真留白率均值 | 最坏模块覆盖均值 | 平均中心位移/短边 | 用满移动预算模块比例 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in results['summary'].items():lines.append(f"| {arm} | {s['cases_with_harmful_global_freeze']} | {s['mean_raw_overlap']:.3%} | {s['mean_exact_void_fraction']:.3%} | {s['mean_worst_module_coverage']:.3%} | {s['mean_center_displacement_fraction']:.3%} | {s['path_budget_hit_module_fraction']:.3%} |")
    lines+=['', '| 方案 | 自身单步限额触及比例 | 自己不限幅却被其他模块压小的比例 | 限制模块质量中位数 | 形变更新被裁剪比例 | 反复进出重叠的模块对比例 |','|---|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        c=s['counts'];lines.append(f"| {arm} | {c['own_bound_fraction']:.3%} | {c['other_module_suppression_fraction']:.3%} | {s['limiter_mass_median']} | {c['shape_clipped_fraction']:.3%} | {c['repeated_toggle_fraction']:.3%} |")
    lines+=['', '大面积作为惯性质量和作为吸引源强度是两件事。当前有界面积耦合C=2mi*mj/(mi+mj)，对固定目标i随源mj递增且C/mi<2；成对力仍相等相反，有限范围不变。仅二维矩形/多体场的实际轨迹和最终Q才能决定收益。','', '在一维、固定方向、近接触的孤立吸引对中，C*(1/mi+1/mj)=2，因此相对间隙的一阶更新率不依赖质量比；这不构成二维、多体和形变全系统的稳定性证明。','', '反复进出重叠是轨迹诊断信号，也可能由相邻模块/形变驱动，不能直接当成数值振荡的证明。原始每模块力和轨迹保留，可逐例追踪。', '', '恢复完整后处理比较时必须保持原planner/配置/检查不变；本轮不因草稿指标而宣布任何方案最终更优。测试封存。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({arm:dict(counts=s['counts'],limiter_mass_median=s['limiter_mass_median']) for arm,s in summary.items()},indent=2))

if __name__=='__main__':main()
