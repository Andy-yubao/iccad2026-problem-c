"""Frozen worst10, unchanged E50, ten-second budget; paired saved five-second results."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['MPLBACKEND']='Agg'
import argparse,json,time,sys,zipfile,atexit
from pathlib import Path
import numpy as np
import torch
from .run import read,save,sha,CHECKPOINT,HERE,ROOT

SOURCE=HERE/'runs/20261025_full500_parallel_v3'
OUT=HERE/'runs/20261025_worst10_budget10s'
IDS=[450,307,467,123,366,202,164,368,246,493]

def report():
    from visualization.layouts import plot_case,geometry_details
    from visualization.saved_gallery import write_gallery
    from datasets.reference_layouts import ReferenceLayouts
    data=ReferenceLayouts('dev_monitor500');table=[]
    (OUT/'images').mkdir(exist_ok=True)
    manifest={'title':'最差10例：原草稿／E50五秒／E50十秒',
              'note':'左：冻结E42草稿｜中：原500例运行的E50五秒结果｜右：同算法新跑E50十秒结果。相同模块编号与坐标尺度；只改变预算，邻接意图尚无经人工校准的自动指标。',
              'stage_labels':['草稿','E50五秒','E50十秒'],
              'groups':[{'key':'budget','label':'固定最差10例：5秒→10秒','indices':IDS}], 'cases':[]}
    for i in IDS:
        old=read(SOURCE/'cases'/f'{i:04d}.json');new=read(OUT/'cases'/f'{i:04d}.json')
        a=next(c for c in old['solver']['candidates'] if c['id']==old['solver']['selected_id'])
        b=next(c for c in new['solver']['candidates'] if c['id']==new['solver']['selected_id'])
        bank=new['solver']['candidates'];best=min(bank,key=lambda c:c['metrics']['cost'])
        row={'index':i,'n':new['n'],'old_Q':old['selected_metrics']['cost'],'new_Q':new['selected_metrics']['cost'],
             'delta_Q':new['selected_metrics']['cost']-old['selected_metrics']['cost'],
             'old_D':a['distance']['D'],'new_D':b['distance']['D'],
             'old_near_direction_change':a['distance']['near_relation_change_fraction'],
             'new_near_direction_change':b['distance']['near_relation_change_fraction'],
             'mean_movement_module_units':b['distance']['mean_center_local'],'p95_movement_module_units':b['distance']['p95_center_local'],
             'best_found_Q':best['metrics']['cost'],'old_best_found_Q':min(c['metrics']['cost'] for c in old['solver']['candidates']),
             'old_soft':a['metrics']['total_soft_violations'],'new_soft':b['metrics']['total_soft_violations'],
             'new_area_gap':b['metrics']['area_gap'],'new_wire_gap':b['metrics']['hpwl_gap'],
             'old_candidates':len(old['solver']['candidates']),'new_candidates':len(bank),
             'full_seconds':new['full_seconds'],'old_full_seconds':old['full_seconds'],
             'hard_legal':new['selected_metrics']['is_feasible'],'draft_max_abs_difference':new['draft_max_abs_difference']}
        table.append(row)
        item=data[i];assert list(item['member'])==old['member']==new['member']
        constraints=np.asarray(item['case'].solver_inputs[5]).tolist()
        variants={'raw':{'positions':old['draft']},'old':{'positions':a['positions'],'metrics':a['metrics']},'new':{'positions':b['positions'],'metrics':b['metrics']}}
        for v in variants.values():v['soft_details']=geometry_details(v['positions'],constraints)
        image=OUT/'images'/f'case_{i:04d}.png'
        plot=plot_case({'panel':'E50 budget comparison','case_id':f'dev index{i}','constraints':constraints,'variants':variants,
                       'caption':f"Same draft and IDs. Near-neighbor direction-change proxy: {row['old_near_direction_change']:.1%} -> {row['new_near_direction_change']:.1%}; adjacency intent not certified."},image,
                       stages=(('raw','Frozen E42 draft'),('old','E50 five-second result'),('new','E50 ten-second result')))
        manifest['cases'].append({'id':i,'group':'budget','image':image.relative_to(OUT).as_posix(),'axes_bounds':plot['axes_bounds'],'metrics':row})
        print(f'Rendered paired index{i}',flush=True)
    q=np.array([r['new_Q'] for r in table]);b=np.array([r['old_Q'] for r in table]);n=np.array([r['n'] for r in table]);w=np.exp((n-n.max())/12)
    result={'count':10,'hard_legal':sum(r['hard_legal'] for r in table),'weighted_Q_5s':float(np.average(b,weights=w)),
            'weighted_Q_10s':float(np.average(q,weights=w)),'mean_Q_5s':float(b.mean()),'mean_Q_10s':float(q.mean()),
            'win_tie_loss':[int(sum(q<b-1e-8)),int(sum(abs(q-b)<=1e-8)),int(sum(q>b+1e-8))],
            'Q_gt1_2_5s':int(sum(b>1.2)),'Q_gt1_2_10s':int(sum(q>1.2)),
            'mean_seconds':float(np.mean([r['full_seconds'] for r in table])),'max_seconds':max(r['full_seconds'] for r in table),
            'over10s':sum(r['full_seconds']>10 for r in table),'paired_cases':table,
            'scope':'post-selected worst10 development cases; one new run; unchanged solver and same draft verified; not independent generalization or intrinsic difficulty proof'}
    save(OUT/'summary.json',result);save(OUT/'manifest.json',manifest);write_gallery(OUT,manifest)
    lines=['# E50最差10例：5秒与10秒预算对照','',
           '按用户2026-10-25要求重新运行原500开发结果中最差的固定10例。算法、E42 best51、CPU推理2线程、搜索2+2线程与在线选解规则保持原样，只将单例预算从5秒增至10秒。常规保护轨仍按原算法上限2.3秒，增加的预算主要用于探索轨。真实Q只在事后评价，不反馈搜索。', '',
           '每例重新执行冻结推理并核对草稿与原500结果最大绝对差≤1e-8。完整计时仍包含推理、候选生成、独立检查、调度及最终选中解评分；初始化、磁盘写入、所有其他候选Q诊断另计。原五秒结果保持原样。本次是一次真实十秒运行，时限截断与数值路径可能造成非单调变化，不保证延长时间逐例只会改善。', '',
           f"硬合法{result['hard_legal']}/10；胜/平/负{result['win_tie_loss']}。这10例子集加权Q {result['weighted_Q_5s']:.8f}→{result['weighted_Q_10s']:.8f}，普通均值{b.mean():.8f}→{q.mean():.8f}；Q>1.2由{result['Q_gt1_2_5s']}例变为{result['Q_gt1_2_10s']}例。完整均时{result['mean_seconds']:.4f}秒，最大{result['max_seconds']:.4f}秒，超10秒{result['over10s']}例。仅代表这10个事后挑选的难例，不能当成完整500的新分数。",'',
           '| index | 5秒Q | 10秒Q | 变化 | 10秒库内最好Q | 软违约5→10 | 近邻方向变化5→10 |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['index']} | {r['old_Q']:.7f} | {r['new_Q']:.7f} | {r['delta_Q']:+.7f} | {r['best_found_Q']:.7f} | {r['old_soft']}→{r['new_soft']} | {r['old_near_direction_change']:.1%}→{r['new_near_direction_change']:.1%} |")
    lines += ['', '用户已澄清“尊重草稿”优先指相对方向及意图邻接保持，不要求小位移。上表沿用原有最近三中心邻居方向代理，只反映一个局部几何侧面，并非经人工校准的意图邻接指标；本轮不因此改算法或追溯解释旧D。新旧图统一编号与坐标尺度，可直接观察拓扑意图是否保留。', '',
              '[打开10例三相图集](index.html)：原草稿／原五秒E50／新十秒E50。', '',
              '改善支持当前草稿仍有本算法可兑现的修复潜力；没有改善只能说明额外预算未在当前算法中兑现，不能单独证明草稿质量低、已达最优或不存在近邻好解。若库内好Q但选中差Q，应区分选择瓶颈与搜索瓶颈。延长预算只作用于当前搜索机制，不等于换成更强的拓扑探索。', '',
              '复现：`python -m experiments.E50_anytime_quality.rerun_worst10 --resume`；仅重建图表：`python -m experiments.E50_anytime_quality.rerun_worst10 --report-only`。']
    (OUT/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    if args.report_only:report();return
    assert read(SOURCE/'execution.json')['complete']
    chosen=read(HERE/'runs/20261025_full500_review/analysis.json')['worst10'];assert [r['index'] for r in chosen]==IDS
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts}
    config={'indices':IDS,'case_seconds':10.,'numerical_threads':4,'inference_threads':2,'source_sha256':sources,
            'checkpoint_sha256':sha(CHECKPOINT),'registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),
            'source_5s_config_sha256':sha(SOURCE/'config.json'),'command':sys.argv,'test_read':False,
            'scope':'user-authorized 10-case budget ablation; frozen solver and topology objective'}
    if args.dry_run:print(json.dumps(config,indent=2));return
    OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'execution.json').exists() and read(OUT/'execution.json')['complete']:report();return
    if (OUT/'config.json').exists():
        assert args.resume,'Use --resume';old=read(OUT/'config.json')
        for k in ('indices','case_seconds','source_sha256','checkpoint_sha256','registry_sha256','source_5s_config_sha256'):assert old[k]==config[k],k
    else:
        save(OUT/'config.json',config)
        with zipfile.ZipFile(OUT/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for filename in sources:z.write(ROOT/filename,filename)
    from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
    from scoring.evaluation import evaluate_case
    from .canonical_native import warmup
    from .baseline.canonical_native import warmup as baseline_warmup
    from .service import Service
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    setup=time.perf_counter();warmup();baseline_warmup();data=ReferenceLayouts('dev_monitor500')
    service=Service(CHECKPOINT);atexit.register(service.close);service.ready()
    save(OUT/'environment.json',{'initial_setup_seconds':time.perf_counter()-setup,'torch_threads':torch.get_num_threads()})
    try:
        for count,i in enumerate(IDS,1):
            file=OUT/'cases'/f'{i:04d}.json'
            if file.exists():continue
            if (OUT/'STOP').exists():break
            old=read(SOURCE/'cases'/f'{i:04d}.json');item=data[i];case=item['case'];assert list(item['member'])==old['member']
            case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
            _,stats=service.solve(case.solver_inputs,case=case,seconds=10.)
            diff=float(np.max(np.abs(np.asarray(stats['raw_positions'])-np.asarray(old['draft']))));assert diff<=1e-8,diff
            chosen=next(c for c in stats['candidates'] if c['id']==stats['selected_id'])
            start=time.perf_counter()
            for candidate in stats['candidates']:
                if 'metrics' not in candidate:candidate['metrics']=evaluate_case(case,candidate['positions'],stats['seconds'])['metrics']
            row={'index':i,'member':old['member'],'n':case.block_count,'draft':stats['raw_positions'],'draft_max_abs_difference':diff,
                 'full_seconds':stats['seconds'],'inference_seconds':stats['inference_seconds'],
                 'offline_diagnostics_seconds':time.perf_counter()-start,'selected_metrics':chosen['metrics'],'solver':stats}
            save(file,row)
            print(f"{count}/10 index{i}: Q5s={old['selected_metrics']['cost']:.7f} Q10s={chosen['metrics']['cost']:.7f} {stats['seconds']:.3f}s candidates={len(stats['candidates'])} draftdiff={diff:.2g}",flush=True)
    finally:service.close()
    count=sum((OUT/'cases'/f'{i:04d}.json').exists() for i in IDS)
    save(OUT/'execution.json',{'complete':count==10,'count':count,'requested':10})
    if count==10:report()

if __name__=='__main__':main()
