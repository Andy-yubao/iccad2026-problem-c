"""One terminal fixed dev500, using the independently copied E98 depth solver."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info
from .io import ROOT, HERE, save, sha, sources, snapshot, exclusive
from .train import RUN
from .data import load_rows
from .predictor import Predictor
from .metrics import module_overlap
from .depth_evaluation.solver import optimize
from .depth_evaluation.repair.canonical_native import warmup
from datasets.reference_layouts import registry, public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator

BASE=ROOT/'experiments/E104_scaling32k/runs/20261104_scaling128x16_seed0'
OUT=RUN/'evaluation_depth_dev500'

def stopped():
    return (HERE/'STOP').exists() or (RUN/'STOP').exists()

def request(predictor, row):
    inputs=row['case'].solver_inputs
    torch.cuda.synchronize()
    start=time.perf_counter()
    draft=predictor.predict(inputs)
    draft_seconds=time.perf_counter()-start
    final, stats=optimize(draft, inputs)
    hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),
                                public_hard_reference(inputs[1],inputs[5],inputs[6]))
    wall=time.perf_counter()-start
    if not hard['is_feasible'] or not np.isfinite(final).all():
        raise RuntimeError('Independent final hard legality failed')
    scoring_start=time.perf_counter()
    scored=evaluate_case(row['case'],final,wall)
    raw=evaluate_case(row['case'],draft,draft_seconds)
    if not scored['metrics']['is_feasible']:
        raise RuntimeError('Official hard legality failed')
    return dict(member=list(row['member']),n=int(row['case'].block_count),
        complete_seconds=wall,draft_seconds=draft_seconds,repair_seconds=stats['seconds'],
        metrics=scored['metrics'],independent_hard_checks=hard,raw_metrics=raw['metrics'],
        raw_overlap=module_overlap(draft),draft_positions=draft.tolist(),positions=final.tolist(),
        fallback=stats['row_retained'],trace=stats['trace'],work=stats['work'],
        scoring_geometry_diagnostics_seconds=time.perf_counter()-scoring_start,test_read=False)

def summarize(records):
    q=np.array([r['metrics']['cost'] for r in records])
    wall=np.array([r['complete_seconds'] for r in records])
    wq=float(official_evaluator().compute_total_score(q.tolist(),[r['n'] for r in records]))
    return dict(count=len(records),weighted_Q=wq,mean_Q=float(q.mean()),mixed_Q=(wq+float(q.mean()))/2,
        p95_Q=float(np.percentile(q,95)),max_Q=float(q.max()),
        hard_legal=sum(bool(r['metrics']['is_feasible']) for r in records),
        independent_hard_legal=sum(bool(r['independent_hard_checks']['is_feasible']) for r in records),
        raw_hard_legal=sum(bool(r['raw_metrics']['is_feasible']) for r in records),
        fallbacks=sum(r['fallback'] for r in records),mean_complete_seconds=float(wall.mean()),
        p95_complete_seconds=float(np.percentile(wall,95)),max_complete_seconds=float(wall.max()),
        over250ms=int((wall>.25).sum()),mean_draft_module_pass10=float(np.mean([r['raw_overlap']['module_pass10_fraction'] for r in records])),
        mean_draft_max_covered_fraction=float(np.mean([r['raw_overlap']['max_covered_fraction'] for r in records])),
        mean_draft_bbox_over_A=float(np.mean([r['raw_overlap']['bbox_over_A'] for r in records])))

def report(current, baseline, best, records, rows, timings):
    history=json.loads((RUN/'progress.json').read_text(encoding='utf-8'))['history']
    old_history=json.loads((BASE/'progress.json').read_text(encoding='utf-8'))['history']
    old_best=min(old_history,key=lambda h:h['dev_monitor']['total'])
    train=best['train_monitor']['total'];dev=best['dev_monitor']['total']
    old_train=old_best['train_monitor']['total'];old_dev=old_best['dev_monitor']['total']
    current.update(train_monitor_loss=train,dev_monitor_loss=dev,absolute_loss_gap=dev-train,
        relative_loss_gap=(dev-train)/abs(train),baseline_dev_loss_relative_change=dev/old_dev-1,
        baseline_train_loss_relative_change=train/old_train-1)
    comparison=dict(baseline=baseline,current=current,baseline_train_loss=old_train,
        baseline_dev_loss=old_dev,baseline_absolute_gap=old_dev-old_train,
        baseline_relative_gap=(old_dev-old_train)/abs(old_train),best_epoch=best['epoch'],
        fixed_loss_terms_at_best=dict(train=best['train_monitor'],dev=best['dev_monitor']),
        timings=timings,test_read=False)
    save(OUT/'comparison.json',comparison)
    # Two runs on the same fixed monitor sets, with both epoch and sample axes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for hs,name,count in [(old_history,'E104 32k / 128x16',32000),(history,'E105 32k / 128x16',32000)]:
        for role,style in [('train_monitor','-'),('dev_monitor','--')]:
            values=[h[role]['total'] for h in hs]
            axes[0].plot([h['epoch'] for h in hs],values,style,label=name+' '+role.split('_')[0])
            axes[1].plot([h['epoch']*count for h in hs],values,style,label=name+' '+role.split('_')[0])
    for ax,label in zip(axes,['Epoch','Gradient-training sample presentations']):
        ax.set_xlabel(label);ax.set_ylabel('Fixed unaugmented monitor Loss');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'loss_curves.png',dpi=160);plt.close(fig)
    text='# E105方案一终末评价\n\n'
    text+='唯一模型由固定dev500原Loss选择，冻结后仅一次同500完整评价。E104修复后best81已完成500结果只读复用。\n\n'
    text+='|指标|E104修复版32k / 128×16|E105方案一32k / 128×16|\n|---|---:|---:|\n'
    pairs=[('train500 Loss',old_train,train),('dev500 Loss',old_dev,dev),('绝对Loss间隙',old_dev-old_train,dev-train),
        ('相对Loss间隙',(old_dev-old_train)/abs(old_train),(dev-train)/abs(train))]
    pairs += [(k,baseline[k],current[k]) for k in ('weighted_Q','mean_Q','mixed_Q','hard_legal','raw_hard_legal','fallbacks','mean_complete_seconds','p95_complete_seconds','max_complete_seconds','over250ms')]
    for name,left,right in pairs:text+=f'|{name}|{left:.9g}|{right:.9g}|\n'
    text+=f'\nLoss-best epoch={best["epoch"]}。dev Loss相对变化={dev/old_dev-1:+.3%}，train Loss相对变化={train/old_train-1:+.3%}。绝对Loss和泛化间隙同时报告，不以间隙变小单独认定改善。\n\n'
    text+='正常dev早停表示登记调度的平台，训练拟合另看曲线。E105为全程固定的方案一、随机初始化32k训练，唯一best由原固定Loss选择；E104历史best81包含step20350的local-prenorm迁移，且这里只保留其修复后曲线，因此该历史比较不能单独归因于结构，未额外启动从头控制组。本次未实施方案二/三，不拟合scaling指数。dev为反复使用的开发面板，单seed，不称独立未见泛化或正式比赛成绩；测试封存。\n\n'
    text+='完整逐例时间含公开输入构造/传输/网络/后处理/独立硬检查；评分、磁盘、绘图、冷启动另存comparison.json；历史和本次并发/温度不同，延迟只作实测描述。训练吞吐与完整epoch另见steps.jsonl/progress.json。\n\n![Loss curves](loss_curves.png)\n\n[同100例图集](gallery/index.html)\n'
    (OUT/'report.md').write_text(text,encoding='utf-8')
    from visualization.layouts import plot_case,geometry_details
    gallery=OUT/'gallery';gallery.mkdir(exist_ok=True);manifest=[]
    for i,row in enumerate(rows[:100]):
        if stopped():raise InterruptedError('STOP during gallery')
        record=records[i];constraints=np.asarray(row['case'].solver_inputs[5]).tolist()
        reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
        variants={name:dict(positions=p,soft_details=geometry_details(p,constraints)) for name,p in
            [('reference',reference),('raw',record['draft_positions']),('final',record['positions'])]}
        image=gallery/f'{i:03d}.png'
        plot_case(dict(case_id=f'dev500:{i}',panel='E105 scheme1 / frozen Loss-best / depth v1',constraints=constraints,variants=variants),
            image,stages=(('reference','Reference'),('raw','E105 raw'),('final','Depth final')),show_metrics=False)
        manifest.append(dict(index=i,member=list(row['member']),image=image.name))
    save(gallery/'manifest.json',dict(cases=manifest,score_controls=False,shared_ids_and_scale=True,test_read=False))
    (gallery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E105 dev500</title><style>img{width:100%}</style>'+''.join(f'<p>{r["index"]}</p><img loading="lazy" src="{r["image"]}">' for r in manifest),encoding='utf-8')

@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import numba
    numba.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if stopped():raise InterruptedError('STOP before evaluation')
    checkpoint=RUN/'checkpoints/best.pt'
    progress=json.loads((RUN/'progress.json').read_text(encoding='utf-8'))
    if progress['status']!='early_stop':raise RuntimeError('Only normal early-stop may trigger evaluation')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['sources']!=sources():raise ValueError('Frozen training source drift')
    if state['progress']['best_epoch']!=progress['best_epoch']:raise ValueError('Best identity drift')
    best=next(h for h in progress['history'] if h['epoch']==progress['best_epoch'])
    baseline_path=BASE/'evaluation_depth_dev500/results.json'
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    if sha(baseline_path)!=plan['baseline_evaluation_sha256']:raise ValueError('Baseline evidence drift')
    if sha(BASE/'progress.json')!=plan['baseline_progress_sha256']:raise ValueError('Baseline Loss history drift')
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    del state
    OUT.mkdir(exist_ok=False)
    start=time.perf_counter()
    with exclusive(HERE/'evaluation.lock'),threadpool_limits(limits=1):
        save(OUT/'state.json',dict(status='loading',pid=os.getpid(),completed=0,test_read=False))
        rows=load_rows('dev_monitor500')
        expected=[r['member'] for r in registry()['sets']['dev_monitor500']['records']]
        if [list(r['member']) for r in rows]!=expected:raise ValueError('Fixed dev panel changed')
        binding=dict(checkpoint_sha256=sha(checkpoint),best_epoch=best['epoch'],best_loss=best['dev_monitor']['total'],
            sources=sources(),members=expected,baseline_evaluation_sha256=sha(baseline_path),selection_frozen_before_Q=True,test_read=False)
        save(OUT/'frozen.json',binding);snapshot(OUT/'source_snapshot.zip',binding)
        cold=time.perf_counter();warmup();predictor=Predictor(checkpoint,rows[0]['case'].solver_inputs)
        timings=dict(initialization_seconds=time.perf_counter()-cold,data_binding_seconds=cold-start,
            torch_intra=torch.get_num_threads(),torch_inter=torch.get_num_interop_threads(),pools=threadpool_info())
        assert all(p['num_threads']<=1 for p in timings['pools'])
        save(OUT/'initialization.json',timings)
        records=[];disk=0.
        for i,row in enumerate(rows):
            if stopped():raise InterruptedError('STOP during evaluation')
            record=request(predictor,row);records.append(record)
            t=time.perf_counter();save(OUT/f'cases/{i:04d}.json',record);disk+=time.perf_counter()-t
            if (i+1)%10==0:
                save(OUT/'state.json',dict(status='evaluating',completed=i+1,total=500,pid=os.getpid(),test_read=False))
                print(f'dev500 {i+1}/500',flush=True)
        if sha(checkpoint)!=binding['checkpoint_sha256'] or sources()!=binding['sources']:raise ValueError('Evaluation identity drift')
        current=summarize(records)
        current.update(status='metrics_complete',checkpoint_sha256=binding['checkpoint_sha256'],best_epoch=best['epoch'],test_read=False,selection_frozen_before_Q=True)
        save(OUT/'results.json',current)
        timings.update(case_disk_seconds=disk,scoring_diagnostics_seconds=sum(r['scoring_geometry_diagnostics_seconds'] for r in records),
            sequential_evaluation_wall_seconds=time.perf_counter()-start,bulk_inference_throughput_not_measured=True)
        t=time.perf_counter();report(current,baseline,best,records,rows,timings)
        timings['report_gallery_seconds']=time.perf_counter()-t
        save(OUT/'timings.json',timings)
        current.update(status='complete',elapsed_seconds=time.perf_counter()-start)
        save(OUT/'results.json',current);save(OUT/'state.json',dict(status='complete',completed=500,test_read=False))
        print(json.dumps(current),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        save(OUT/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False,no_automatic_retry=True))
        raise
