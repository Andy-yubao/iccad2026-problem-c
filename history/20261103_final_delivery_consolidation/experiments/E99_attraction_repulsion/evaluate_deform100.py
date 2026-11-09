"""V3 fixed100, matched translation-only / deformation controls, no training."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from dataclasses import replace
import datetime,hashlib,json,subprocess,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case,geometry_details
from .deformation import Config,preprocess
from .diagnostics import measures
from .planner.solver import optimize
from .planner.canonical_native import warmup
from .preflight import check_final

HERE=Path(__file__).resolve().parent
PLAN=HERE/'plan_deform_v3.json'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def stats(x):
    a=np.asarray(x);return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))
def directions(a,b):
    d=np.asarray(b)-np.asarray(a);return dict(improved=int((d< -1e-12).sum()),unchanged=int((np.abs(d)<=1e-12).sum()),worsened=int((d>1e-12).sum()))


def main():
    plan=read(PLAN);out=HERE/plan['output']
    if out.exists():raise FileExistsError('Existing evidence protected: '+str(out))
    out.mkdir(parents=True);(out/'coordinates').mkdir();(out/'gallery').mkdir()
    records=[];begin=time.perf_counter()
    try:
        cfg=Config(**plan['config']);configs=dict(translation=replace(cfg,shape_enabled=False),deformation=cfg)
        options=read(HERE/'planner_config.json');previous=read(HERE/plan['previous_v2']/'results.json')
        assert previous['finished'] and previous['count']==100
        data=ReferenceLayouts('dev_monitor500');reg=data.registry
        for p in read(HERE/'planner_provenance.json')['files']:
            assert sha(ROOT/p['copy'])==p['copy_sha256']==sha(ROOT/p['source'])
        paths=[*HERE.rglob('*.py'),PLAN,HERE/'planner_config.json',HERE/'planner_provenance.json',
            *[ROOT/p for p in ('datasets/reference_layouts.py','datasets/paths.py','scoring/evaluation.py','scoring/checks.py','scoring/official.py','visualization/layouts.py')]]
        identity={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
        save(out/'config.json',dict(plan=plan,source_sha256=identity,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             members=[x['member'] for x in reg['sets']['dev_monitor500']['records'][:100]],
             registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),paths_sha256=sha(ROOT/'protocol/paths.json'),
             frozen_hashes=reg['frozen_original_hashes'],previous_result_sha256=sha(HERE/plan['previous_v2']/'results.json'),
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
        with threadpool_limits(limits=1):
            import torch,numba
            torch.set_num_threads(1);torch.set_num_interop_threads(1);numba.set_num_threads(1)
            cold=time.perf_counter();warmup();cold=time.perf_counter()-cold
            for index in range(100):
                if (HERE/'runs/STOP').exists() or (out/'STOP').exists():
                    save(out/'state.json',dict(status='stopped',completed=len(records),test_read=False));return
                row=data[index];inputs=row['case'].solver_inputs
                source=ROOT/plan['source_draft'].format(index=index);base=read(source);old=previous['records'][index]
                assert list(row['member'])==base['member']==old['member']
                draft=np.asarray(base['draft_positions'],float);short=draft[:,2:].min(1)
                c=np.asarray(inputs[5]);pre=c[:,1]!=0;shape_locked=c[:,:2].any(1)
                arrays=dict(reference=row['positions'],positions=draft,constraints=c,preplaced=pre,
                            fixed_shape=c[:,0]!=0,baseline_final=np.asarray(base['positions'],float))
                with np.load(HERE/plan['previous_v2']/f'coordinates/{index:04d}.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['positions'],draft)
                    arrays['v2_draft']=saved['relaxed'].copy();arrays['v2_final']=saved['relaxed_final'].copy()
                arm_results={};order=('translation','deformation') if index%2==0 else ('deformation','translation')
                for arm in order:
                    if (HERE/'runs/STOP').exists() or (out/'STOP').exists():
                        save(out/'state.json',dict(status='stopped',completed=len(records),partial_index=index,finished_arms=list(arm_results),test_read=False));return
                    start=time.perf_counter()
                    revised,info=preprocess(draft,inputs,config=configs[arm],return_history=True)
                    history=info.pop('history')
                    final,trace=optimize(revised,inputs,**options)
                    hard=check_final(final,inputs);wall=time.perf_counter()-start
                    centers=history[:,:,:2]+history[:,:,2:]/2
                    moves=np.linalg.norm(np.diff(centers,axis=0),axis=-1)
                    assert (moves<=cfg.step_fraction*short+1e-10).all()
                    assert (moves.sum(0)<.2*short).all()
                    assert np.array_equal(revised[pre],draft[pre]) and np.array_equal(revised[shape_locked,2:],draft[shape_locked,2:])
                    assert np.max(np.abs(revised[:,2:].prod(1)/draft[:,2:].prod(1)-1))<2e-12
                    if arm=='translation':np.testing.assert_array_equal(revised[:,2:],draft[:,2:])
                    score=evaluate_case(row['case'],final,wall)
                    assert score['metrics']['is_feasible'] and score['independent_hard_checks']['is_feasible']
                    arm_results[arm]=dict(relaxation=info,raw=measures(revised),metrics=score['metrics'],hard=hard,
                        seconds=wall,trace=trace,final_geometry=measures(final))
                    arrays[arm+'_draft']=revised;arrays[arm+'_final']=final;arrays[arm+'_history']=history
                    path=out/'coordinates'/f'{index:04d}.npz';tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
                    tmp.replace(path)
                    save(out/'progress.json',dict(completed=len(records),total=100,partial_index=index,
                        partial_arms=arm_results,records=records,test_read=False))
                record=dict(index=index,member=list(row['member']),n=int(inputs[0]),source=source.relative_to(ROOT).as_posix(),
                    source_sha256=sha(source),arm_order=order,raw_before=old['raw_before'],raw_v2=old['raw_after'],
                    baseline_metrics=old['control']['metrics'],v2_metrics=old['mass']['metrics'],**arm_results)
                records.append(record)
                save(out/'progress.json',dict(completed=len(records),total=100,records=records,test_read=False))
                save(out/'state.json',dict(status='evaluating',completed=len(records),total=100,test_read=False))
                if index%10==0 or index==99:print(json.dumps(dict(completed=index+1,
                    translation_Q=record['translation']['metrics']['cost'],deformation_Q=record['deformation']['metrics']['cost'],
                    deformed=record['deformation']['relaxation']['deformed_modules'])),flush=True)
            pools=threadpool_info();assert all(x['num_threads']<=1 for x in pools)
        counts=[r['n'] for r in records]
        baseline=[r['baseline_metrics']['cost'] for r in records];v2=[r['v2_metrics']['cost'] for r in records]
        weighted=lambda q:official_evaluator().compute_total_score(q,counts)
        summaries={}
        for arm in configs:
            q=[r[arm]['metrics']['cost'] for r in records]
            summaries[arm]=dict(weighted_Q=weighted(q),mean_Q=float(np.mean(q)),hard_legal=100,
                Q_vs_baseline=directions(baseline,q),Q_vs_v2=directions(v2,q),
                raw_overlap_vs_input=directions([r['raw_before']['pair_intersection_over_area_sum'] for r in records],[r[arm]['raw']['pair_intersection_over_area_sum'] for r in records]),
                raw_void_vs_input=directions([r['raw_before']['exact_void_fraction'] for r in records],[r[arm]['raw']['exact_void_fraction'] for r in records]),
                mean_raw_overlap=float(np.mean([r[arm]['raw']['pair_intersection_over_area_sum'] for r in records])),
                mean_raw_pass10=float(np.mean([r[arm]['raw']['fraction_strictly_below_10pct'] for r in records])),
                mean_raw_worst_coverage=float(np.mean([r[arm]['raw']['worst_module_coverage'] for r in records])),
                preprocessor_seconds=stats([r[arm]['relaxation']['seconds'] for r in records]),
                cpu_draft_to_final_seconds=stats([r[arm]['seconds'] for r in records]),
                max_center_displacement_fraction=max(r[arm]['relaxation']['max_center_displacement_fraction'] for r in records),
                max_center_path_fraction=max(r[arm]['relaxation']['max_center_path_fraction'] for r in records),
                max_dimension_factor=max(r[arm]['relaxation']['max_dimension_factor'] for r in records),
                max_area_error=max(r[arm]['relaxation']['max_relative_area_error'] for r in records),
                total_deformed_modules=sum(r[arm]['relaxation']['deformed_modules'] for r in records),
                total_free_shape_modules=sum(r[arm]['relaxation']['free_shape_modules'] for r in records))
        t=[r['translation']['metrics']['cost'] for r in records];d=[r['deformation']['metrics']['cost'] for r in records]
        result=dict(status='scored_rendering',finished=False,count=100,baseline_weighted_Q=weighted(baseline),
            previous_v2_weighted_Q=weighted(v2),arms=summaries,shape_final_Q_vs_translation=directions(t,d),
            shape_raw_overlap_vs_translation=directions([r['translation']['raw']['pair_intersection_over_area_sum'] for r in records],[r['deformation']['raw']['pair_intersection_over_area_sum'] for r in records]),
            cold_native_warmup_seconds=cold,threads=pools,timing_scope=plan['timing'],
            records=records,test_read=False,no_training=True)
        save(out/'results.json',result)
        assets=[]
        for index in range(100):
            if (HERE/'runs/STOP').exists() or (out/'STOP').exists():
                save(out/'state.json',dict(status='stopped_rendering',scored=100,rendered=len(assets),test_read=False));return
            with np.load(out/'coordinates'/f'{index:04d}.npz',allow_pickle=False) as saved:
                c=saved['constraints'].tolist()
                variants={name:dict(positions=saved[key].tolist(),soft_details=geometry_details(saved[key],c)) for name,key in
                    (('reference','reference'),('raw','positions'),('deformed','deformation_draft'),('final','deformation_final'))}
            path=out/'gallery'/f'{index:04d}.png'
            plot_case(dict(case_id=f'dev500:{index}',panel='E99 V3 / translation + deformation / fixed100',constraints=c,variants=variants),path,
                stages=(('reference','Reference'),('raw','Saved C3 draft'),('deformed','Deformation / 10 steps'),('final','Deformation + planner')),show_metrics=False)
            assets.append(dict(index=index,image=path.name,sha256=sha(path)))
            if index%20==0:print('RENDER',index+1,'/100',flush=True)
        save(out/'gallery/manifest.json',dict(count=100,assets=assets,common_IDs_and_scale=True,score_controls=False))
        html='<!doctype html><meta charset="utf-8"><title>E99 V3 fixed100</title><style>body{font:16px system-ui}img{width:100%}</style><h1>E99 V3 translation + deformation / fixed100</h1>'
        html+=''.join(f'<p>dev500:{i}</p><img loading="lazy" src="{i:04d}.png">' for i in range(100))
        (out/'gallery/index.html').write_text(html,encoding='utf-8')
        assert all(sha(ROOT/p)==digest for p,digest in identity.items())
        result.update(status='complete',finished=True,gallery_count=100,elapsed_seconds=time.perf_counter()-begin)
        save(out/'results.json',result);save(out/'state.json',dict(status='complete',completed=100,gallery_count=100,test_read=False))
        save(out/'completion_audit.json',dict(status='passed',all_sources_match=True,count=100,final_calls=200,gallery_count=100,
            all_final_hard_checks_passed=True,area_and_fixed_geometry_verified=True,center_and_shape_budgets_verified=True,test_read=False))
        lines=['# E99 V3：扩大移动范围并加入形变', '',
            '同固定100例；10轮、单轮中心移动不超过原短边3%，累计中心路程及位移不超过19.99%（严格小于20%）。软模块形变保持面积：w=w0·exp(q)，h=h0·exp(-q)。每轮尺寸因子≤1.015，累计对数形变路程≤log(1.15)，尺寸在原来的1/1.15～1.15倍内。固定形状尺寸与预放置完整几何锁定。', '',
            '对照为相同移动预算下只平移；新增形变只使用方向性成对受力，不使用MIB/group/边界/外框/线长目标。旧4轮v2与无前置最终坐标只读复用，没有Loss/Q相关性重算或系数扫描。', '',
            '| 指标 | 10轮只平移 | 10轮平移+形变 |','|---|---:|---:|']
        for name,key in (('子集加权Q','weighted_Q'),('草稿平均交面积/总面积','mean_raw_overlap'),('草稿平均覆盖<10%模块比例','mean_raw_pass10'),('草稿每例最坏覆盖均值','mean_raw_worst_coverage'),('最大中心位移/原短边','max_center_displacement_fraction'),('最大尺寸变化因子','max_dimension_factor')):
            lines.append(f"| {name} | {summaries['translation'][key]:.9f} | {summaries['deformation'][key]:.9f} |")
        for arm in configs:
            s=summaries[arm];lines+=['',f"{arm}：草稿重叠改善/持平/恶化={s['raw_overlap_vs_input']}；真实空白率变化={s['raw_void_vs_input']}；最终Q相比原流程={s['Q_vs_baseline']}。",
                f"前置均时/P95={s['preprocessor_seconds']['mean']*1000:.3f}/{s['preprocessor_seconds']['p95']*1000:.3f}ms；CPU草稿到最终硬检查={s['cpu_draft_to_final_seconds']['mean']:.6f}/{s['cpu_draft_to_final_seconds']['p95']:.6f}s。"]
        lines+=['',f"形变相比相同10轮平移的最终Q：{result['shape_final_Q_vs_translation']}；草稿交面积：{result['shape_raw_overlap_vs_translation']}。",
            f"无前置原流程加权Q={result['baseline_weighted_Q']:.9f}；旧4轮v2={result['previous_v2_weighted_Q']:.9f}。",'',
            '这是已曝光开发子集；上述Q不是完整dev500或独立泛化证据。两组最终100/100通过现有官方与独立硬检查的数值容差，不能据此称所有浮点交集严格为零。时间不含网络/输入构造/传输，主机其他任务未隔离，不代表完整请求延迟。', '',
            '| ID | N | 原交面积→只平移→形变 | 形变覆盖<10%比例 | 形变最坏覆盖及ID | 原Q→只平移Q→形变Q |','|---|---:|---:|---:|---:|---:|']
        for r in records:
            a,b,c=r['raw_before'],r['translation']['raw'],r['deformation']['raw']
            lines.append(f"| {r['index']} | {r['n']} | {a['pair_intersection_over_area_sum']:.4%} → {b['pair_intersection_over_area_sum']:.4%} → {c['pair_intersection_over_area_sum']:.4%} | {c['fraction_strictly_below_10pct']:.2%} | {c['worst_module_coverage']:.2%} (ID{c['worst_module']}) | {r['baseline_metrics']['cost']:.6f} → {r['translation']['metrics']['cost']:.6f} → {r['deformation']['metrics']['cost']:.6f} |")
        lines+=['','[同100例四列图](gallery/index.html)；坐标、两组逐轮轨迹和原v2对照均保存在每例唯一NPZ中，未自动打开界面。']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','threads')},indent=2))
    except BaseException as error:
        save(out/'error.json',dict(error=repr(error),completed=len(records),test_read=False,no_automatic_retry=True));raise


if __name__=='__main__':main()
