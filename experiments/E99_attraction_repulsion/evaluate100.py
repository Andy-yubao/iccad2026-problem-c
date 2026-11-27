"""Registered area-mass check on fixed100; reuse completed control geometry."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from dataclasses import replace
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case, geometry_details
from .relaxation import Config
from .pipeline import preprocess
from .diagnostics import measures
from .planner.solver import optimize
from .planner.canonical_native import warmup
from .preflight import check_final

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE/'plan_mass_v2.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))


def main():
    plan=read(PLAN_PATH);output=HERE/plan['output']
    if output.exists():
        raise FileExistsError('Existing evidence protected: '+str(output))
    output.mkdir(parents=True);(output/'coordinates').mkdir();(output/'gallery').mkdir()
    begin=time.perf_counter();records=[]
    try:
        config=Config(**plan['config']);control_config=replace(config,mass_weighted=False)
        planner_config=read(HERE/'planner_config.json')
        data=ReferenceLayouts('dev_monitor500');reg=data.registry
        for item in read(HERE/'planner_provenance.json')['files']:
            assert sha(ROOT/item['copy'])==item['copy_sha256']==sha(ROOT/item['source'])
        paths=[*HERE.rglob('*.py'),PLAN_PATH,HERE/'planner_config.json',HERE/'planner_provenance.json',
               *[ROOT/p for p in ('datasets/paths.py','datasets/reference_layouts.py','datasets/worksets.py','datasets/cases.py','scoring/evaluation.py','scoring/checks.py','scoring/official.py','visualization/layouts.py')]]
        identity={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
        save(output/'config.json',dict(plan=plan,source_sha256=identity,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),
             paths_sha256=sha(ROOT/'protocol/paths.json'),frozen_hashes=reg['frozen_original_hashes'],
             members=[x['member'] for x in reg['sets']['dev_monitor500']['records'][:100]],
             created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False))
        with zipfile.ZipFile(output/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(paths)):archive.write(path,path.relative_to(ROOT).as_posix())
        with threadpool_limits(limits=1):
            import torch,numba
            torch.set_num_threads(1);torch.set_num_interop_threads(1);numba.set_num_threads(1)
            start=time.perf_counter();warmup();cold=time.perf_counter()-start
            for index in range(100):
                if (HERE/'runs/STOP').exists() or (output/'STOP').exists():
                    save(output/'state.json',dict(status='stopped',completed=len(records),test_read=False))
                    return
                row=data[index];case=row['case'];inputs=case.solver_inputs
                source=ROOT/plan['source_draft'].format(index=index);saved=read(source)
                assert saved['member']==list(row['member']) and saved['n']==case.block_count
                draft=np.asarray(saved[plan['source_field']],dtype=float)
                # Labels remain in row for scoring/drawing only. Solver arguments
                # are exclusively the public seven-input tuple and saved draft.
                fixed=np.asarray(inputs[5])[:,:2].any(1);specs=np.asarray(inputs[6])
                assert (specs[~fixed]==-1).all()
                v1,v1info=preprocess(draft,inputs,config=control_config)
                if index<5:
                    with np.load(HERE/f'runs/20261102_translation_v1_check5_attempt02/coordinates/{index:04d}.npz') as old:
                        np.testing.assert_array_equal(v1,old['relaxed'])
                # Include trajectory retention in the measured preprocessor cost.
                start=time.perf_counter()
                revised,info=preprocess(draft,inputs,config=config,return_history=True)
                history=info.pop('history')
                final,trace=optimize(revised,inputs,**planner_config)
                hard=check_final(final,inputs)
                seconds=time.perf_counter()-start
                local=draft[:,2:].min(1)
                steps=np.linalg.norm(np.diff(history[:,:,:2],axis=0),axis=-1)
                assert (steps<=.02*local+1e-10).all() and (steps.sum(0)<=.06*local+1e-10).all()
                locked=np.asarray(inputs[5])[:,1]!=0
                assert np.array_equal(revised[:,2:],draft[:,2:]) and np.array_equal(revised[locked],draft[locked])
                baseline=np.asarray(saved['positions'],dtype=float)
                baseline_score=evaluate_case(case,baseline,0.)
                score=evaluate_case(case,final,seconds)
                assert score['independent_hard_checks']['is_feasible'] and baseline_score['independent_hard_checks']['is_feasible']
                np.savez_compressed(output/'coordinates'/f'{index:04d}.npz',reference=row['positions'],
                    positions=draft,v1_no_mass=v1,relaxed=revised,history=history,preplaced=locked,
                    constraints=np.asarray(inputs[5]),control_final=baseline,relaxed_final=final)
                record=dict(index=index,member=list(row['member']),n=case.block_count,
                    source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),
                    raw_before=measures(draft),raw_v1_no_mass=measures(v1),raw_after=measures(revised),
                    relaxation=info,control=dict(reused=True,metrics=baseline_score['metrics'],
                        legacy_Q_abs_error=abs(baseline_score['metrics']['cost']-saved['metrics']['cost']),
                        historical_repair_seconds=saved['repair_seconds']),
                    mass=dict(metrics=score['metrics'],hard=hard,seconds=seconds,trace=trace))
                records.append(record)
                save(output/'progress.json',dict(completed=len(records),total=100,records=records,test_read=False))
                save(output/'state.json',dict(status='evaluating',completed=len(records),total=100,test_read=False))
                if index%10==0 or index==99:
                    print(json.dumps(dict(completed=index+1,control_Q=record['control']['metrics']['cost'],
                        mass_Q=record['mass']['metrics']['cost'],max_displacement=info['max_displacement_fraction'])),flush=True)
            pools=threadpool_info();assert all(p['num_threads']<=1 for p in pools)
        q=lambda arm:official_evaluator().compute_total_score([r[arm]['metrics']['cost'] for r in records],[r['n'] for r in records])
        result=dict(status='scored_rendering',finished=False,count=100,test_read=False,no_training=True,
            panel='v5 fixed dev_monitor500 first100; adaptive development subset, not full dev500',
            baseline_weighted_Q=q('control'),mass_weighted_Q=q('mass'),
            final_improved_cases=sum(r['mass']['metrics']['cost']<r['control']['metrics']['cost']-1e-12 for r in records),
            final_worsened_cases=sum(r['mass']['metrics']['cost']>r['control']['metrics']['cost']+1e-12 for r in records),
            final_hard_legal=100,control_hard_legal=100,
            raw_overlap_improved_cases=sum(r['raw_after']['pair_intersection_over_area_sum']<r['raw_before']['pair_intersection_over_area_sum']-1e-12 for r in records),
            raw_overlap_worsened_cases=sum(r['raw_after']['pair_intersection_over_area_sum']>r['raw_before']['pair_intersection_over_area_sum']+1e-12 for r in records),
            raw_void_improved_cases=sum(r['raw_after']['exact_void_fraction']<r['raw_before']['exact_void_fraction']-1e-12 for r in records),
            raw_mass_vs_v1_improved_cases=sum(r['raw_after']['pair_intersection_over_area_sum']<r['raw_v1_no_mass']['pair_intersection_over_area_sum']-1e-12 for r in records),
            max_displacement_fraction=max(r['relaxation']['max_displacement_fraction'] for r in records),
            max_path_fraction=max(r['relaxation']['max_path_fraction'] for r in records),
            preprocessor_seconds=stats([r['relaxation']['seconds'] for r in records]),
            cpu_draft_to_final_seconds=stats([r['mass']['seconds'] for r in records]),
            timing_scope=plan['timing'],cold_native_warmup_seconds=cold,threads=pools,
            legacy_Q_max_abs_error=max(r['control']['legacy_Q_abs_error'] for r in records),
            records=records)
        save(output/'results.json',result)
        assets=[]
        for index in range(100):
            if (HERE/'runs/STOP').exists() or (output/'STOP').exists():
                save(output/'state.json',dict(status='stopped_rendering',scored=100,rendered=len(assets),test_read=False));return
            with np.load(output/'coordinates'/f'{index:04d}.npz') as stored:
                constraints=stored['constraints'].tolist()
                variants={name:dict(positions=stored[key].tolist(),soft_details=geometry_details(stored[key],constraints))
                    for name,key in (('reference','reference'),('raw','positions'),('corrected','relaxed'),('final','relaxed_final'))}
            path=output/'gallery'/f'{index:04d}.png'
            plot_case(dict(case_id=f'dev500:{index}',panel='E99 area mass / fixed100 development check',constraints=constraints,variants=variants),path,
                stages=(('reference','Reference'),('raw','Saved C3 draft'),('corrected','Area mass / 4 steps'),('final','Area mass + planner')),show_metrics=False)
            assets.append(dict(index=index,image=path.name,sha256=sha(path)))
            if index%20==0:print('RENDER',index+1,'/100',flush=True)
        save(output/'gallery/manifest.json',dict(count=100,assets=assets,score_controls=False,common_IDs_and_scale=True))
        html='<!doctype html><meta charset="utf-8"><title>E99 area mass fixed100</title><style>body{font:16px system-ui}img{width:100%}</style><h1>E99 area mass / fixed100 development check</h1>'
        html+=''.join(f'<p>dev500:{i}</p><img loading="lazy" src="{i:04d}.png">' for i in range(100))
        (output/'gallery/index.html').write_text(html,encoding='utf-8')
        assert all(sha(ROOT/p)==digest for p,digest in identity.items())
        result.update(status='complete',finished=True,gallery_count=100,elapsed_seconds=time.perf_counter()-begin)
        save(output/'results.json',result)
        save(output/'state.json',dict(status='complete',completed=100,total=100,gallery_count=100,test_read=False))
        save(output/'completion_audit.json',dict(status='passed',all_sources_match=True,source_files=len(identity),
             count=100,gallery_count=100,all_hard_legal=True,movement_and_immutable_geometry_passed=True,
             v1_first5_exact_reproduction=True,no_new_network_inference=True,test_read=False))
        lines=['# E99 面积质量版本：固定100例', '',
            '质量=模块初始面积/中位面积；位移响应与质量成反比，统一限幅保留相对响应。默认4轮，单步短边2%、累计路程与位移6%上限不变。', '',
            '使用固定v5 dev_monitor500前100例、保存C3草稿；这是已自适应曝光的开发子集，不是完整dev500或独立泛化。原split、索引及任务变换不变。没有训练、测试访问或系数扫描。', '',
            f"| 指标 | 结果 |",'|---|---:|',f"| 原流程加权Q（复用已存最终坐标） | {result['baseline_weighted_Q']:.9f} |",
            f"| 面积质量前置+相同规划器加权Q | {result['mass_weighted_Q']:.9f} |",
            f"| 最终Q改善/恶化/持平 | {result['final_improved_cases']} / {result['final_worsened_cases']} / {100-result['final_improved_cases']-result['final_worsened_cases']} |",
            f"| 两组最终硬合法 | 100 / 100 |",f"| 草稿交面积改善/恶化 | {result['raw_overlap_improved_cases']} / {result['raw_overlap_worsened_cases']} |",
            f"| 草稿真实空白率改善 | {result['raw_void_improved_cases']} / 100 |",
            f"| 相比v1草稿交面积更小 | {result['raw_mass_vs_v1_improved_cases']} / 100 |",
            f"| 前置均时 / P95 | {result['preprocessor_seconds']['mean']*1000:.3f} / {result['preprocessor_seconds']['p95']*1000:.3f} ms |",
            f"| CPU草稿到最终返回及硬检查均时 / P95 | {result['cpu_draft_to_final_seconds']['mean']:.6f} / {result['cpu_draft_to_final_seconds']['p95']:.6f} s |",
            f"| 最大位移 / 短边 | {result['max_displacement_fraction']:.6%} |", '',
            'Q由原官方评分函数计算，中性时间处理；参考标签和分母只用于评分/绘图。控制复用已完成坐标，未重复求解；其历史时间不构成配对提速证据。以上时间不含网络、公开输入构造和传输，主机其他任务未被暂停隔离，不代表完整请求延迟。', '',
            'v1对照是旧单位质量+逐模块限幅，面积质量版本同时采用统一限幅。因此两者比较包含响应和限幅方式的共同改变，不能归因于质量一项。', '',
            '| ID | N | 交面积前→后 | 覆盖<10%模块前→后 | 最坏覆盖前→后 | 原Q→新Q |', '|---|---:|---:|---:|---:|---:|']
        for r in records:
            a,b=r['raw_before'],r['raw_after']
            lines.append(f"| {r['index']} | {r['n']} | {a['pair_intersection_over_area_sum']:.4%} → {b['pair_intersection_over_area_sum']:.4%} | {a['fraction_strictly_below_10pct']:.2%} → {b['fraction_strictly_below_10pct']:.2%} | {a['worst_module_coverage']:.2%} (ID{a['worst_module']}) → {b['worst_module_coverage']:.2%} (ID{b['worst_module']}) | {r['control']['metrics']['cost']:.6f} → {r['mass']['metrics']['cost']:.6f} |")
        lines+=['','[同100例四列图](gallery/index.html)。未自动打开界面。完整坐标只在coordinates保存，源码与数据身份见config/source_snapshot。']
        (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','threads')},indent=2))
    except BaseException as error:
        save(output/'error.json',dict(error=repr(error),completed=len(records),no_automatic_retry=True,test_read=False))
        raise


if __name__=='__main__':
    main()
