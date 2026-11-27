"""One configurable fixed100 evaluation, caching, resume, reports and galleries."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse,contextlib,datetime,hashlib,inspect,json,re,subprocess,time,zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from visualization.layouts import plot_case,geometry_details
from .deformation import preprocess
from .diagnostics import measures
from .planner.solver import optimize
from .planner.canonical_native import warmup
from .preflight import check_final
from .tuning_config import load

HERE=Path(__file__).resolve().parent
RUNS=HERE/'runs/tuning'
PROTOCOL=HERE/'tuning_protocol.json'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def numerical_key(identity):
    """Rendering/report changes do not invalidate completed numerical results.

    Full source snapshots remain bound for each actual solve. Cache equivalence
    uses the complete numeric settings plus the geometry/planner/data/scoring
    sources; display, tests and orchestration metadata are audited separately.
    """
    if 'source_sha256' not in identity:return digest(identity)
    value={k:v for k,v in identity.items() if k not in ('source_sha256','render_views')}
    exact={'experiments/E99_attraction_repulsion/deformation.py','experiments/E99_attraction_repulsion/relaxation.py'}
    prefixes=('experiments/E99_attraction_repulsion/planner/','datasets/','scoring/')
    value['source_sha256']={k:v for k,v in identity['source_sha256'].items() if k in exact or k.startswith(prefixes)}
    return digest(value)
def save(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def stats(x):
    a=np.asarray(x,float);return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))
def direction(a,b):
    d=np.asarray(b)-np.asarray(a);return dict(improved=int((d < -1e-12).sum()),unchanged=int((np.abs(d)<=1e-12).sum()),worsened=int((d>1e-12).sum()))


def effective_planner(options):
    params=inspect.signature(optimize).parameters
    out={k:v.default for k,v in params.items() if k not in ('draft','inputs')}
    for k,v in options.items():
        if k not in out:raise ValueError('Unknown planner setting: '+k)
        default=out[k]
        if isinstance(default,bool) and not isinstance(v,bool):raise ValueError(k+' must be boolean')
        if type(default) is int and (type(v) is not int):raise ValueError(k+' must be integer')
        if isinstance(default,(float,int)) and not isinstance(default,bool):
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError(k+' must be finite numeric')
        out[k]=v
    if out['seconds']<=0 or out['iterations']<0 or out['snap']<=0:raise ValueError('Invalid planner budget/tolerance')
    return out


@contextlib.contextmanager
def process_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e:raise RuntimeError('Another tuning run is active') from e
            try:yield
            finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)


def inputs_binding(settings,parameters):
    data=ReferenceLayouts('dev_monitor500');reg=data.registry;protocol=read(PROTOCOL)
    records=reg['sets']['dev_monitor500']['records'][:100]
    originals=[ROOT/protocol['source_draft'].format(index=i) for i in range(100)]
    source_hashes=[sha(p) for p in originals]
    cached=read(HERE/'runs/20261102_deform_v3_dev100/results.json')['records']
    cached_ok=all(cached[i]['source_sha256']==source_hashes[i] and cached[i]['member']==records[i]['member'] for i in range(100))
    manifest=read(HERE/'planner_provenance.json')
    cached_ok=cached_ok and all(sha(ROOT/p['copy'])==p['copy_sha256'] for p in manifest['files'])
    options=effective_planner(settings['planner'])
    cached_ok=cached_ok and options==effective_planner(read(HERE/'planner_config.json'))
    baseline_mode='cached' if settings['comparison']['baseline_mode']=='auto' and cached_ok else 'fresh'
    paths=[*HERE.glob('*.py'),*HERE.joinpath('planner').rglob('*.py'),PROTOCOL,HERE/'planner_provenance.json',
        *[ROOT/p for p in ('datasets/reference_layouts.py','datasets/paths.py','datasets/worksets.py','datasets/cases.py',
            'scoring/evaluation.py','scoring/checks.py','scoring/official.py','visualization/layouts.py')]]
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    semantic=dict(revision=protocol['revision'],relaxation=asdict(parameters),planner=options,
        baseline_mode=baseline_mode,baseline_requested=settings['comparison']['baseline_mode'],
        members=[r['member'] for r in records],registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),
        data_paths_sha256=sha(ROOT/'protocol/paths.json'),source_draft_sha256=source_hashes,source_sha256=sources)
    return data,originals,semantic,paths


def acquire_run(settings,identity,resume=None):
    RUNS.mkdir(parents=True,exist_ok=True)
    key=numerical_key(identity)
    if resume:
        out=Path(resume)
        if not out.is_absolute():out=ROOT/out
        out=out.resolve()
        if not out.is_relative_to(RUNS.resolve()) or not out.is_dir():raise ValueError('Resume must point to an existing runs/tuning directory')
        if numerical_key(read(out/'binding.json')['identity'])!=key:raise ValueError('Resume numerical settings/source differ; use unchanged settings snapshot or start a new run')
        return out,(out/'results.json').exists() and read(out/'results.json').get('finished',False),True
    for binding in sorted(RUNS.glob('*/binding.json')):
        b=read(binding)
        if numerical_key(b['identity'])!=key:continue
        out=binding.parent
        if (out/'results.json').exists() and read(out/'results.json').get('finished'):return out,True,False
        raise RuntimeError('Matching unfinished run exists. Resume explicitly: '+str(out))
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    label=re.sub('[^A-Za-z0-9_-]+','_',settings['label']).strip('_')[:50] or 'run'
    out=(RUNS/f'{stamp}_{label}_{key[:10]}').resolve()
    if not out.is_relative_to(RUNS.resolve()):raise ValueError('Invalid run target')
    out.mkdir(exist_ok=False);(out/'coordinates').mkdir();(out/'cases').mkdir();(out/'gallery').mkdir()
    save(out/'binding.json',dict(identity_sha256=digest(identity),numerical_sha256=key,identity=identity))
    save(out/'settings.json',settings)
    return out,False,False


def verify_case(out,index,source_hash):
    path=out/'cases'/f'{index:04d}.json'
    if not path.exists():return None
    row=read(path);coordinates=out/'coordinates'/f'{index:04d}.npz'
    if row['source_sha256']!=source_hash or sha(coordinates)!=row['coordinates_sha256']:raise ValueError('Completed case binding changed')
    return row


def complete_artifacts_valid(out):
    if not (out/'report.md').is_file():return False
    binding=read(out/'binding.json')['identity']
    for i,source in enumerate(binding['source_draft_sha256']):
        if verify_case(out,i,source) is None:return False
    if (out/'gallery/PRIMARY.json').is_file():
        pointer=read(out/'gallery/PRIMARY.json');manifest_path=(ROOT/pointer['manifest']).resolve()
        if not manifest_path.is_relative_to((out/'gallery').resolve()) or not manifest_path.is_file():return False
        manifest=read(manifest_path)
        return manifest.get('count')==100 and all((manifest_path.parent/x['image']).is_file() and sha(manifest_path.parent/x['image'])==x['sha256'] for x in manifest['assets'])
    if not (out/'gallery/manifest.json').is_file():return False
    manifest=read(out/'gallery/manifest.json')
    return all((out/'gallery'/x['view']/x['image']).is_file() and sha(out/'gallery'/x['view']/x['image'])==x['sha256'] for x in manifest['assets'])


def publish_latest(out,result):
    primary=read(out/'gallery/PRIMARY.json')['gallery'] if (out/'gallery/PRIMARY.json').exists() else (out/'gallery/index.html').relative_to(ROOT).as_posix()
    save(RUNS/'LATEST.json',dict(run=out.relative_to(ROOT).as_posix(),report=(out/'report.md').relative_to(ROOT).as_posix(),
        gallery=primary,status=result['status'],finished=result.get('finished',False)))
    save(HERE/'state.json',dict(status='tuning_'+result['status'],active_run=None if result.get('finished') else out.relative_to(ROOT).as_posix(),
        run=out.relative_to(ROOT).as_posix(),settings_snapshot=(out/'settings.json').relative_to(ROOT).as_posix(),
        original_weighted_Q=result['original']['weighted_Q'],hybrid_weighted_Q=result['hybrid']['weighted_Q'],
        test_read=False,training_started=False,formal_dev500_evaluated=False))
    rows=[]
    for path in sorted(RUNS.glob('*/results.json')):
        r=read(path)
        if not r.get('finished'):continue
        s=read(path.parent/'settings.json');p=s['relaxation']
        rows.append(f"| [{s['label']}]({path.parent.name}/report.md) | {p['iterations']} | {p['total_fraction']:.1%} | {1+p['shape_total_fraction']:.3f} | {r['original']['weighted_Q']:.9f} | {r['hybrid']['weighted_Q']:.9f} | {r['hybrid']['hard_legal']}/100 | {r['hybrid']['preprocessor_seconds']['mean']*1000:.3f} |")
    (RUNS/'index.md').write_text('# E99 手动配置运行记录\n\n按时间列出，不按Q选best。不同源码或后处理配置不可直接当作单因素比较。\n\n| 运行 | 轮数 | 位移总预算 | 尺寸因子 | 原Q | 新Q | 硬检查 | 前置ms |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n',encoding='utf-8')


def summarize(records,mode):
    weights=np.exp((np.array([r['n'] for r in records])-120)/12);weights/=weights.sum()
    result={}
    for arm in ('original','hybrid'):
        rows=[r[arm] for r in records];q=np.array([r['metrics']['cost'] for r in rows])
        result[arm]=dict(weighted_Q=float(weights@q),mean_Q=float(q.mean()),p95_Q=float(np.percentile(q,95)),max_Q=float(q.max()),
            hard_legal=sum(r['metrics']['is_feasible'] and r['hard']['is_feasible'] for r in rows),
            mean_wire=float(np.mean([r['metrics']['hpwl_total'] for r in rows])),mean_bbox=float(np.mean([r['metrics']['bbox_area'] for r in rows])),
            mean_soft_violations=float(np.mean([r['metrics']['total_soft_violations'] for r in rows])),
            seconds=stats([r['seconds'] for r in rows]),timing_mode='fresh_CPU_draft_to_final_hard_check' if arm=='hybrid' or mode=='fresh' else 'cached_historical_repair_only_unpaired')
    result['hybrid']['preprocessor_seconds']=stats([r['relaxation']['seconds'] for r in records])
    a=np.array([r['original']['metrics']['cost'] for r in records]);b=np.array([r['hybrid']['metrics']['cost'] for r in records]);delta=b-a
    contribution=weights*delta;best=int(np.argmin(contribution));keep=np.arange(len(records))!=best
    result.update(Q_changes=direction(a,b),weighted_Q_change=float(weights@delta),
        Q_influence=dict(index=best,contribution=float(contribution[best]),weighted_Q_change_without_this_case=float(weights[keep]@delta[keep]/weights[keep].sum())),
        raw_overlap_changes=direction([r['raw_before']['pair_intersection_over_area_sum'] for r in records],[r['raw_after']['pair_intersection_over_area_sum'] for r in records]),
        raw_void_changes=direction([r['raw_before']['exact_void_fraction'] for r in records],[r['raw_after']['exact_void_fraction'] for r in records]),
        mean_raw_overlap_before=float(np.mean([r['raw_before']['pair_intersection_over_area_sum'] for r in records])),
        mean_raw_overlap_after=float(np.mean([r['raw_after']['pair_intersection_over_area_sum'] for r in records])),
        mean_pass10_before=float(np.mean([r['raw_before']['fraction_strictly_below_10pct'] for r in records])),
        mean_pass10_after=float(np.mean([r['raw_after']['fraction_strictly_below_10pct'] for r in records])),
        mean_worst_coverage_before=float(np.mean([r['raw_before']['worst_module_coverage'] for r in records])),
        mean_worst_coverage_after=float(np.mean([r['raw_after']['worst_module_coverage'] for r in records])),
        max_center_path_fraction=max(r['relaxation']['max_center_path_fraction'] for r in records),
        max_center_displacement_fraction=max(r['relaxation']['max_center_displacement_fraction'] for r in records),
        max_dimension_factor=max(r['relaxation']['max_dimension_factor'] for r in records),
        max_area_error=max(r['relaxation']['max_relative_area_error'] for r in records))
    return result


def render(out,settings,records,halt):
    from .tuning_gallery import render_four
    return render_four(out,settings,records,halt)


def write_report(out,settings,result,records):
    p=settings['relaxation'];a=result['original'];b=result['hybrid'];d=result['Q_changes']
    lines=['# E99 配置运行：'+settings['label'],'',
        f"固定100例；{p['iterations']}轮，中心单步短边{p['step_fraction']:.1%}、累计路程/位移{p['total_fraction']:.1%}；尺寸单步因子{1+p['shape_step_fraction']:.3f}、累计因子{1+p['shape_total_fraction']:.3f}。保持面积、固定形状与预放置；来源C3草稿、公开面积变换、split与索引均不变。",'',
        '| 指标 | 原后处理 | 吸引排斥＋相同后处理 |','|---|---:|---:|',
        f"| 加权Q | {a['weighted_Q']:.9f} | {b['weighted_Q']:.9f} |",f"| 算术平均Q | {a['mean_Q']:.9f} | {b['mean_Q']:.9f} |",
        f"| Q P95 / 最大值 | {a['p95_Q']:.6f} / {a['max_Q']:.6f} | {b['p95_Q']:.6f} / {b['max_Q']:.6f} |",
        f"| 官方及独立硬检查 | {a['hard_legal']}/100 | {b['hard_legal']}/100 |",f"| 平均软约束违例 | {a['mean_soft_violations']:.3f} | {b['mean_soft_violations']:.3f} |",'',
        f"最终Q改善{d['improved']}例、持平{d['unchanged']}例、恶化{d['worsened']}例。",'',
        f"草稿平均交面积/总面积：{result['mean_raw_overlap_before']:.5%} → {result['mean_raw_overlap_after']:.5%}；变化={result['raw_overlap_changes']}。真实空白率变化={result['raw_void_changes']}。",'',
        f"覆盖<10%模块比例均值：{result['mean_pass10_before']:.3%} → {result['mean_pass10_after']:.3%}；每例最坏覆盖均值：{result['mean_worst_coverage_before']:.3%} → {result['mean_worst_coverage_after']:.3%}。",'',
        f"前置操作均时/P95={b['preprocessor_seconds']['mean']*1000:.3f}/{b['preprocessor_seconds']['p95']*1000:.3f}ms。CPU草稿到新流程最终硬检查均时/P95={b['seconds']['mean']:.6f}/{b['seconds']['p95']:.6f}s。",'',
        f"原后处理时间口径：{a['timing_mode']}，均时/P95={a['seconds']['mean']:.6f}/{a['seconds']['p95']:.6f}s。历史缓存时间不是同期完整阶段测速，不能计算配对提速。所有时间不含NN/公开输入构造/传输；未隔离其他用户任务，不代表完整请求延迟。",'',
        f"加权变化贡献最大改善来自ID{result['Q_influence']['index']}；去掉该例并重归一化后的Q差={result['Q_influence']['weighted_Q_change_without_this_case']:+.9f}（事后影响诊断，不据此删例）。",'',
        'Q为原官方评分的中性时间开发结果；本面板已多次曝光，手动调参具有自适应开发性质，不代表正式比赛或独立泛化。硬检查按既有数值容差，不宣称所有浮点交集严格为零。无训练、封存测试未读、无自动扫描或唤醒。', '',
        '[同100例图片入口](gallery/index.html) · [配置快照](settings.json) · [逐例统计](results.json) · [源码/成员身份](binding.json)', '',
        '| ID | N | 草稿交面积前→后 | 覆盖<10%前→后 | 最坏覆盖前→后 | 原Q→新Q |','|---|---:|---:|---:|---:|---:|']
    for r in records:
        x,y=r['raw_before'],r['raw_after']
        lines.append(f"| {r['index']} | {r['n']} | {x['pair_intersection_over_area_sum']:.4%} → {y['pair_intersection_over_area_sum']:.4%} | {x['fraction_strictly_below_10pct']:.2%} → {y['fraction_strictly_below_10pct']:.2%} | {x['worst_module_coverage']:.2%} (ID{x['worst_module']}) → {y['worst_module_coverage']:.2%} (ID{y['worst_module']}) | {r['original']['metrics']['cost']:.6f} → {r['hybrid']['metrics']['cost']:.6f} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute(settings_path,resume=None):
    settings,parameters=load(settings_path)
    effective_planner(settings['planner'])
    with process_lock(RUNS/'.tuning.lock'):
        data,originals,identity,paths=inputs_binding(settings,parameters)
        out,cached,resuming=acquire_run(settings,identity,resume)
        print('RUN',out,flush=True)
        if cached:
            if complete_artifacts_valid(out):
                result=read(out/'results.json')
                display=read(out/'settings.json')
                display['comparison']={**display.get('comparison',{}),'previous_best_run':settings['comparison']['previous_best_run']}
                from .tuning_gallery import render_four
                if not render_four(out,display,result['records'],lambda:(out/'STOP').exists()):return out
                publish_latest(out,result)
                print('Identical completed numerical run reused; no new solves.\nREPORT '+str(out/'report.md'),flush=True);return out
            if not resuming:raise RuntimeError('Some artifacts are missing; restore with explicit --resume '+str(out))
        # Snapshot is the only authority from this point; edits to the live file
        # during execution never change this run's settings or identity.
        settings=read(out/'settings.json');settings_path=Path(settings_path)
        if not resuming:
            save(out/'registration.json',dict(protocol=read(PROTOCOL),identity_sha256=digest(identity),
                input_settings_sha256=sha(settings_path),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False))
            with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
                z.write(out/'settings.json','settings.json')
        halt=lambda:(HERE/'runs/STOP').exists() or (out/'STOP').exists()
        begin=time.perf_counter();records=[]
        prior=read(out/'state.json').get('elapsed_seconds',0.) if resuming and (out/'state.json').exists() else 0.
        def state(status,**extra):
            value=dict(status=status,completed=len(records),total=100,run=out.relative_to(ROOT).as_posix(),
                elapsed_seconds=prior+time.perf_counter()-begin,test_read=False,**extra)
            save(out/'state.json',value);save(RUNS/'LATEST.json',value)
            save(HERE/'state.json',dict(status='tuning_'+status,active_run=None if status in ('complete','stopped','stopped_rendering','error') else out.relative_to(ROOT).as_posix(),
                 run=out.relative_to(ROOT).as_posix(),completed=len(records),settings=asdict(parameters),test_read=False,training_started=False))
        try:
            with threadpool_limits(limits=1):
                import torch,numba
                torch.set_num_threads(1)
                if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
                numba.set_num_threads(1)
                start=time.perf_counter();warmup();cold=time.perf_counter()-start
                for index in range(100):
                    completed=verify_case(out,index,identity['source_draft_sha256'][index])
                    if completed is not None:records.append(completed);continue
                    if halt():state('stopped');return out
                    row=data[index];inputs=row['case'].solver_inputs;source=read(originals[index])
                    if source['member']!=list(row['member']):raise ValueError('Frozen member mismatch')
                    if sha(originals[index])!=identity['source_draft_sha256'][index]:raise ValueError('Source draft changed during run')
                    draft=np.asarray(source['draft_positions'],float);c=np.asarray(inputs[5]);short=draft[:,2:].min(1)
                    arrays=dict(reference=row['positions'],positions=draft,constraints=c,preplaced=c[:,1]!=0,fixed_shape=c[:,0]!=0)
                    path=out/'coordinates'/f'{index:04d}.npz'
                    partial_path=out/'cases'/f'{index:04d}.partial.json'
                    partial={}
                    if partial_path.exists():
                        partial=read(partial_path)
                        if partial['source_sha256']!=identity['source_draft_sha256'][index] or sha(path)!=partial['coordinates_sha256']:
                            raise ValueError('Partial case binding changed')
                        with np.load(path,allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
                    def retain_partial():
                        temp=path.with_suffix('.tmp')
                        with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
                        temp.replace(path)
                        partial['source_sha256']=identity['source_draft_sha256'][index]
                        partial['coordinates_sha256']=sha(path);save(partial_path,partial)
                    def hybrid():
                        if 'hybrid' in partial:return partial['hybrid'],partial['relaxation'],partial['raw_after']
                        start=time.perf_counter();revised,info=preprocess(draft,inputs,config=parameters,return_history=True)
                        history=info.pop('history');final,trace=optimize(revised,inputs,**identity['planner']);hard=check_final(final,inputs);wall=time.perf_counter()-start
                        centers=history[:,:,:2]+history[:,:,2:]/2;move=np.linalg.norm(np.diff(centers,axis=0),axis=-1)
                        if (move>parameters.step_fraction*short+1e-10).any() or (move.sum(0)>parameters.total_fraction*short+1e-10).any():raise AssertionError('movement audit failed')
                        arrays.update(relaxed=revised,history=history,hybrid_final=final)
                        result=dict(metrics=evaluate_case(row['case'],final,wall)['metrics'],hard=hard,seconds=wall,trace=trace)
                        partial.update(hybrid=result,relaxation=info,raw_after=measures(revised));retain_partial()
                        return result,info,partial['raw_after']
                    def original():
                        if 'original' in partial:return partial['original']
                        if identity['baseline_mode']=='cached':
                            final=np.asarray(source['positions'],float);hard=check_final(final,inputs)
                            result=dict(metrics=evaluate_case(row['case'],final,source['repair_seconds'])['metrics'],hard=hard,
                                seconds=source['repair_seconds'],reused=True,trace=None)
                        else:
                            start=time.perf_counter();final,trace=optimize(draft,inputs,**identity['planner']);hard=check_final(final,inputs);wall=time.perf_counter()-start
                            result=dict(metrics=evaluate_case(row['case'],final,wall)['metrics'],hard=hard,seconds=wall,reused=False,trace=trace)
                        arrays['original_final']=final;partial['original']=result;retain_partial();return result
                    if identity['baseline_mode']=='fresh' and index%2==0:
                        base=original()
                        if halt():state('stopped',partial_case=index);return out
                        new,info,raw=hybrid()
                    else:
                        new,info,raw=hybrid()
                        if halt():state('stopped',partial_case=index);return out
                        base=original()
                    record=dict(index=index,member=list(row['member']),n=int(inputs[0]),source_sha256=identity['source_draft_sha256'][index],
                        coordinates_sha256=sha(path),original=base,hybrid=new,relaxation=info,raw_before=measures(draft),raw_after=raw)
                    save(out/'cases'/f'{index:04d}.json',record);records.append(record);state('evaluating')
                    if index%10==0 or index==99:print(json.dumps(dict(completed=index+1,Q_original=base['metrics']['cost'],Q_new=new['metrics']['cost'])),flush=True)
                pools=threadpool_info()
                if not all(p['num_threads']<=1 for p in pools):raise AssertionError('thread limit not retained')
            result=dict(status='rendering',finished=False,count=100,identity_sha256=digest(identity),
                **summarize(records,identity['baseline_mode']),records=records,cold_native_warmup_seconds=cold,threads=pools,
                baseline_mode=identity['baseline_mode'],test_read=False,no_training=True,full_request_timing=False)
            save(out/'results.json',result);state('rendering')
            if not render(out,settings,records,halt):state('stopped_rendering');return out
            if not all(sha(ROOT/p)==v for p,v in identity['source_sha256'].items()):raise ValueError('Bound source changed while run active')
            result.update(status='complete',finished=True,elapsed_seconds=prior+time.perf_counter()-begin,gallery_cases=100,
                          gallery_views=settings['render_views'])
            save(out/'results.json',result);write_report(out,settings,result,records)
            save(out/'completion_audit.json',dict(status='passed',count=100,source_binding_verified=True,completed_cases_hashed=True,
                area_and_lock_audits=True,movement_and_shape_audits=True,test_read=False))
            state('complete');publish_latest(out,result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('records','threads')},indent=2));print('REPORT',out/'report.md',flush=True)
            return out
        except BaseException as e:
            save(out/'error.json',dict(error=repr(e),completed=len(records),test_read=False,no_automatic_retry=True));state('error');raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=HERE/'tuning.json');p.add_argument('--resume',type=Path)
    args=p.parse_args();execute(args.config,args.resume)


if __name__=='__main__':main()
