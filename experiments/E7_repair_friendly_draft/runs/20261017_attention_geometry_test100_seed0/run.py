"""User-authorized frozen test100, followed by ten prespecified layout figures.

--prepare and --check read metadata/checkpoints/saved dev output only.
The long test evaluation is started by the user with no mode flag.
"""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import signal
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from datasets.worksets import LayoutReader,workset_manifest,member_pairs
from datasets.cases import _build_case
from experiments.repair_friendly_draft.records import save,append,sha,provenance,snapshot,source_hashes,replace
from experiments.repair_friendly_draft.evaluate import load_model,summarize
from experiments.repair_friendly_draft.inference import prepare_inference
from experiments.repair_friendly_draft.batches import decode
from experiments.repair_friendly_draft.occupancy import exact_diagnostics
from experiments.repair_friendly_draft.solver import optimize
from scoring.evaluation import evaluate_case

OUT=Path(__file__).resolve().parent
HERE=ROOT/'experiments/repair_friendly_draft'
NAME='attention_geometry'
CHECKPOINT=HERE/'runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'


def identity(member):return f"local_test:{member['shard']}:{member['sample_index']}"


def runtime_hashes():
    values=source_hashes()
    values.update({p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/"visualization").glob("*.py"))})
    values[Path(__file__).resolve().relative_to(ROOT).as_posix()]=sha(__file__)
    return values


def expected_plan():
    m=workset_manifest()
    rank=lambda value:hashlib.sha256(('E7:20261017:'+value).encode('utf-8')).hexdigest()
    pool=member_pairs(m,'local_test')
    by_worker={w:sorted((p for p in pool if p[0].startswith(f'worker_{w}/')),
                       key=lambda p:rank(f'{p[0]}:{p[1]}')) for w in range(40,100)}
    extra_workers=sorted(by_worker,key=lambda w:rank(f'test100_extra_worker:{w}'))[:40]
    pairs=sorted([by_worker[w][0] for w in by_worker]+[by_worker[w][1] for w in extra_workers],
                 key=lambda p:(int(p[0].split('/')[0][7:]),p[0],p[1]))
    members=[{'shard':s,'sample_index':r} for s,r in pairs]
    gallery_workers=sorted(sorted(by_worker,key=lambda w:rank(f'test100_gallery_worker:{w}'))[:10])
    gallery=[{'shard':by_worker[w][0][0],'sample_index':by_worker[w][0][1]} for w in gallery_workers]
    old50=json.loads((HERE/'panels.json').read_text(encoding='utf-8'))['test50']
    train_shards={s for s,r in member_pairs(m,'train')}
    assert len(set(pairs))==100 and len(gallery)==10
    assert not train_shards&{s for s,r in pairs}
    assert {identity(p) for p in old50}<={identity(p) for p in members}
    assert {identity(p) for p in gallery}<={identity(p) for p in members}
    control=json.loads((HERE/'runs/20261017_attention_geometry_dev10_seed0/config.json').read_text(encoding='utf-8'))
    if sha(CHECKPOINT)!=control['checkpoint_sha256']:raise ValueError('Retained best differs from the frozen dev10 model')
    return {'version':1,'panel_seed':20261017,'model_seed':0,'model':{'architecture':'attention','width':96,'layers':4},
        'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha(CHECKPOINT),'best_step':48125,'best_epoch':77,
        'workset_sha256':m['manifest_sha256'],'split_sha256':sha(ROOT/'protocol/split_v1.json'),
        'members':members,'gallery_members':gallery,'expected_cases':100,'expected_figures':10,
        'previously_exposed_workers_excluded':list(range(40)),
        'worker_coverage':{'workers':60,'workers_with_two_cases':40,'workers_with_one_case':20},
        'selection':'Within each unexposed worker40..99 rank original v2 local_test members by fixed metadata hash; take one per worker and a second in40 metadata-ranked workers. Gallery takes first member in10 independently metadata-ranked distinct workers.',
        'includes_unread_preselected_test50':True,'source_sha256':runtime_hashes(),
        'postprocess':{'seconds':120.,'shape_steps':24,'reweighted':True,'mip_seconds':0.},
        'threads':2,'device':'cuda','primary_metric':'arithmetic mean neutral-runtime Q, all100cases including errors',
        'scope':'Local held-out100subset of frozen workset_v2;60workers, not the official hidden set or100worker panel',
        'time_policy':'Potentially exceeds20min; user launches the complete invocation. No assistant-run partial test batches.'}


def read_plan():
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    if plan!=expected_plan():raise ValueError('Frozen plan, source or checkpoint changed; do not mix evaluations')
    return plan


def read_rows(repair=False):
    path=OUT/'cases.jsonl'
    if not path.exists():return []
    lines=path.read_bytes().splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):
            if not repair or i!=len(lines)-1:raise
            tmp=OUT/'cases.recovery.tmp';tmp.write_bytes(b''.join(lines[:i]));replace(tmp,path)
            append(OUT/'recovery.jsonl',{'created_utc':datetime.now(timezone.utc).isoformat(),
                'action':'removed only the last incomplete uncommitted JSONL line; completed cases preserved'})
    return rows


def plot_case(row,path):
    from visualization.layouts import plot_evaluation_case
    return plot_evaluation_case(row,NAME,path)


def report_and_gallery(plan,rows):
    if len(rows)!=100:raise ValueError('Complete100cases before producing the final gallery')
    by_id={r['case_id']:r for r in rows};stats=summarize(rows,[NAME])['variants'][NAME]
    figures=OUT/'figures';figures.mkdir(exist_ok=True);items=[]
    for i,m in enumerate(plan['gallery_members'],1):
        row=by_id[identity(m)];filename=f'{i:02d}_worker_{m["shard"].split("/")[0][7:]}.png'
        plot_case(row,figures/filename)
        items.append({'case_id':row['case_id'],'path':'figures/'+filename,'Q':row['variants'][NAME]['final']['metrics']['cost']})
    save(OUT/'gallery.json',{'selection':'prespecified metadata-only; no selection by Q','figures':items})
    header=f'<h1>attention_geometry: local test100</h1><p>Mean Q={stats["Q_mean"]:.6f}; feasible={stats["feasible"]}/100. Ten cases selected before scoring.</p>'
    body=''.join(f'<section><h2>{html.escape(x["case_id"])} / Q={x["Q"]:.4f}</h2><a href="{x["path"]}"><img loading="lazy" style="max-width:100%" src="{x["path"]}"></a></section>' for x in items)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>test100 gallery10</title><style>body{font-family:Arial,sans-serif;margin:24px auto;max-width:1500px}h2{font-size:18px}section{margin:32px 0}</style>'+header+body,encoding='utf-8')
    lines=['# attention_geometry：冻结100例本地测试','',
        '用户于E7终止后单独授权测试既有attention_geometry；没有恢复新损失或架构训练。模型为第77轮best、step48125，后处理参数和100个成员及10个展示样本在读取测试前固定。', '',
        '| 指标 | 结果 |','|---|---:|',f'| Q算术均值 | {stats["Q_mean"]:.6f} |',
        f'| 块数加权Q | {stats["Q_weighted"]:.6f} |',f'| Q中位数 / P95 / 最大值 | {stats["Q_median"]:.6f} / {stats["Q_p95"]:.6f} / {stats["Q_max"]:.6f} |',
        f'| 硬合法 | {stats["feasible"]}/100 |',f'| 执行错误 | {stats["execution_errors"]} |',
        f'| Q＜1.2 / Q＜1.15 | {stats["Q_below1_2"]} / {stats["Q_below1_15"]} |']
    timing=stats['postprocess_seconds']
    if timing:lines.append(f'| 后处理均值 / 中位 / P95秒 | {timing["mean"]:.3f} / {timing["median"]:.3f} / {timing["p95"]:.3f} |')
    for k,label in [('hpwl_gap_mean','HPWL有符号gap均值'),('area_gap_mean','bbox有符号gap均值'),('boundary_violations_mean','边界违反均值'),('grouping_violations_mean','group违反均值'),('mib_violations_mean','MIB违反均值')]:
        value=stats[k];lines.append(f'| {label} | {value:.6f} |' if value is not None else f'| {label} | 未定义 |')
    lines+=['','主分母包含所有100例，错误计Q=10；未返回布局的耗时/几何分项不能冒充全100例均值。Q为官方中性时间对照，不是正式含时间因子的比赛成绩。', '',
        '该面板来自原workset_v2的local_test，保守排除旧评价及可能在途的worker0..39，覆盖余下60个worker；其中40个worker各2例、20个各1例。不是100worker全覆盖，也不声称按模块数精确分层。选样仅依赖路径和行号，训练/开发分片与之不交叉；没有额外验证跨分片生成家族独立性。', '',
        '数据加载器以分片文件为容器，只对预选100个成员逐例解码和评分；不主动评价、画像或用其余测试成员选模型。已访问分片记录在cases.jsonl的source中，后续测试暴露应按实际访问记录披露。', '',
        '10个图例事先按元数据选定，来自10个不同worker，不按分数挑最好或最差例；图中红框标重叠、颜色标group、纹理标MIB，三栏同尺度。图像只从保存坐标绘制，不再次读取测试张量或运行后处理。', '',
        '[浏览10例图集](index.html)；[精确汇总](results.json)；[冻结面板及流程](plan.json)。', '',
        '这是当前单一冻结模型的100例本地测试观察。若后续按这些分数/图像修改模型或选解流程，须披露反馈使用，不再宣称同面板成绩为独立未见检验。','']
    (OUT/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    return items


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--report-only',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare:
        plan=expected_plan();path=OUT/'plan.json'
        if path.exists():read_plan()
        else:save(path,plan)
        print(json.dumps({'status':'prepared_not_run','cases':100,'workers':60,'gallery':10,'test_tensor_reads':0}));return
    plan=read_plan()
    if args.check:
        state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        assert state['step']==plan['best_step'] and state['model_config']==plan['model']
        fixture=json.loads((HERE/'runs/20261017_attention_geometry_dev10_seed0/cases.jsonl').read_text(encoding='utf-8').splitlines()[0])
        checks=OUT/'checks';checks.mkdir(exist_ok=True);plot_case(fixture,checks/'gallery_format.png')
        save(OUT/'checks.json',{'status':'passed','panel_size':100,'gallery_size':10,'distinct_gallery_workers':10,
            'metadata_membership_disjoint_from_train':True,'test50_nested':True,'checkpoint_metadata_matches':True,
            'saved_dev_plot':'checks/gallery_format.png','test_tensor_reads':0,'model_forward_calls':0})
        print(json.dumps({'status':'checked_not_run','test_tensor_reads':0,'model_forward_calls':0}));return
    if args.report_only:
        rows=read_rows();items=report_and_gallery(plan,rows)
        result_path=OUT/'results.json'
        if result_path.exists():
            result=json.loads(result_path.read_text(encoding='utf-8'));result.update(gallery_status='completed',gallery_figures=len(items));save(result_path,result)
        print(json.dumps({'status':'reported','figures':len(items),'test_tensor_reads':0}));return
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);begin=time.perf_counter()
    config_path=OUT/'config.json'
    if config_path.exists() and not args.resume:raise ValueError('Evaluation exists; use --resume')
    if not config_path.exists():
        config={'plan_sha256':sha(OUT/'plan.json'),'test_authorization':'current explicit user request for100test cases and10layout figures',**provenance()}
        save(config_path,config);files=dict(plan['source_sha256']);files[(OUT/'plan.json').relative_to(ROOT).as_posix()]=sha(OUT/'plan.json');snapshot(OUT,files)
    elif json.loads(config_path.read_text(encoding='utf-8'))['plan_sha256']!=sha(OUT/'plan.json'):raise ValueError('Plan hash mismatch')
    rows=read_rows(repair=args.resume);allowed={identity(m) for m in plan['members']};done={r['case_id'] for r in rows}
    if len(done)!=len(rows) or not done<=allowed:raise ValueError('Duplicate or out-of-panel result')
    if len(rows)==100:
        report_and_gallery(plan,rows);print(json.dumps({'status':'already_complete','cases':100,'test_tensor_reads':0}));return
    append(OUT/'invocations.jsonl',{'created_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],'resume':args.resume,'completed_before':len(rows)})
    reader=LayoutReader(workset_manifest(),'local_test',allow_local_test=True)
    indices={p:i for i,p in enumerate(reader.members)};model=load_model(CHECKPOINT,'cuda');stopped=[False];reads=0;warm=False
    signal.signal(signal.SIGINT,lambda *unused:stopped.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *unused:stopped.__setitem__(0,True))
    try:
        for member in plan['members']:
            if identity(member) in done:continue
            if stopped[0] or (OUT/'STOP').exists():break
            raw=reader.case(indices[(member['shard'],member['sample_index'])]);reads+=1
            scoring=_build_case(raw.case_id,raw.source,raw.solver_inputs[1:6],raw.reference_positions,raw.stored_metrics)
            if not warm:
                with torch.no_grad():model(prepare_inference(raw.solver_inputs,'cuda'))
                torch.cuda.synchronize();warm=True
            value={};row={'case_id':raw.case_id,'source':raw.source,'block_count':raw.block_count,
                'reference_positions':raw.reference_positions,'constraints':np.asarray(raw.solver_inputs[5]).tolist(),'variants':{NAME:value}}
            try:
                t=time.perf_counter();g=prepare_inference(raw.solver_inputs,'cuda');torch.cuda.synchronize();prep=time.perf_counter()-t;t=time.perf_counter()
                with torch.no_grad():pos=(decode(model(g),g)[0]*g['scale'][0]).cpu().numpy().astype(float)
                torch.cuda.synchronize();forward=time.perf_counter()-t
                c=np.asarray(raw.solver_inputs[5]);spec=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
                pos[locked,2:]=spec[locked,2:];pos[c[:,1]!=0]=spec[c[:,1]!=0]
                value.update(raw_positions=pos.tolist(),raw_diagnostics=exact_diagnostics(pos))
                t=time.perf_counter();final,trace=optimize(pos,raw.solver_inputs,**plan['postprocess']);post=time.perf_counter()-t
                evaluated=evaluate_case(scoring,final,prep+forward+post)
                areas=np.asarray(raw.solver_inputs[1]).copy();areas[locked]=spec[locked,2:].prod(1)
                distance=np.linalg.norm(final[:,:2]+final[:,2:]/2-pos[:,:2]-pos[:,2:]/2,axis=1)/np.sqrt(areas)
                value.update(final=evaluated,final_diagnostics=exact_diagnostics(final),feature_seconds=prep,forward_seconds=forward,
                    postprocess_seconds=post,end_to_end_seconds=prep+forward+post,trace=trace,
                    shape_iterations=sum(x.get('phase')=='shape' for x in trace['trace']),trace_subproblems=len(trace['trace']),
                    guard_fallback=int(trace['guard_fallback']),center_displacement_local_mean=float(distance.mean()))
            except Exception as exc:
                value.update(error={'type':type(exc).__name__,'message':str(exc)},final={'metrics':{'cost':10.,'is_feasible':False}})
            append(OUT/'cases.jsonl',row);rows.append(row);done.add(raw.case_id)
            save(OUT/'results.json',{'status':'running','completed_cases':len(rows),'expected_cases':100,'test_payload_reads_this_invocation':reads,**summarize(rows,[NAME])})
            print(json.dumps({'complete':len(rows),'case_id':raw.case_id,'Q':value['final']['metrics']['cost'],
                'feasible':value['final']['metrics']['is_feasible'],'postprocess_seconds':value.get('postprocess_seconds'),
                'elapsed_seconds':time.perf_counter()-begin}),flush=True)
    finally:
        summary=summarize(rows,[NAME])
        save(OUT/'results.json',{'status':'completed' if len(rows)==100 else 'interrupted','completed_cases':len(rows),'expected_cases':100,
            'test_payload_reads_this_invocation':reads,'evaluated_test_cases_cumulative':len(rows),'wall_seconds_this_invocation':time.perf_counter()-begin,
            'scope':plan['scope'],'gallery_status':'pending','Q_below1_2_on_complete_panel':bool(len(rows)==100 and summary['variants'][NAME]['Q_mean']<1.2 and summary['variants'][NAME]['feasible']==100),**summary})
    if len(rows)==100:
        items=report_and_gallery(plan,rows);result=json.loads((OUT/'results.json').read_text(encoding='utf-8'));result.update(gallery_status='completed',gallery_figures=len(items));save(OUT/'results.json',result)
        print(json.dumps({'status':'completed','Q_mean':result['variants'][NAME]['Q_mean'],'gallery_figures':len(items),'gallery':str(OUT/'index.html')}),flush=True)


if __name__=='__main__':main()
