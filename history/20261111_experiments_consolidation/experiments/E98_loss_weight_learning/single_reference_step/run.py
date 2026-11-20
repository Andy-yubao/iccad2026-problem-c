"""One candidate train -> train-only comparison -> frozen terminal dev500 -> stop."""
import argparse
import gc
import json
import os
import signal
import threading
import time
import traceback
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..io import exclusive
from ..generalization_baseline.data import load_rows,register_resident
from ..dataset_view import ReferenceLayouts
from ..resident import ResidentRows
from ..qualification import evaluate as legacy
from ..depth_evaluation.preflight import sources as depth_sources
from ..depth_evaluation.repair.canonical_native import warmup
from datasets.reference_layouts import public_hard_reference
from .common import ROOT,HERE,OWNER,RUN,PLAN_PATH,read,save,sha,sources,q_summary
from .training import train_branch,Halt
from .evaluation import Predictor,model_hash,input_hash,evaluate

def prepare_cases(rows):
    for row in rows:
        case=row['case']
        case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])

def render_gallery(dev,parent_records,candidate_records,check):
    from visualization.layouts import plot_case,geometry_details
    folder=RUN/'gallery';folder.mkdir(exist_ok=True)
    for i,row in enumerate(dev[:100]):
        check()
        c=np.asarray(row['case'].solver_inputs[5]).tolist()
        reference=(row['graph']['reference']*row['graph']['scale']).numpy().tolist()
        variants={name:dict(positions=p,soft_details=geometry_details(p,c)) for name,p in (
            ('reference',reference),('baseline',parent_records[i]['draft_positions']),
            ('candidate',candidate_records[i]['draft_positions']),('final',candidate_records[i]['positions']))}
        image=folder/f'{i:03d}.png'
        if not image.exists():
            plot_case(dict(case_id=f'dev500:{i}',panel='E98 single reference weakening',constraints=c,
                variants=variants),image,stages=(('reference','Reference'),('baseline','Baseline L'),
                ('candidate','Weakened L'),('final','Candidate final')),show_metrics=False)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{width:100%}</style>'+''.join(
        f'<p>{i}</p><img loading="lazy" src="{i:03d}.png">' for i in range(100)),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    plan=read(PLAN_PATH);current=sources()
    gate=read(OWNER/'preflight.json')
    assert gate['status']=='passed' and gate['sources']==current
    parent=ROOT/plan['parent_checkpoint'];base_eval=ROOT/plan['parent_evaluation']
    assert sha(parent)==plan['parent_sha256']
    assert sha(base_eval/'frozen.json')==plan['parent_frozen_evaluation_sha256']
    assert sha(base_eval/'results.json')==plan['parent_evaluation_results_sha256']
    old_frozen=read(base_eval/'frozen.json')
    for name,digest in depth_sources().items():
        assert old_frozen['sources'][name]==digest,'Depth evaluator drift'
    assert read(HERE/'depth_evaluation/preflight.json')['sources']==depth_sources()
    identity=dict(plan_sha256=sha(PLAN_PATH),parent_sha256=sha(parent),sources=current)
    with exclusive(RUN/'worker.lock'):
        if (RUN/'results.json').exists():
            print('Single step already complete; not repeating.',flush=True);return
        if (RUN/'config.json').exists():
            assert args.resume and read(RUN/'config.json')['identity']==identity
            assert not (RUN/'error.json').exists(),'Error requires diagnosis'
            prior=read(RUN/'state.json')['elapsed_seconds']
        else:
            assert not args.resume
            prior=0.
            save(RUN/'config.json',dict(identity=identity,plan=plan))
            with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in current:archive.write(ROOT/name,name)
        start=time.perf_counter();requested=[False];closed=threading.Event();lock=threading.RLock()
        signal_before=signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
        status=dict(status='running',stage='initializing',test_read=False,pid=os.getpid())
        def check():
            if requested[0] or (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise Halt('user_stop')
        def beat(**fields):
            with lock:
                status.update(fields)
                save(RUN/'state.json',dict(**status,elapsed_seconds=prior+time.perf_counter()-start,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
        try:
            check();legacy.setup();beat(stage='loading_train16k_dev500')
            rows=load_rows('train');dev=load_rows('dev_monitor500')
            assert len(rows)==16000 and len(dev)==500
            assert [list(r['member']) for r in rows]==[r['member'] for r in ReferenceLayouts('train').records]
            assert [list(r['member']) for r in dev]==old_frozen['members']
            pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
            lookup={tuple(r['member']):i for i,r in enumerate(rows)}
            tr_monitor=[rows[lookup[tuple(r['member'])]] for r in ReferenceLayouts('train_monitor500').records]
            spec=dict(round=1,sampler_seed=plan['sampler_seed'])
            candidate,training=train_branch(RUN/'candidate',parent,plan['candidate_coefficients'],spec,
                'weaken_reference',pack,tr_monitor,dev,identity,check,beat)
            check()
            freeze=dict(checkpoint=candidate.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(candidate),
                parent_sha256=sha(parent),coefficients=plan['candidate_coefficients'],
                selection='own fixed dev500 Loss best; frozen before any Q',test_read=False)
            if (RUN/'candidate_frozen.json').exists():assert read(RUN/'candidate_frozen.json')==freeze
            else:save(RUN/'candidate_frozen.json',freeze)
            feedback=[rows[i] for i in plan['feedback_train_indices']]
            assert [list(r['member']) for r in feedback]==plan['feedback_members']
            prepare_cases(feedback);prepare_cases(dev)
            torch.set_num_threads(1)
            with torch.inference_mode(),threadpool_limits(limits=1):
                warmup();predictor=Predictor(parent,feedback[0]['case'].solver_inputs)
                threads=legacy.thread_check()
            save(RUN/'evaluation_environment.json',threads)
            base_train,_=evaluate(predictor,parent,feedback,RUN/'train_feedback/baseline',100,check,beat,'train_baseline')
            cand_train,_=evaluate(predictor,candidate,feedback,RUN/'train_feedback/candidate',100,check,beat,'train_candidate')
            train_comparison=dict(baseline=base_train,candidate=cand_train,
                mixed_Q_delta=cand_train['mixed_Q']-base_train['mixed_Q'],
                improved=cand_train['mixed_Q']<base_train['mixed_Q']-plan['minimum_gain'] and cand_train['hard_legal']==100,
                feedback_unique_cases=100,feedback_pool_cases=1000,test_read=False)
            save(RUN/'train_comparison.json',train_comparison)
            check()
            baseline_records=[read(base_eval/f'cases/{i:04d}.json') for i in range(500)]
            for row,record in zip(dev,baseline_records):
                assert record['member']==list(row['member']) and record['input_hash']==input_hash(row['case'].solver_inputs)
            baseline=read(base_eval/'results.json')
            candidate_state=torch.load(candidate,map_location='cpu',weights_only=False)
            same_model=model_hash(candidate_state['model'])==old_frozen['model_hash']
            del candidate_state
            if same_model:
                terminal=dict(baseline,identical_to_parent_model=True,reused_completed_baseline=True)
                candidate_records=baseline_records
                beat(stage='terminal_same_model_reused',completed=500,total=500)
            else:
                terminal,candidate_records=evaluate(predictor,candidate,dev,RUN/'terminal_dev500',500,check,beat,'terminal_candidate_dev500')
            paired=np.array([c['metrics']['cost']-b['metrics']['cost'] for b,c in zip(baseline_records,candidate_records)])
            comparison=dict(baseline={k:baseline[k] for k in ('weighted_Q','mean_Q','mixed_Q','hard_legal','mean_complete_seconds','p95_complete_seconds','max_complete_seconds','over250ms')},
                candidate=terminal,weighted_Q_delta=terminal['weighted_Q']-baseline['weighted_Q'],
                mean_Q_delta=terminal['mean_Q']-baseline['mean_Q'],mixed_Q_delta=terminal['mixed_Q']-baseline['mixed_Q'],
                improved_cases=int((paired<-1e-9).sum()),worsened_cases=int((paired>1e-9).sum()),
                unchanged_cases=int((np.abs(paired)<=1e-9).sum()),
                same_depth_evaluator=True,baseline_reused=True,checkpoint_not_reselected_by_Q=True,test_read=False)
            save(RUN/'dev500_comparison.json',comparison)
            beat(stage='rendering_same100',completed=500,total=500)
            render_gallery(dev,baseline_records,candidate_records,check)
            assert sha(parent)==plan['parent_sha256'] and sha(candidate)==freeze['checkpoint_sha256']
            result=dict(status='complete',single_candidate_only=True,training=training,candidate_frozen=freeze,
                train100_comparison=train_comparison,dev500_comparison=comparison,
                previous_search_results_excluded=True,automatic_next_stage=False,test_read=False,
                elapsed_seconds=prior+time.perf_counter()-start)
            save(RUN/'results.json',result)
            report=['# E98单步削弱位置与形状reference','',
                '唯一父模型为泛化增强best71；仅position/shape除以1.2，其余8项不变，不作总和归一化。旧搜索结果不参与。',
                f'候选训练：{training["status"]}，{training["epochs"]}轮，best={training["best_epoch"]}。验证Loss选定唯一best后才计算Q。','',
                '|面板/模型|加权Q|平均Q|混合Q|硬合法|','|---|---:|---:|---:|---:|']
            for panel,name,score in [('train100','baseline',base_train),('train100','candidate',cand_train),('dev500','baseline',baseline),('dev500','candidate',terminal)]:
                report.append(f'|{panel}/{name}|{score["weighted_Q"]:.9f}|{score["mean_Q"]:.9f}|{score["mixed_Q"]:.9f}|{score["hard_legal"]}/{score["count"]}|')
            report += ['',f'训练反馈混合Q变化：{train_comparison["mixed_Q_delta"]:+.9f}；达到接纳门槛：{train_comparison["improved"]}。',
                f'dev500混合Q变化：{comparison["mixed_Q_delta"]:+.9f}；改善/变差/持平例数：{comparison["improved_cases"]}/{comparison["worsened_cases"]}/{comparison["unchanged_cases"]}。',
                '',f'候选完整耗时均值/P95：{terminal["mean_complete_seconds"]*1000:.3f}/{terminal["p95_complete_seconds"]*1000:.3f}ms；超250ms：{terminal["over250ms"]}。',
                '父模型使用已完成的同深度评价，耗时为历史测量，不称同场配对速度实验。新旧目标总Loss不能直接比较。',
                '这是基线devQ已曝光后的单步开发实验，收益也可能来自额外训练；不称独立未见泛化或纯调权因果收益。',
                'devQ不触发其他候选或checkpoint重选。测试未读取，所有旧搜索和停止标记保留。','',
                '[同100例reference/基线草稿/候选草稿/候选最终布局](gallery/index.html)','']
            (RUN/'report.md').write_text('\n'.join(report),encoding='utf-8')
            beat(status='complete',stage='complete_no_next_stage')
            print(json.dumps(dict(status='complete',train_mixed_Q_delta=train_comparison['mixed_Q_delta'],
                dev_mixed_Q_delta=comparison['mixed_Q_delta'],candidate_best_epoch=training['best_epoch'])),flush=True)
        except Halt:
            beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry')
            raise
        finally:
            closed.set();worker.join(timeout=6);signal.signal(signal.SIGINT,signal_before)

if __name__=='__main__':main()
