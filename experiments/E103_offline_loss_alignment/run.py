"""Bounded background database -> offline ranking -> direction check -> report."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import zipfile
import torch
from threadpoolctl import threadpool_limits
from .common import ROOT,OWNER,RUN,read,save,sha,sources,setup,check
from .cache import load_rows
from .worker import initialize,bank_case,direction_case

def materialize(indices):
    panel=read(OWNER/'panel.json')
    missing=[i for i in indices if not (RUN/f'inputs/{i:04d}.pt').exists()]
    if not missing:return
    rows=load_rows('train',[panel['train_indices'][i] for i in missing])
    for i,row in zip(missing,rows):
        check()
        assert list(row['member'])==panel['members'][i]
        path=RUN/f'inputs/{i:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(row,path)
    print('Prepared input rows',len(missing),flush=True)

def parallel_map(function,indices,identity,beat,stage,completed_offset=0):
    done_count=completed_offset;indices=iter(indices)
    context=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initialize,initargs=(identity,)) as pool:
        pending={}
        for _ in range(4):
            i=next(indices,None)
            if i is not None:pending[pool.submit(function,i)]=i
        while pending:
            check()
            ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                i=pending.pop(future);future.result();done_count+=1
                beat(stage=stage,completed=done_count)
                print(stage,done_count,'case',i,flush=True)
                check()
                j=next(indices,None)
                if j is not None:pending[pool.submit(function,j)]=j

def write_report(analysis,direction):
    lines=['# E103 1000×5离线Loss对齐研究','',
        '五个完成训练的不同模型，同1000个原训练成员，共5000份原始草稿。E98固定计算深度后处理，四进程单线程CPU执行。',
        '','## 五个来源的同1000题结果','',
        '|模型|加权Q|平均Q|混合Q|硬合法|','|---|---:|---:|---:|---:|']
    for mid,r in analysis['source_summary'].items():
        lines.append(f"|{mid}|{r['weighted_Q']:.9f}|{r['mean_Q']:.9f}|{r['mixed_Q']:.9f}|{r['hard_legal']}/1000|")
    lines+=['','## 按题目隔离的内部检查200题','',
        '|Loss|同题排序准确率|选草稿混合Q|相对每题最优的混合遗憾|',
        '|---|---:|---:|---:|']
    for name,r in analysis['final_internal_check'].items():
        lines.append(f"|{name}|{r['pair_accuracy']:.4%}|{r['mixed_Q']:.9f}|{r['mixed_regret']:.9f}|")
    lines+=['','## 固定权重提案','',
        '```json',json.dumps(analysis['proposal']['physical_coefficients'],ensure_ascii=False,indent=2),'```',
        '','## 局部下降方向检查','',json.dumps(direction,ensure_ascii=False,indent=2),
        '','所有1000题都属于源模型的训练分片；检查200题仅未参与系数拟合，不代表网络未见数据。',
        '通过排序不等于训练有效；布局空间下降不等于可实现的神经网络参数更新。',
        '四路并发墙钟耗时只作本次吞吐诊断，不作为逐例部署延迟或比赛成绩。',
        '没有读取验证后处理Q或测试载荷，没有更改或恢复E98，也没有自动启动新神经网络长训练。',
        '后续根据离线与方向证据登记固定目标和有限同父模型训练对照，不能按内部检查集继续扫权重。','']
    (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--recovery',help='Explicit reviewed I/O repair migration record, relative to repository')
    args=parser.parse_args()
    setup();current=sources();recovery=None
    if args.recovery:
        assert args.resume
        recovery=read(ROOT/args.recovery)
        assert recovery['status']=='validated_for_restart'
        assert recovery['new_identity']['source_hashes']==current
        assert sha(RUN/'config.json')==recovery['original_config_sha256']
        assert recovery['targeted_io_check']['status']=='passed'
        identity=recovery['new_identity']
        for record in recovery['preserved_bank_records']:
            assert sha(ROOT/record['path'])==record['sha256']
    else:
        gate=read(OWNER/'preflight.json')
        assert gate['status']=='passed' and gate['sources']==current
        identity=gate['identity'];assert identity['source_hashes']==current
    if not args.resume and (RUN/'config.json').exists():raise RuntimeError('Run already launched')
    if (RUN/'results.json').exists() or (RUN/'error.json').exists():raise RuntimeError('Completed/error run requires review, no blind retry')
    # OS lock is released even if supervisor or worker crashes.
    lockpath=RUN/'pipeline.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a+b') as handle:
        import msvcrt
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        if args.resume:
            if recovery:
                assert read(RUN/'config.json')['identity']==recovery['old_identity']
                save(RUN/'recovery_config.json',dict(identity=identity,recovery_record=args.recovery))
            else:assert read(RUN/'config.json')['identity']==identity
        else:
            save(RUN/'config.json',dict(identity=identity,plan=read(OWNER/'plan.json')))
            with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in current:archive.write(ROOT/name,name)
        started=time.perf_counter();hb=dict(status='running',stage='initializing',workers=4,test_read=False)
        mutex=threading.RLock();closed=threading.Event()
        def beat(**fields):
            with mutex:
                hb.update(fields)
                save(RUN/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        def stop(*_):
            (RUN/'STOP').write_text('User interrupt; preserve completed rows and stop later stages.\n',encoding='utf-8')
        prior=signal.signal(signal.SIGINT,stop)
        try:
            check();beat(stage='materializing_train1000_inputs',total=1000,completed=0)
            materialize(range(1000))
            missing=[i for i in range(1000) if not (RUN/f'bank/{i:04d}.json').exists()]
            complete=1000-len(missing)
            check();beat(stage='database1000x5',total=1000,completed=complete)
            print('Preserved completed cases',complete,'remaining',len(missing),flush=True)
            parallel_map(bank_case,missing,identity,beat,'database1000x5',completed_offset=complete)
            check();beat(stage='offline_weight_analysis',total=4,completed=0)
            from .analysis import analyze,direction_summary
            with threadpool_limits(limits=1):analysis=analyze()
            check();indices=read(OWNER/'panel.json')['direction_indices']
            beat(stage='local_descent_check',total=40,completed=0)
            parallel_map(direction_case,indices,identity,beat,'local_descent_check')
            check();direction=direction_summary();write_report(analysis,direction)
            result=dict(status='complete_offline_research',core_cases=1000,models=5,core_drafts=5000,
                direction_cases=40,direction_additional_evaluations=160,workers=4,
                descent_gate_passed=direction['descent_gate_passed'],
                next_action=direction['training_recommendation'],neural_training_started=False,
                test_read=False,elapsed_seconds=time.perf_counter()-started)
            save(RUN/'results.json',result);beat(status='complete',stage='offline_report_ready')
            print(json.dumps(result),flush=True)
        except InterruptedError:
            beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry');raise
        finally:
            closed.set();thread.join(timeout=6);signal.signal(signal.SIGINT,prior)

if __name__=='__main__':main()
