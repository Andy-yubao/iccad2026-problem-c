"""Four-way bounded diagnosis using existing data and fixed registered routes."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os
import threading
import time
import traceback
import zipfile
import numpy as np
from ..common import ROOT,OWNER,read,save,sha,setup
from .study import HERE,RUN,BANK,run_task

def main():
    setup();plan=read(HERE/'plan.json')
    if (RUN/'config.json').exists():raise RuntimeError('Diagnosis already started; no duplicate or blind retry')
    assert read(BANK/'audits/q_sanity.json')['status']=='passed'
    inputs=plan['frozen_input_hashes']
    for name,digest in inputs.items():assert sha(ROOT/name)==digest
    current={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.glob('*.py')}
    for name in ('analysis.py','simple_loss.py','batches.py','common.py','worker.py','loss_config.py'):
        current[(OWNER/name).relative_to(ROOT).as_posix()]=sha(OWNER/name)
    RUN.mkdir(parents=True,exist_ok=True)
    save(RUN/'config.json',dict(plan=plan,sources=current,plan_sha256=sha(HERE/'plan.json')))
    with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in current:z.write(ROOT/name,name)
        z.write(HERE/'plan.json',(HERE/'plan.json').relative_to(ROOT).as_posix())
    started=time.perf_counter();state=dict(status='running',stage='preparing_fit_tune_data',completed=0,total=4,workers=4,test_read=False)
    mutex=threading.RLock();closed=threading.Event()
    def beat(**fields):
        with mutex:
            state.update(fields)
            save(RUN/'state.json',dict(**state,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    def heartbeat():
        while not closed.wait(5):beat()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    def check():
        if (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise InterruptedError('user stop')
    try:
        panel=read(OWNER/'panel.json');indices=panel['analysis_splits']['fit']+panel['analysis_splits']['tune']
        assert len(indices)==800 and not set(indices).intersection(panel['analysis_splits']['check'])
        x=[];q=[];n=[]
        bank_digests={r['path']:r['sha256'] for r in read(BANK/'audits/q_sanity.json')['bank_hashes']}
        for i in indices:
            check();path=BANK/f'bank/{i:04d}.json';assert sha(path)==bank_digests[path.relative_to(ROOT).as_posix()]
            row=read(path)
            x.append([[d['loss_terms'][k] for k in plan['loss_keys']] for d in row['drafts']])
            q.append([d['Q'] for d in row['drafts']]);n.append(row['n'])
        np.savez_compressed(RUN/'fit_tune.npz',x=np.asarray(x),q=np.asarray(q),n=np.asarray(n),indices=np.asarray(indices))
        beat(stage='four_parallel_diagnostics')
        completed=[]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            pending={pool.submit(run_task,name):name for name in ('capacity','stability','canonical_view','descent')}
            while pending:
                check();ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in ready:
                    name=pending.pop(future);future.result();completed.append(name)
                    beat(completed=len(completed),completed_tasks=completed)
                    print('Completed diagnosis',name,flush=True)
        result=dict(status='complete',tasks=completed,training_started=False,new_model_inference=0,
            new_postprocessor_calls=0,check200_used_for_new_tuning=False,test_read=False,
            canonical_candidate_gate_passed=read(RUN/'canonical_view.json')['geometry_candidate_gate_passed'])
        save(RUN/'results.json',result)
        lines=['# E103第二阶段：Loss机制诊断','',
            'Q数值核验已通过。所有新拟合仅使用原600拟合题和200调参题，不再用已曝光check200调权。',
            '原始1000×5数据库、首轮系数和后处理器不修改；本阶段没有新网络训练或后处理调用。','']
        capacity=read(RUN/'capacity.json')
        lines+=['## 表达能力与来源混淆','', '|诊断|调参同题排序准确率|调参混合遗憾|','|---|---:|---:|']
        for name,r in capacity['tune'].items():lines.append(f"|{name}|{r['pair_accuracy']:.4%}|{r['mixed_regret']:.9f}|")
        lines+=['','## 解码几何监督','',json_repr(read(RUN/'canonical_view.json')),
            '','## 局部下降机制','',json_repr(read(RUN/'descent_diagnosis.json')['summary']),
            '','## 权重稳定性','',json_repr(read(RUN/'stability.json')),
            '','这些是同一开发数据上的机制诊断，不是新未见泛化证据。带符号/二次拟合只作表达能力诊断，不作为可直接训练的Loss。',
            '新神经网络训练仍须固定目标并有下降方向证据；未因排序指标较好而绕过原先失败的门槛。','']
        (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        beat(status='complete',stage='diagnostic_report_ready')
        print(result,flush=True)
    except InterruptedError:
        beat(status='user_stop',stage='stopped')
    except BaseException as exc:
        save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        beat(status='error',stage='failed_no_retry');raise
    finally:closed.set();thread.join(timeout=6)

def json_repr(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'

if __name__=='__main__':main()
