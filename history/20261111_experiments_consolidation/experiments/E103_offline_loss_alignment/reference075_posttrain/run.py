"""One explicit train -> freeze -> dev500 experiment, no search or extra branches."""
import gc
import os
import threading
import time
import traceback
import zipfile
import torch
from .common import ROOT,OWNER,RUN,read,save,sha,sources

def main():
    current=sources();gate=read(OWNER/'preflight.json')
    assert gate['status']=='passed' and gate['sources']==current
    if (RUN/'pipeline_config.json').exists():raise RuntimeError('Pipeline already launched; no duplicate or blind retry')
    RUN.mkdir(parents=True,exist_ok=True)
    save(RUN/'pipeline_config.json',dict(plan=read(OWNER/'plan.json'),sources=current))
    with zipfile.ZipFile(RUN/'pipeline_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in current:z.write(ROOT/name,name)
    started=time.perf_counter();stage=['training'];closed=threading.Event()
    def stopped():return (RUN/'STOP').exists() or (OWNER/'STOP').exists()
    def beat(status='running',**fields):
        progress_path=RUN/'state.json' if stage[0]=='training' else RUN/'evaluation_depth_dev500/state.json'
        progress=read(progress_path) if progress_path.exists() else {}
        save(RUN/'pipeline_state.json',dict(status=status,stage=stage[0],pid=os.getpid(),active_progress=progress,
            elapsed_seconds=time.perf_counter()-started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),test_read=False,**fields))
    def heartbeat():
        while not closed.wait(5):beat()
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if stopped():raise InterruptedError('user stop')
        beat()
        from .train import main as train
        train()
        train_state=read(RUN/'state.json')
        if stopped() or train_state['status']=='user_stop':raise InterruptedError('user stop before evaluation')
        assert train_state['status']=='complete' and read(RUN/'results.json')['status']=='early_stop'
        # Release the training-only pinned buffers before batch1 evaluation.
        from . import data
        data._resident.clear();data._loaded.clear();gc.collect();torch.cuda.empty_cache()
        stage[0]='evaluating_frozen_dev500';beat()
        from .evaluation.run import main as validate
        validate()
        if stopped() or read(RUN/'evaluation_depth_dev500/state.json')['status']=='user_stop':
            raise InterruptedError('user stop during evaluation')
        train_result=read(RUN/'results.json');evaluation=read(RUN/'evaluation_depth_dev500/results.json')
        assert evaluation['status']=='complete'
        result=dict(status='complete',training=train_result,validation=evaluation,
            coefficients=read(OWNER/'plan.json')['coefficients'],single_candidate=True,
            elapsed_seconds=time.perf_counter()-started,automatic_next_stage=False,test_read=False)
        save(RUN/'pipeline_results.json',result)
        (RUN/'pipeline_report.md').write_text('# E98参考位置/形状权重×0.75实验\n\n'+
            f"训练正常早停于{train_result['epochs']}轮，best={train_result['best_epoch']}，新验证Loss={train_result['best_dev_loss']:.9f}。\n\n"+
            f"同dev500混合Q：{evaluation['parent']['mixed_Q']:.9f}→{evaluation['candidate']['mixed_Q']:.9f}。\n\n"+
            '[完整验证报告与同100例图](evaluation_depth_dev500/report.md)\n\n'+
            '仅一个候选，父模型不训练，测试未读取，流水线完成后停止。\n',encoding='utf-8')
        closed.set();thread.join(timeout=6)
        stage[0]='complete_no_next_stage';beat(status='complete')
    except InterruptedError:
        closed.set();thread.join(timeout=6);beat(status='user_stop')
    except BaseException as exc:
        closed.set();thread.join(timeout=6)
        save(RUN/'pipeline_error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
        beat(status='error');raise
    finally:closed.set();thread.join(timeout=6)

if __name__=='__main__':main()
