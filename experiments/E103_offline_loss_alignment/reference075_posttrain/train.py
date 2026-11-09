"""One reference075 training phase; enclosing pipeline performs terminal dev500."""
import argparse
import gc
import os
import signal
import threading
import time
import traceback
import zipfile
import torch
from datasets.reference_layouts import ReferenceLayouts
from .common import ROOT,OWNER,RUN,PLAN_PATH,MODEL,read,save,sha,sources
from .runtime import setup
from .data import load_rows,register_resident
from .resident import ResidentRows
from .training import train_branch,Halt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    setup();plan=read(PLAN_PATH);current=sources();gate=read(OWNER/'preflight.json')
    assert gate['status']=='passed' and gate['sources']==current
    assert plan['maximum_epochs'] is None and plan['maximum_seconds'] is None
    parent=ROOT/plan['source_checkpoint'];assert sha(parent)==plan['source_sha256']
    assert plan['coefficients']['position']==.75 and plan['coefficients']['shape']==.1875
    identity=dict(plan_sha256=sha(PLAN_PATH),sources=current,parent_sha256=plan['source_sha256'],
        coefficient_policy='original E98 position/shape multiplied by.75, other8 unchanged')
    if (RUN/'results.json').exists() or (RUN/'error.json').exists():raise RuntimeError('Completed/error run requires review')
    if not args.resume and (RUN/'config.json').exists():raise RuntimeError('Run already started')
    RUN.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (RUN/'training.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        if args.resume:assert read(RUN/'config.json')['identity']==identity
        else:
            save(RUN/'config.json',dict(identity=identity,plan=plan))
            with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in current:archive.write(ROOT/name,name)
        prior_elapsed=read(RUN/'state.json')['elapsed_seconds'] if args.resume else 0.
        started=time.perf_counter();hb=dict(status='running',stage='initializing',test_read=False)
        mutex=threading.RLock();closed=threading.Event();stopped=[False]
        def check():
            if stopped[0] or (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise Halt('user_stop')
        def beat(**fields):
            with mutex:
                hb.update(fields)
                save(RUN/'state.json',dict(**hb,pid=os.getpid(),elapsed_seconds=prior_elapsed+time.perf_counter()-started,
                    updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        def heartbeat():
            while not closed.wait(5):beat()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        previous=signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
        try:
            check();beat(stage='loading_train16k_and_dev500')
            rows=load_rows('train');dev=load_rows('dev_monitor500')
            assert len(rows)==16000 and len(dev)==500
            assert [list(r['member']) for r in rows]==[r['member'] for r in ReferenceLayouts('train').records]
            assert [list(r['member']) for r in dev]==[r['member'] for r in ReferenceLayouts('dev_monitor500').records]
            lookup={tuple(r['member']):i for i,r in enumerate(rows)}
            tr=[rows[lookup[tuple(r['member'])]] for r in ReferenceLayouts('train_monitor500').records]
            pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
            check();beat(stage='training',epoch=0,step=0,lr=.0001)
            best,summary=train_branch(RUN,parent,plan['coefficients'],plan['phase'],'reference075',
                pack,tr,dev,identity,check,beat)
            frozen=dict(checkpoint=best.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(best),
                best_epoch=summary['best_epoch'],coefficients=plan['coefficients'],selection='own fixed dev500 Loss only',test_read=False)
            save(RUN/'frozen.json',frozen)
            summary.update(frozen=frozen,maximum_epochs=None,maximum_seconds=None,protection_epochs=0,
                postprocessing_calls=0,automatic_next_stage=True)
            save(RUN/'results.json',summary)
            (RUN/'report.md').write_text('# E98参考监督0.75倍后训练\n\n'+
                f"结束原因：{summary['status']}；完成{summary['epochs']}轮/{summary['updates']}更新，best={summary['best_epoch']}。\n\n"+
                f"新固定验证Loss={summary['best_dev_loss']:.9f}。初始LR1e-4、无保护、无轮数/时间上限，按验证Loss正常早停。\n\n"+
                '父模型为E98泛化增强best71；新建AdamW状态，保留数据增强、Dropout和原更新保护。没有根据Q重选权重；由单次流水线随后执行固定dev500，测试封存。\n',encoding='utf-8')
            beat(status='complete',stage='training_complete_ready_for_dev500')
        except Halt:beat(status='user_stop',stage='stopped_no_next_stage')
        except BaseException as exc:
            save(RUN/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),test_read=False))
            beat(status='error',stage='failed_no_retry');raise
        finally:
            closed.set();thread.join(timeout=6);signal.signal(signal.SIGINT,previous)
            gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
