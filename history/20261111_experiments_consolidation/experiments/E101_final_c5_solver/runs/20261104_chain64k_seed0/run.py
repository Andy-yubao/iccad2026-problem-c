"""Authorized full historical chain, all training stages normal early-stop only."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,datetime,shutil,zipfile
import torch
from prepare import ROOT,BUNDLE,read,save,sha,link
PYTHON=ROOT/'.conda/envs/iccad2026/python.exe'
E42='E42_joint_regularization';E74='E74_collision_32k_posttrain'
R42=BUNDLE/'s42/experiments'/E42/'runs/replay_seed0'
R74=BUNDLE/'s74/experiments'/E74/'runs/replay_seed0'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stopped():return (BUNDLE/'STOP').exists()
def stage(name,cwd,args,stop_paths=()):
    if stopped():raise InterruptedError('STOP blocks subsequent stages')
    print('Starting',name,utc(),flush=True)
    env=dict(os.environ);env.pop('E16_REGISTRY',None)
    if name in ('E42','E74'):
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]='4'
    with (BUNDLE/(name+'.log')).open('a',encoding='utf-8') as log:
        p=subprocess.Popen([str(PYTHON),'-u','-B',*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save(BUNDLE/'state.json',dict(status='running',stage=name,pipeline_pid=os.getpid(),child_pid=p.pid,updated_utc=utc(),test_read=name=='test2000'))
        while p.poll() is None:
            if stopped():
                for path in stop_paths:path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            try:p.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
    if stopped():raise InterruptedError('User STOP; current stage saved, subsequent stages blocked')
    if p.returncode:raise RuntimeError(f'{name} exited{p.returncode}; see{name}.log; no automatic retry')

def bind_e74():
    result=read(R42/'results.json');assert result['status']=='early_stop','E42 must normally early stop before E74 starts'
    selected=R42/'checkpoints/best.pt';state=torch.load(selected,map_location='cpu',weights_only=False);here=BUNDLE/'s74/experiments'/E74
    alias=BUNDLE/'s74/experiments'/E42/'runs/replay_seed0/checkpoints/best.pt';link(selected,alias)
    if not (here/'initialization.py.frozen_original').exists():shutil.copy2(here/'initialization.py',here/'initialization.py.frozen_original')
    p=read(here/'provenance.json');p.update(initialization_checkpoint=alias.relative_to(BUNDLE/'s74').as_posix(),initialization_sha256=sha(selected),source_best_epoch=state['progress']['best_epoch'],source_best_step=state['progress']['step'],pretraining_elapsed_seconds=state['elapsed_seconds'],source_completed_epochs=result['epochs'],source_completed_elapsed_seconds=result['elapsed_seconds'])
    save(here/'provenance.json',p)
    text=(here/'initialization.py.frozen_original').read_text(encoding='utf-8');text=text.replace("'source_best_epoch': 51, 'source_best_step': 25500,","'source_best_epoch': p['source_best_epoch'], 'source_best_step': p['source_best_step'],");text=text.replace("'budget': 'new stage at most 2h/200 epochs; inherited pretraining time recorded separately'","'budget': 'normal early stop only, no time or epoch cap; full pretraining budget retained'")
    text=text.replace('E42 best51','new E42 unique own-proxy best')
    (here/'initialization.py').write_text(text,encoding='utf-8');save(BUNDLE/'E42_to_E74_binding.json',dict(selected_E42_sha256=sha(selected),best_epoch=state['progress']['best_epoch'],best_step=state['progress']['step'],transform='original exact96to192 widening; graph-local mappings frozen; fresh AdamW and scheduler',E42_status=result['status']))

def main(resume=False):
    from experiments.E101_final_c5_solver.common import exclusive
    with exclusive(BUNDLE/'pipeline.lock'):
        if stopped():raise InterruptedError('Remove own STOP only on explicit resume')
        expected=read(BUNDLE/'controller_sources.json')
        assert all(sha(BUNDLE/name)==digest for name,digest in expected.items()),'Registered pipeline controller changed'
        previous=ROOT/'experiments/E101_final_c5_solver/runs/20261103_replay/state.json'
        while read(previous)['status']=='running':
            if stopped():raise InterruptedError('STOP while waiting for prior evaluation')
            save(BUNDLE/'state.json',dict(status='waiting',stage='waiting_for_previous_evaluation',pipeline_pid=os.getpid(),updated_utc=utc(),test_read=False))
            time.sleep(5)
        assert read(previous)['status']=='complete','Previous evaluation failed; avoid overlapping or silently skipping it'
        if not (BUNDLE/'preflight.json').exists():stage('preflight',BUNDLE,[str(BUNDLE/'preflight.py')],[BUNDLE/'sf/experiments/E101_final_c5_solver/runs/replay_F_seed0/STOP'])
        assert read(BUNDLE/'preflight.json')['status']=='passed'
        if not (R42/'results.json').exists() or read(R42/'results.json')['status']!='early_stop':
            extra=['--resume'] if resume and (R42/'checkpoints/latest.pt').exists() else []
            stage('E42',BUNDLE/'s42',['-m',f'experiments.{E42}.train','--arm','rectangle_ab','--output',str(R42),'--device','cuda','--seed','0','--batch-size','32','--scheduler-start-epoch','15',*extra],[R42/'STOP'])
        if not (R74/'checkpoints/latest.pt').exists():bind_e74()
        if not (R74/'results.json').exists() or read(R74/'results.json')['status']!='early_stop':
            extra=['--resume'] if resume and (R74/'checkpoints/latest.pt').exists() else []
            stage('E74',BUNDLE/'s74',['-m',f'experiments.{E74}.train','--arm','topology','--output',str(R74),'--seed','0','--batch-size','32','--engine','cuda_graph',*extra],[R74/'STOP'])
        assert read(R74/'results.json')['status']=='early_stop'
        f=read(BUNDLE/'F_runtime.json');fr=Path(f['root']);fh=Path(f['here']);run=Path(f['run']);link(R74/'checkpoints/best.pt',run/'L/checkpoints/best.pt');shutil.copy2(R74/'results.json',run/'L/results.json')
        if not (run/'teacher/results.json').exists():stage('F_prepare',fr,[str(BUNDLE/'prepare_F_labels.py')],[run/'STOP'])
        if not (run/'F/results.json').exists() or read(run/'F/results.json')['status']!='early_stop':
            extra=['--resume'] if resume and (run/'F/checkpoints/latest.pt').exists() else []
            stage('F',fr,['-m','experiments.E101_final_c5_solver.train','F',*extra],[run/'STOP'])
        assert read(run/'F/results.json')['status']=='early_stop'
        if not (run/'frozen_selection.json').exists():stage('freeze',fr,[str(BUNDLE/'freeze_F.py')],[run/'STOP'])
        for role,name in [('dev_monitor500','dev500'),('local_test','test2000')]:
            if not (run/'evaluation'/role/'results.json').exists():stage(name,fr,['-m','experiments.E101_final_c5_solver.evaluate',role],[run/'STOP'])
        selection=read(run/'frozen_selection.json')
        report=dict(status='complete',training={name:read(path/'results.json') for name,path in [('E42',R42),('E74',R74),('F',run/'F')]},selection=selection,evaluation={role:read(run/'evaluation'/role/'results.json') for role in ('dev_monitor500','local_test')},delivery_model=str(fh/'delivery/model.pt'),test_scope='Repeated registered2000 already exposed by E101; no test feedback',completed_utc=utc())
        assert all(v['status']=='complete' for v in report['evaluation'].values())
        save(BUNDLE/'results.json',report)
        lines=['# E42 → E74 → C5 F：最终运行结果','','全部阶段按原早停规则结束；唯一分阶段Loss最佳组合在评价前冻结。测试为已曝光2000例的重复评价，未用于调参或改选权重。','',f"模型：{report['delivery_model']}",f"SHA256：{selection['checkpoint_sha256']}",'','| 面板 | 硬合法 | 加权Q | 完整均时(s) | P95(s) | 回退数 |','|---|---:|---:|---:|---:|---:|']
        for role,value in report['evaluation'].items():
            s=value['summary'];lines.append(f"| {role} | {s['hard_legal']}/{s['count']} | {s['weighted_Q']} | {s['mean_complete_seconds']} | {s['p95_complete_seconds']} | {s['fallbacks']} |")
        (BUNDLE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        save(BUNDLE/'state.json',dict(status='complete',stage='full_training_and_repeated_test2000_complete',updated_utc=utc(),test_read=True,no_test_feedback=True))

if __name__=='__main__':
    sys.path.insert(0,str(ROOT));ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    try:main(args.resume)
    except BaseException as e:save(BUNDLE/'state.json',dict(status='user_stop' if isinstance(e,InterruptedError) else 'error',error=repr(e),updated_utc=utc(),no_automatic_retry=True));raise
