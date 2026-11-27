"""One authorized study run: train -> unique-best flash500 -> report/drafts -> exit."""
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .io import HERE,ROOT,save,sha,exclusive,sources
from .loss_config import configuration
from .final import summarize

STUDY=HERE/'study.json'
CONTROL=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc)

def report(folder,study):
    result=read(folder/'terminal/results.json');summary=result['summary'];target=study['target']
    checks={'weighted_Q':summary['weighted_Q']<target['weighted_Q_lt'],
            'hard_legal':summary['hard_legal']==500 and summary['errors']==0,
            'p95':summary['p95_Q']<=target['p95_Q_max'],
            'worst5pct':summary['worst5pct_mean_Q']<=target['worst5pct_mean_Q_max'],
            'max_Q':summary['max_Q']<=target['max_Q_max']+1e-9}
    base=read(CONTROL/'terminal/results.json')['records'];current=result['records']
    if [r['member'] for r in base]!=[r['member'] for r in current]:raise ValueError('Evaluation panels differ')
    from .paired import paired
    value={'summary':summary,'targets':target,'checks':checks,'numerical_pass':all(checks.values()),
        'paired_E42':paired(current,base),'raw_full_dev':read(folder/'full_dev.json')['losses'],
        'draft_visual_status':'awaiting_agent_inspection','test_read':False,'single_seed':True,
        'scope':'adaptive exposed development500, not hidden or blind test; no checkpoint selection by Q'}
    save(folder/'acceptance.json',value)
    (folder/'analysis.md').write_text('# E53 固定Loss候选\n\n'+f'加权Q {summary["weighted_Q"]:.9f}，硬合法 {summary["hard_legal"]}/500，P95 {summary["p95_Q"]:.6f}，最差5% {summary["worst5pct_mean_Q"]:.6f}，最大Q {summary["max_Q"]:.6f}。\n\n'+f'数值门槛：{checks}；全部通过={all(checks.values())}。草稿视觉核查尚未完成，不仅凭Q宣布成功。\n',encoding='utf-8')
    return value

def run(arm,seed):
    name=f'20261026_{arm}_seed{seed}';folder=HERE/'runs'/name;control=HERE/'runs'/(name+'_control')
    control.mkdir(parents=True,exist_ok=True)
    with exclusive(HERE/'runs/study_worker.lock'):
        study=read(STUDY);deadline=datetime.fromisoformat(study['deadline_utc'])
        if sha(HERE/'model.py')!=study['model_sha256'] or sha(HERE/'solver.py')!=study['flash_solver_sha256']:
            raise ValueError('Frozen architecture or flash changed')
        for source_name,digest in study['frozen_core_sources'].items():
            if sha(HERE/source_name)!=digest:raise ValueError('Frozen core changed: '+source_name)
        record=next((r for r in study['runs'] if r['name']==name),None)
        if record and record['status']=='complete':print('Already complete; no repeated evaluation.');return
        if now()>=deadline:raise RuntimeError('Global12h deadline reached; no further work')
        if record is None:
            if study['training_runs_started']>=study['max_training_runs']:raise RuntimeError('Eight-training limit reached')
            record={'name':name,'arm':arm,'seed':seed,'output':folder.relative_to(ROOT).as_posix(),
                'status':'starting','created_utc':now().isoformat(),'loss':configuration(arm),
                'max_hours':min(2.,max(.01,(deadline-now()).total_seconds()/3600-.05)),
                'source_sha256':sources()}
            study['runs'].append(record);study['training_runs_started']+=1
        study['status']='running';save(STUDY,study)
        stopped=[False];old=signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
        def stop():return stopped[0] or (HERE/'runs/STOP').exists() or (control/'STOP').exists() or now()>=deadline
        def stage(kind,argv):
            if stop():return False
            record.update(status=kind,active_command=argv,updated_utc=now().isoformat());save(STUDY,study)
            env=dict(os.environ,E53_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
            with (control/'console.log').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'stage':kind,'argv':argv,'utc':now().isoformat()})+'\n');log.flush()
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                save(control/'active.json',{'stage':kind,'pid':child.pid,'argv':argv})
                while child.poll() is None:
                    if stop():
                        folder.mkdir(exist_ok=True);(folder/'STOP').touch()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            (control/'active.json').unlink(missing_ok=True)
            if child.returncode:raise RuntimeError(f'{kind} exited {child.returncode}; see {control}')
            return not stop()
        try:
            train=[sys.executable,'-u','-B','-m','experiments.E53_loss_geometry.train','--arm',arm,'--seed',str(seed),
                '--output',str(folder),'--device','cuda','--batch-size','32','--engine','cuda_graph',
                '--max-hours',str(record['max_hours']),'--max-epochs','200','--scheduler-start-epoch','15']
            if (folder/'checkpoints/latest.pt').exists():train.append('--resume')
            if not (folder/'results.json').exists() or read(folder/'results.json')['status'] not in ('early_stop','max_epochs','budget'):
                if not stage('training',train):record['status']='stopped';return
            if not stage('evaluation',[sys.executable,'-u','-B','-m','experiments.E53_loss_geometry.final','--output',str(folder)]):
                record['status']='stopped';return
            if not (folder/'terminal/results.json').exists():raise RuntimeError('Incomplete terminal500; do not report success')
            accepted=report(folder,study)
            if not stage('draft_images',[sys.executable,'-u','-B','-m','experiments.E53_loss_geometry.render_drafts','--output',str(folder)]):
                record['status']='stopped';return
            record.update(status='complete',completed_utc=now().isoformat(),summary=accepted['summary'],
                numerical_pass=accepted['numerical_pass'],draft_visual_status='awaiting_agent_inspection')
            study['status']='awaiting_analysis';print(json.dumps(record,ensure_ascii=False),flush=True)
        except BaseException as error:
            record.update(status='error',error=repr(error),updated_utc=now().isoformat());study['status']='needs_analysis'
            save(control/'failure.json',{'error':repr(error),'utc':now().isoformat()});raise
        finally:
            record.pop('active_command',None);save(STUDY,study);signal.signal(signal.SIGINT,old)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arm',default='geometry');ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args();configuration(args.arm);run(args.arm,args.seed)
