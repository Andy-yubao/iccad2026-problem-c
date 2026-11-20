"""Exactly two authorized budget-reallocation runs and their terminal outputs."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
OWNER=Path(__file__).resolve().parent
NAMES=('E112_deep_geometry_budget','E113_deep_head_budget')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,path)


def now():return datetime.now(timezone.utc).isoformat()


def main():
    if (OWNER/'queue_state.json').exists():raise FileExistsError('Already dispatched; no automatic queue restart')
    jobs=[]
    for name in NAMES:
        for phase,module in [('train','train'),('dev500','evaluation.run'),('five_case_display','diagnostics.layer_trace')]:
            jobs.append(dict(name=name,phase=phase,module=f'experiments.{name}.{module}'))
    jobs.append(dict(name='combined_display',phase='publish',module='visualization.layer_budget_viewer'))
    state=dict(status='running',pid=os.getpid(),started_at=now(),jobs=jobs,completed=[],active=None,
               scope='Only E112/E113 scratch training, each frozen-best dev500 plus same-five display, then combined viewer; A reused, no test')
    def beat(**kw):state.update(kw);state['updated_at']=now();save(OWNER/'queue_state.json',state)
    try:
        for index,job in enumerate(jobs,1):
            if (OWNER/'QUEUE_STOP').exists():raise InterruptedError('Queue STOP')
            folder=ROOT/'experiments'/job['name'] if job['name'] in NAMES else OWNER
            if (folder/'STOP').exists():raise InterruptedError('Candidate STOP')
            logs=folder/'launch_logs';logs.mkdir(exist_ok=True)
            tag=job['phase']
            with (logs/(tag+'.stdout.log')).open('x',encoding='utf-8') as stdout,(logs/(tag+'.stderr.log')).open('x',encoding='utf-8') as stderr:
                child=subprocess.Popen([sys.executable,'-u','-B','-m',job['module']],cwd=ROOT,stdout=stdout,stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                beat(active=dict(**job,index=index,total=len(jobs),pid=child.pid),active_started_at=now())
                while child.poll() is None:
                    if (OWNER/'QUEUE_STOP').exists():(folder/'STOP').write_text('User queue stop; preserve state, no next task\n',encoding='utf-8')
                    time.sleep(5)
                if child.returncode!=0:raise RuntimeError(f"{job['module']} exited {child.returncode}; no retry or next task")
            if job['name'] in NAMES:
                plan=read(folder/'plan.json');run=folder/'runs'/plan['run']
                if job['phase']=='train':
                    result=read(run/'results.json')
                    if result['status'] not in ('early_stop','budget','max_epochs') or result.get('error'):
                        raise RuntimeError('Training not normally completed; no continuation')
                elif job['phase']=='dev500':
                    result=read(run/'evaluation_depth_dev500/results.json')
                    if result['status']!='complete':raise RuntimeError('Incomplete dev500')
                else:
                    result=read(run/'layer_trace5_20261107/verification.json')
                    if result['status']!='complete':raise RuntimeError('Incomplete five-case trace')
            state['completed'].append(dict(**job,completed_at=now()));beat(active=None)
        control=ROOT/'experiments/E109_deep_supervision_shared/runs/20261107_shared_seed0/evaluation_depth_dev500/results.json'
        outcomes={name:read(ROOT/'experiments'/name/'runs'/read(ROOT/'experiments'/name/'plan.json')['run']/'evaluation_depth_dev500/results.json') for name in NAMES}
        save(OWNER/'comparison.json',dict(status='complete',models=outcomes,baseline_A=read(control),
            baseline_A_results_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
            caveat='Single seed, existing A control. Near-equal parameter count; different FLOPs/feedback count and loss exit count. Geometry recomputation registered; no test and no Q checkpoint selection.'))
        beat(status='complete',active=None,completed_at=now())
    except BaseException as error:
        beat(status='user_stop' if isinstance(error,InterruptedError) else 'needs_attention',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
