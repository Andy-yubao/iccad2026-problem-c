"""One registered round2 candidate, preflight then train then unique full terminal."""
import datetime,json,subprocess,sys,time
from .round2_candidate4 import HERE,ROOT,OUTPUT,PREFLIGHT,source_identity
from .train_io.io import save,exclusive

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stopped():
    study=read(HERE/'round2_study.json')
    return (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(study['deadline_utc'])

def main():
    with exclusive(HERE/'runs/round2_candidate4_pipeline.lock'):
        control=HERE/'runs/20261102_round2_current_control/results.json'
        while not stopped():
            if (control.parent/'error.json').exists():raise RuntimeError('Current control failed; diagnose before training')
            if control.exists() and read(control).get('finished'):break
            time.sleep(2)
        if stopped():return
        prefix=[sys.executable,'-u','-B','-m']
        if (PREFLIGHT/'error.json').exists():raise RuntimeError('Preflight error requires diagnosis; no automatic retry')
        if not (PREFLIGHT/'results.json').exists():subprocess.run([*prefix,'experiments.E97_gpu_hybrid_solver.round2_candidate4','--preflight'],cwd=ROOT,check=True)
        check=read(PREFLIGHT/'results.json');assert check['status']=='passed' and check['source_sha256']==source_identity()
        if stopped():return
        if not (OUTPUT/'results.json').exists():
            command=[*prefix,'experiments.E97_gpu_hybrid_solver.round2_candidate4','--execute']
            if (OUTPUT/'checkpoints/latest.pt').exists():command.append('--resume')
            subprocess.run(command,cwd=ROOT,check=True)
        if stopped() or not (OUTPUT/'results.json').exists() or not read(OUTPUT/'results.json').get('finished'):return
        if (OUTPUT/'evaluation/error.json').exists():raise RuntimeError('Terminal error requires diagnosis, no automatic retry')
        if not (OUTPUT/'evaluation/results.json').exists():subprocess.run([*prefix,'experiments.E97_gpu_hybrid_solver.round2_candidate4_evaluate'],cwd=ROOT,check=True)
        result=OUTPUT/'evaluation/results.json';complete=result.exists() and read(result).get('finished')
        study=read(HERE/'round2_study.json');study.update(current_stage='inspect round2 candidate4 full result' if complete else 'round2 candidate4 terminal incomplete')
        for r in study['candidate_ledger']:
            if r['candidate']==4:r['status']='complete' if complete else 'evaluation_incomplete'
        save(HERE/'round2_study.json',study);save(HERE/'round2_state.json',dict(status='candidate4_complete' if complete else 'candidate4_evaluation_incomplete',active_run=OUTPUT.relative_to(HERE).as_posix(),formal_candidates_started=study['formal_candidates_started'],max_formal_training_candidates=8,test_read=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(HERE/'runs/round2_candidate4_pipeline_error.json',dict(error=repr(error),no_automatic_retry=True,test_read=False));raise
