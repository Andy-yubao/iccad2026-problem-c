"""One already-preflighted student training and its unique terminal, no extra arm."""
import datetime,json,subprocess,sys
from .candidate2 import HERE,ROOT,OUTPUT,PREFLIGHT,source_identity
from .train_io.io import exclusive,save

def main():
    with exclusive(HERE/'runs/candidate2_pipeline.lock'):
        study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(study['deadline_utc']) or (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():return
        check=json.loads((PREFLIGHT/'results.json').read_text(encoding='utf-8'));assert check['status']=='passed' and check['source_sha256']==source_identity()
        result=OUTPUT/'results.json';prefix=[sys.executable,'-u','-B','-m']
        if not result.exists() or not json.loads(result.read_text(encoding='utf-8')).get('finished'):
            command=[*prefix,'experiments.E97_gpu_hybrid_solver.candidate2','--execute']
            if (OUTPUT/'checkpoints/latest.pt').exists():command.append('--resume')
            subprocess.run(command,cwd=ROOT,check=True)
        if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists() or not result.exists() or not json.loads(result.read_text(encoding='utf-8')).get('finished'):return
        if not (OUTPUT/'checkpoints/best.pt').exists():return
        if (OUTPUT/'evaluation/error.json').exists():raise RuntimeError('Evaluation error requires diagnosis, no automatic retry')
        if not (OUTPUT/'evaluation/results.json').exists():subprocess.run([*prefix,'experiments.E97_gpu_hybrid_solver.evaluate_candidate2'],cwd=ROOT,check=True)
        evaluation=OUTPUT/'evaluation/results.json'
        complete=evaluation.exists() and json.loads(evaluation.read_text(encoding='utf-8')).get('finished') and (OUTPUT/'evaluation/gallery/index.html').exists()
        status='candidate2_complete' if complete else 'candidate2_evaluation_incomplete'
        study=json.loads((HERE/'study.json').read_text(encoding='utf-8'));study.update(status=status,current_stage='inspect candidate2 result' if complete else 'candidate2 evaluation remains incomplete')
        for x in study['candidate_ledger']:
            if x['candidate']==2:x.update(status='complete' if complete else 'evaluation_incomplete',output=OUTPUT.relative_to(HERE).as_posix())
        save(HERE/'study.json',study);save(HERE/'state.json',dict(status=status,active_run=OUTPUT.relative_to(HERE).as_posix(),formal_candidates_started=study['formal_candidates_started'],max_formal_training_candidates=8,test_read=False))

if __name__=='__main__':main()
