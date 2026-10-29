"""Validate original E77 checkpoint and consume the user's new resume authorization."""
import json
from datetime import datetime,timezone
import torch
from experiments.E77_compact_refinement.io import HERE,save,sources,exclusive,sha
from experiments.E77_compact_refinement.pipeline import OUTPUT
from experiments.E77_compact_refinement.initialization import verify_protected


def main():
    with exclusive(HERE/'runs/pipeline.lock'),exclusive(OUTPUT.with_suffix('.lock')):
        if (OUTPUT/'STOP').exists():raise ValueError('Unrecognized run-level STOP; inspect before proceeding')
        latest=OUTPUT/'checkpoints/latest.pt'
        state=torch.load(latest,map_location='cpu',weights_only=False)
        config=json.loads((OUTPUT/'config.json').read_text(encoding='utf-8'))
        preflight=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
        current=sources()
        assert state['identity']==config['identity']
        assert state['identity']['sources']==current==preflight['sources'] and preflight['status']=='passed'
        assert not state['finished']
        assert torch.cuda.is_available()
        protected=verify_protected()
        budget=json.loads((OUTPUT/'budget.json').read_text(encoding='utf-8'))
        prior=max(state['elapsed_seconds'],budget['elapsed_seconds'])
        assert prior<state['identity']['max_seconds']
        stop=HERE/'runs/STOP'
        old_stop=stop.read_text(encoding='utf-8') if stop.exists() else None
        evidence={'authorized_utc':datetime.now(timezone.utc).isoformat(),
            'user_message':'请你先启动E77的训练',
            'checkpoint':str(latest),'checkpoint_sha256':sha(latest),
            'restored_step':state['progress']['step'],'restored_epoch':state['progress']['epoch'],
            'prior_budget_seconds':prior,'remaining_original_budget_seconds':state['identity']['max_seconds']-prior,
            'checkpoint_and_production_sources_unchanged':True,'exclusive_locks_acquired':True,
            'only_pause_stop_removed':str(stop),'previous_stop_content':old_stop,
            'pause_wall_time_not_charged_as_training':True,'budget_not_reset':True,'test_read':False}
        save(HERE/'runs/recovery/resume_authorization_20261029.json',evidence)
        if stop.exists():stop.unlink()
        assert verify_protected()==protected
        print(json.dumps({k:evidence[k] for k in ('restored_step','restored_epoch','prior_budget_seconds','remaining_original_budget_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
