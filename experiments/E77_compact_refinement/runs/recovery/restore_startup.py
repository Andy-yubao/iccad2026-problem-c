"""Reconstruct a zero-update checkpoint after process loss before first save.

This is not recovery of any learned state: none was durably recorded. Keep
the source/config/snapshot/invocation identity and charge the entire unknown
interruption interval to the ORIGINAL two-hour candidate budget.
"""
import json,os
from datetime import datetime,timezone
import torch
from experiments.E77_compact_refinement.io import HERE,ROOT,sources,sha,save,checkpoint,exclusive
from experiments.E77_compact_refinement.pipeline import OUTPUT
from experiments.E77_compact_refinement.train import parser,make_identity,seed_all,PlateauController,fresh_progress,rng_state,WEIGHT_DECAY
from experiments.E77_compact_refinement.model import GeometryModel
from experiments.E77_compact_refinement.arms import model_config
from experiments.E77_compact_refinement.initialization import load,configure_trainability,verify_protected


def main():
    with exclusive(HERE/'runs/pipeline.lock'),exclusive(OUTPUT.with_suffix('.lock')):
        assert not (OUTPUT/'checkpoints').exists()
        assert not list(OUTPUT.rglob('*.pt'))
        assert not (OUTPUT/'progress.json').exists() and not (OUTPUT/'budget.json').exists()
        assert not (OUTPUT/'results.json').exists()
        original=json.loads((OUTPUT/'config.json').read_text(encoding='utf-8'))
        calls=json.loads((OUTPUT/'invocations.json').read_text(encoding='utf-8'))
        assert len(calls)==1 and not calls[0]['resume'] and calls[0]['prior_seconds']==0
        assert original['identity']['sources']==sources()
        # The original invocation reached config/source snapshot, but no
        # training output. Never infer how long the actual dead process lived.
        start=datetime.fromisoformat(calls[0]['started_utc'].replace('Z','+00:00'))
        now=datetime.now(timezone.utc);charged=(now-start).total_seconds()
        assert 0<charged<original['identity']['max_seconds']
        args=parser().parse_args(['--output',str(OUTPUT),'--arm','topology','--max-hours','2','--max-epochs','200'])
        seed_all(0);identity=make_identity(args)
        assert identity==original['identity'], 'Original source/data/device/budget identity must match exactly'
        protected=verify_protected();seed_all(0)
        model=configure_trainability(GeometryModel(**model_config('topology')).to('cuda'))
        source=load(model,model_config('topology'))
        assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in model.source.state_dict().items())
        assert torch.count_nonzero(model.corrector.output[-1].weight)==0
        assert torch.count_nonzero(model.corrector.output[-1].bias)==0
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=WEIGHT_DECAY)
        scheduler=PlateauController(optimizer,start_epoch=15)
        # No prior train shuffle/update existed. Epoch0 monitoring remains pending.
        progress=fresh_progress(32000)
        charged=(datetime.now(timezone.utc)-start).total_seconds()
        state={'model':model.state_dict(),'model_config':model_config('topology'),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(),
            'progress':progress,'identity':identity,'elapsed_seconds':charged,
            'status':'startup_interruption_zero_updates_reconstructed','finished':False,'graph_step_control':None}
        checkpoint(OUTPUT/'checkpoints/latest.pt',state)
        save(OUTPUT/'budget.json',{'elapsed_seconds':charged,'startup_interruption_charged_seconds':charged})
        assert verify_protected()==protected
        save(HERE/'runs/recovery/startup_interruption.json',{
            'status':'zero_update_checkpoint_reconstructed','observed_utc':now.isoformat(),
            'original_started_utc':calls[0]['started_utc'],'original_controller_pid':48444,'original_train_pid':20016,
            'actual_processes_absent_checked':True,'locks_acquired_exclusively':True,
            'original_tool_session':53415,'tool_session_missing':True,
            'death_time_and_cause':'Unknown; no exception or first checkpoint. Do not claim a numerical/model failure.',
            'original_sources_config_and_source_snapshot_retained':True,
            'snapshot_sha256':sha(OUTPUT/'source_snapshot.zip'),'original_identity_exact':True,
            'optimizer_updates_recovered':0,'source_E76_bit_exact':True,'new_head_zero_init':True,
            'charged_interruption_seconds':charged,'remaining_original_budget_seconds':7200-charged,
            'budget_policy':'Conservatively charge whole unknown interval including downtime; do not reset original2h.',
            'rng_policy':'Original seed0 initialization and fresh zero-step optimizer; no claim of restoring any unrecorded update/RNG.',
            'restored_by_pid':os.getpid(),'test_read':False})
        print('Zero-update recovery; budget charged seconds',charged,'remaining',7200-charged,flush=True)


if __name__=='__main__':main()
