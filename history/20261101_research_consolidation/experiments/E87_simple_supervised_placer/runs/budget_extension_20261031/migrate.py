"""Explicit user-authorized +1h; preserve all optimization state and original evidence."""
import sys,json,copy,shutil,time,importlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.E87_simple_supervised_placer.io import HERE,sha,save,checkpoint,sources,snapshot,exclusive
from experiments.E87_simple_supervised_placer.preflight import verify
from experiments.E87_simple_supervised_placer import loss_config,train

M=Path(__file__).resolve().parent
RUN=HERE/'runs/20261030_simple_seed0'

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)

def main():
    begin=time.perf_counter()
    with exclusive(HERE/'runs/pipeline.lock'),exclusive(RUN.with_suffix('.lock')):
        if (M/'migration.json').exists():raise ValueError('This one-hour extension already has a migration record')
        original_check=verify();before_sources=sources()
        original_cal=json.loads((HERE/'runs/registered_loss.json').read_text(encoding='utf-8'))
        assert original_cal['sources']==before_sources
        old_states={name:torch.load(RUN/f'checkpoints/{name}.pt',map_location='cpu',weights_only=False) for name in ('best','latest')}
        assert not old_states['latest']['finished'] and old_states['latest']['status']=='user_stop'
        for old in old_states.values():assert old['identity']['sources']==before_sources and old['identity']['max_seconds']==3600
        originals={}
        paths=[HERE/'train.py',HERE/'pipeline.py',HERE/'plan.json',HERE/'runs/registered_loss.json',HERE/'runs/preflight/results.json',
               RUN/'checkpoints/best.pt',RUN/'checkpoints/latest.pt']
        for path in paths:
            dest=M/'before'/path.relative_to(HERE);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise ValueError('Refuse overwriting migration evidence: '+str(dest))
            shutil.copy2(path,dest);originals[path]=dest
        save(M/'journal.json',dict(status='prepared',authorization='2026-10-31 user: 之前训练太慢了，训练预算再给1h',
            old_cap_seconds=3600,new_cap_seconds=7200,scope='simple arm only; no LR/Loss/data/model changes',
            before_hashes={p.relative_to(HERE).as_posix():sha(p) for p in paths}))
        try:
            p=HERE/'train.py';text=p.read_text(encoding='utf-8')
            text=replace_once(text,"if args.max_hours > 1 or args.max_epochs > 200 or args.batch_size != 32:",
                "if args.max_hours > (2 if args.arm=='simple' else 1) or args.max_epochs > 200 or args.batch_size != 32:")
            text=replace_once(text,"E87 is limited to one hour/200 epochs per arm, batch32","E87 limits: simple2h, legacy1h; 200 epochs and batch32")
            p.write_text(text,encoding='utf-8')
            p=HERE/'pipeline.py';text=p.read_text(encoding='utf-8')
            text=replace_once(text,"'--max-hours','1'","'--max-hours',str(2 if a.arm=='simple' else 1)")
            p.write_text(text,encoding='utf-8')
            p=HERE/'plan.json';plan=json.loads(p.read_text(encoding='utf-8'))
            plan.update(simple_authorized_extension_hours=1,simple_total_hours=2,new_training_stage_cap_seconds=25200,
                budget_extension_authorization='2026-10-31 user explicitly granted E87 simple an additional1h; preserve elapsed time')
            save(p,plan)
            after_sources=sources();allowed={f'experiments/E87_simple_supervised_placer/{n}' for n in ('train.py','pipeline.py','plan.json')}
            changed={k for k in set(before_sources)|set(after_sources) if before_sources.get(k)!=after_sources.get(k)}
            assert changed==allowed,changed
            # Re-register identical calibrated numbers against the budget-only source delta.
            calibration=copy.deepcopy(original_cal);calibration['sources']=after_sources
            calibration['budget_only_migration']='runs/budget_extension_20261031/migration.json'
            save(HERE/'runs/registered_loss.json',calibration)
            for key in original_cal:
                if key!='sources':equal(original_cal[key],calibration[key])
            importlib.reload(train)
            c=loss_config.load_coefficients(HERE/'runs/registered_loss.json')
            args=train.parser().parse_args(['--arm','simple','--output',str(RUN),'--max-hours','2','--scheduler-start-epoch','0','--resume'])
            train.seed_all(args.seed);new_identity=train.make_identity(args);new_identity['calibration']=c
            identity_differences={k for k in set(new_identity)|set(old_states['latest']['identity']) if new_identity.get(k)!=old_states['latest']['identity'].get(k)}
            assert identity_differences=={'sources','max_seconds','calibration'},identity_differences
            assert new_identity['scheduler_start_epoch']==0 and new_identity['max_seconds']==7200
            # Exercise rejection boundaries before any initialization or parameter update.
            for arm,hours in (('simple','3'),('legacy','1.5')):
                invalid=train.parser().parse_args(['--arm',arm,'--output',str(HERE/'runs/preflight/budget_rejection'),'--max-hours',hours])
                try:train._run(invalid,implementation_check=True)
                except ValueError as error:assert 'E87 limits' in str(error)
                else:raise AssertionError('Budget limit not enforced')
            for name,old in old_states.items():
                state=copy.copy(old);state['identity']=copy.deepcopy(new_identity)
                checkpoint(RUN/f'checkpoints/{name}.pt',state)
                restored=torch.load(RUN/f'checkpoints/{name}.pt',map_location='cpu',weights_only=False)
                assert restored.keys()==old.keys()
                for key in old:
                    if key!='identity':equal(old[key],restored[key])
            inherited=copy.deepcopy(original_check);inherited['sources']=after_sources
            inherited['verification_method']='Original actual five-case algorithm checks inherited; only budget/source identity delta checked here, no new optimization updates'
            inherited['budget_only_source_verification']=dict(artifact='runs/budget_extension_20261031/migration.json',
                prior_result='runs/budget_extension_20261031/before/runs/preflight/results.json',
                allowed_changed_files=sorted(allowed),all_other_sources_byte_identical=True,
                calibrated_values_unchanged=True,checkpoint_payloads_unchanged_except_identity=True,
                simple_limit_seconds=7200,legacy_limit_seconds=3600,budget_rejection_checks_passed=True)
            save(HERE/'runs/preflight/results.json',inherited)
            verify()
            latest=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            assert latest['identity']==new_identity
            snapshot(M/'source_after.zip',dict(sources=after_sources))
            save(M/'active_identity.json',new_identity)
            elapsed=max(float(latest['elapsed_seconds']),json.loads((RUN/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds'])
            save(M/'migration.json',dict(status='complete',registered_utc=datetime.now(timezone.utc).isoformat(),
                authorization='User explicitly granted one additional hour to the active E87 training branch',
                old_cap_seconds=3600,new_cap_seconds=7200,elapsed_seconds_preserved=elapsed,remaining_seconds=max(0,7200-elapsed),
                step_preserved=latest['progress']['step'],epoch_preserved=latest['progress']['epoch'],cursor_preserved=latest['progress']['cursor'],
                model_optimizer_rng_scheduler_progress_unchanged=True,calibrated_coefficients_unchanged=True,
                no_learning_rate_protection_restored=True,all_algorithm_data_loss_sources_unchanged=True,
                changed_source_files=sorted(changed),identity_differences=sorted(identity_differences),
                preserved_original_run_config_and_logs=True,original_checkpoints_preserved_under='before/',
                elapsed_migration_seconds=time.perf_counter()-begin,parameter_updates=0,test_read=False))
            save(M/'journal.json',dict(status='complete',record='migration.json'))
            print(json.dumps(dict(status='complete',total_cap_hours=2,elapsed_seconds=elapsed,remaining_seconds=7200-elapsed,
                step=latest['progress']['step'],optimization_state_unchanged=True),ensure_ascii=False),flush=True)
        except BaseException:
            for path,backup in originals.items():shutil.copy2(backup,path)
            save(M/'journal.json',dict(status='rolled_back',elapsed_seconds=time.perf_counter()-begin))
            raise

if __name__=='__main__':main()
