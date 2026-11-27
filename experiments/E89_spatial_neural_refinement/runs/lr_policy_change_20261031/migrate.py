"""Paused E89 only: remove LR protection without replaying history or training."""
from pathlib import Path
import copy,json,sys,shutil,zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from experiments.E89_spatial_neural_refinement.io import HERE,sha,save,sources,exclusive,checkpoint

ARCHIVE=Path(__file__).resolve().parent
RUN=HERE/'runs/20261031_spatial_seed0'


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def main():
    assert (HERE/'runs/STOP').exists()
    assert not (RUN/'frozen.json').exists(), 'Do not migrate an already terminal-evaluated candidate'
    assert not (ARCHIVE/'migration.json').exists(), 'Migration already completed'
    with exclusive(HERE/'runs/pipeline.lock'),exclusive(RUN.with_suffix('.lock')):
        latest=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert latest['status']=='user_stop' and not latest['finished']
        old_sources=sources()
        old_pre=json.loads((HERE/'runs/training_preflight/results.json').read_text(encoding='utf-8'))
        assert old_pre['sources']==old_sources
        before=ARCHIVE/'before';before.mkdir(exist_ok=False)
        for name in ('best.pt','latest.pt'):shutil.copy2(RUN/'checkpoints'/name,before/name)
        files={'train.py':HERE/'train.py','pipeline.py':HERE/'pipeline.py','initialization.py':HERE/'initialization.py',
               'preflight.py':HERE/'preflight.py','plan.json':HERE/'plan.json','provenance.json':HERE/'provenance.json',
               'registration.json':HERE/'runs/registered_training_loss.json',
               'preflight_result.json':HERE/'runs/training_preflight/results.json',
               'prepared_source_snapshot.zip':HERE/'runs/training_prepared_source_snapshot.zip',
               'budget.json':RUN/'budget.json'}
        for name,path in files.items():shutil.copy2(path,before/name)
        path=HERE/'train.py';text=path.read_text(encoding='utf-8')
        text=text.replace("REVISION = 'E89.frozen_placer_spatial_corrector.v1'", "REVISION = 'E89.frozen_placer_spatial_corrector.lr_unprotected.v2'")
        text=text.replace('def __init__(self, optimizer, start_epoch=15):','def __init__(self, optimizer, start_epoch=0):')
        text=text.replace('choices=(15,), default=15','choices=(0,), default=0')
        text=text.replace("getattr(args, 'scheduler_start_epoch', 15)","getattr(args, 'scheduler_start_epoch', 0)")
        text=text.replace('fresh Adam1e-3;14 protected epochs; frozen placer, only corrector trains',
                          'fresh Adam1e-3; no LR protection, epoch0 baseline; frozen placer, only corrector trains')
        path.write_text(text,encoding='utf-8')
        path=HERE/'pipeline.py';text=path.read_text(encoding='utf-8').replace("'--scheduler-start-epoch','15'","'--scheduler-start-epoch','0'")
        path.write_text(text,encoding='utf-8')
        path=HERE/'initialization.py';text=path.read_text(encoding='utf-8').replace(
            'new corrector from scratch:14 protected epochs, scheduling starts15, four exact tiers patience3',
            'E89 user-updated policy: no protection, epoch0 baseline, four exact tiers patience3')
        path.write_text(text,encoding='utf-8')
        path=HERE/'preflight.py';text=path.read_text(encoding='utf-8').replace("baseline['scheduler']['start_epoch']==15","baseline['scheduler']['start_epoch']==0")
        text=text.replace('lr1e3_and_protection14=True','lr1e3_no_protection=True');path.write_text(text,encoding='utf-8')
        for filename in ('plan.json','provenance.json'):
            path=HERE/filename;value=json.loads(path.read_text(encoding='utf-8'))
            value['lr_policy_update_authorization']='停止训练，先调整学习率下降规则；本次取消E89固定14轮保护，不自动续训'
            value['scheduler_start_epoch']=0
            value['lr_policy']='fixed dev500 same Loss; meaningful improvement max(abs1e-5,relative0.1%);3 stagnant epochs per drop; same four LR tiers; lowest tier2 full epochs'
            if filename=='plan.json':
                value['migration']='spatial stopped after epoch6/step3000; current epoch6 monitor pending. Anchor last complete epoch5, counters0, no retroactive LR drops.'
                value['comparison_limit']='Spatial first6 epochs used old protection; pooled not started. A later continuation versus new pooled policy is not a perfectly matched full-policy comparison.'
            else:value['new_stage_scheduler']='no protection; fresh baseline at0; existing spatial migrates at explicit pause without replaying old checks'
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        current_sources=sources()
        changed=sorted(k for k in old_sources if current_sources[k]!=old_sources[k])
        expected=sorted('experiments/E89_spatial_neural_refinement/'+x for x in
                        ('train.py','pipeline.py','initialization.py','preflight.py','plan.json','provenance.json'))
        assert changed==expected

        from experiments.E89_spatial_neural_refinement.train import PlateauController,parser,make_identity
        def controller():
            opt=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))],lr=.001)
            return PlateauController(opt,start_epoch=0)
        check=controller();check.step(1.,0)
        for epoch in range(1,12):
            check.step(1.,epoch)
            assert check.drops==min(epoch//3,3)
            assert check.should_stop()==(epoch>=11)
        check=controller();check.step(1.,0)
        for epoch,value in ((1,.9996),(2,.9992),(3,.9988)):check.step(value,epoch)
        assert check.drops==0 and check.lr_bad==0
        replica=controller();replica.load_state_dict(check.state_dict());same(check.state_dict(),replica.state_dict())
        checks=dict(no_protection=True,three_stagnant_checks_per_tier=True,
            small_improvements_accumulate=True,lowest_two_epochs_required=True,scheduler_restore_exact=True,
            new_neural_updates=0,new_real_inference_cases=0)
        save(ARCHIVE/'scheduler_checks.json',checks)

        (HERE/'runs/registered_training_loss.json').unlink()
        from experiments.E89_spatial_neural_refinement.calibrate import main as register
        register()
        old_coeff=json.loads((before/'registration.json').read_text(encoding='utf-8'))['coefficients']
        registration=json.loads((HERE/'runs/registered_training_loss.json').read_text(encoding='utf-8'))
        assert old_coeff==registration['coefficients']
        from experiments.E89_spatial_neural_refinement.loss_config import load_coefficients
        calibration=load_coefficients(HERE/'runs/registered_training_loss.json')
        args=parser().parse_args(['--arm','spatial','--output',str(RUN),'--max-hours','1','--max-epochs','200','--scheduler-start-epoch','0'])
        new_identity=make_identity(args);new_identity['calibration']=calibration
        allowed={'revision','sources','initialization','new_corrector_lr_policy','scheduler_start_epoch','calibration'}
        identity_changes={k for k in new_identity if new_identity[k]!=latest['identity'][k]}
        assert identity_changes.issubset(allowed),identity_changes

        anchors={}
        for filename in ('best.pt','latest.pt'):
            state=torch.load(before/filename,map_location='cpu',weights_only=False)
            original=copy.deepcopy(state)
            anchor=state['progress']['history'][-1]
            # Keep actual current LR/drop/counters. Only install the previously
            # missing meaningful-best baseline and enable subsequent checks.
            scheduler=copy.deepcopy(state['scheduler']);scheduler['start_epoch']=0
            assert scheduler['best'] is None and scheduler['drops']==0 and scheduler['lr_bad']==scheduler['early_bad']==0
            scheduler['best']=anchor['dev_monitor']['total']
            state['scheduler']=scheduler;state['identity']=new_identity
            for key in original:
                if key not in ('identity','scheduler'):same(state[key],original[key])
            checkpoint(RUN/'checkpoints'/filename,state)
            restored=torch.load(RUN/'checkpoints'/filename,map_location='cpu',weights_only=False)
            same(state,restored)
            anchors[filename]=dict(epoch=anchor['epoch'],loss=scheduler['best'],
                original_sha256=sha(before/filename),new_sha256=sha(RUN/'checkpoints'/filename))
        previous_budget=json.loads((before/'budget.json').read_text(encoding='utf-8'))['elapsed_seconds']
        result=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
        charged=max(previous_budget,latest['elapsed_seconds'],result['elapsed_seconds'])
        save(RUN/'budget.json',dict(elapsed_seconds=charged))

        updated=copy.deepcopy(old_pre);updated['sources']=current_sources
        updated.pop('lr1e3_and_protection14',None);updated['lr1e3_no_protection']=True
        updated['policy_update']=dict(changed_sources=changed,
            original_neural_numeric_preflight=str((before/'preflight_result.json').relative_to(HERE)),
            original_neural_preflight_sha256=sha(before/'preflight_result.json'),
            new_scheduler_checks=str((ARCHIVE/'scheduler_checks.json').relative_to(HERE)),
            note='Model/Loss/data/CUDA update unchanged; inherit actual five-case checks and separately exercise new scheduler; no retraining')
        save(HERE/'runs/training_preflight/results.json',updated)
        record=dict(status='migrated_paused',authorization='user stop and LR-rule revision',
            changed_sources=changed,identity_changes=sorted(identity_changes),anchors=anchors,
            preserved=['model','Adam','RNG','data order/cursor','sample exposure','progress/history','step accounting','cumulative elapsed'],
            epoch=latest['progress']['epoch'],step=latest['progress']['step'],cursor=latest['progress']['cursor'],
            pending_monitor=latest['progress']['pending_monitor'],current_lr=latest['optimizer']['param_groups'][0]['lr'],
            charged_seconds=charged,budget_before=previous_budget,cap_seconds=new_identity['max_seconds'],
            no_retroactive_LR_drop=True,loss_coefficients_unchanged=True,tests_sealed=True,
            next='explicit user resume required; STOP remains; pooled unstarted')
        save(ARCHIVE/'migration.json',record)
        save(HERE/'state.json',dict(status='user_stop',stage='paused',arm='spatial',epoch=record['epoch'],step=record['step'],
            best_epoch=latest['progress']['best_epoch'],current_lr=record['current_lr'],
            elapsed_seconds=charged,lr_rule='no protection;3 stagnant dev checks; no historical replay',
            pipeline_lock_released=True,pooled_started=False,test_read=False))
        from experiments.E89_spatial_neural_refinement.preflight import verify
        verify()
        print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
