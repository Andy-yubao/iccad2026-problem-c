"""One explicit, audited scheduler migration; no training or model inference."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import copy,json,time,shutil
from datetime import datetime,timezone
import numpy as np
import torch
from experiments.E85_cooperative_space_refinement.io import HERE,ROOT,save,sha,sources,checkpoint,snapshot
from experiments.E85_cooperative_space_refinement.train import PlateauController,parser,make_identity
from experiments.E85_cooperative_space_refinement.source_compatibility import reviewed_sources

OUT=HERE/'runs/lr_policy_change'
RUN=HERE/'runs/20261030_P74_F81_cooperative_seed0'
CONTROL=HERE/'runs/20261030_P74_F81_control_seed0'

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

def controller(start=0):
    opt=torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))],lr=.001)
    return opt,PlateauController(opt,start_epoch=start)

def check_rule():
    opt,s=controller()
    drops=[]
    for epoch in range(12):
        old=s.drops;s.step(.5,epoch)
        if s.drops!=old:drops.append(epoch)
        if epoch<11:assert not s.should_stop()
    assert drops==[3,6,9] and s.should_stop() and opt.param_groups[0]['lr']==5e-5
    opt,s=controller();s.step(.5,0);s.step(.5,1);s.step(.499,2)
    assert s.lr_bad==0 and s.early_bad==0
    for epoch in (3,4,5):s.step(.499,epoch)
    assert s.drops==1 and opt.param_groups[0]['lr']==.00025
    opt,s=controller(15)
    for epoch in range(15):s.step(.5,epoch)
    assert s.best is None and s.drops==0
    for epoch in range(15,19):s.step(.5,epoch)
    assert s.drops==1
    return dict(no_protection_first_drop_epoch=3,all_drop_epochs=[3,6,9],earliest_flat_loss_stop_epoch=11,
                meaningful_improvement_resets_patience=True,from_scratch_protection_retained=True)

def prime(before):
    old=before['scheduler'];assert old['drops']==0 and old['start_epoch']==15
    best=None;bad=0
    for record in before['progress']['history']:
        value=record['dev_monitor']['total']
        delta=max(1e-5,abs(best)*.001) if best is not None else 0.
        if best is None or value<best-delta:best=value;bad=0
        else:bad+=1
    state=dict(old)
    state.update(start_epoch=0,best=best,lr_bad=bad,early_bad=bad)
    # We count observed stagnation, but never pretend earlier LR reductions ran.
    # At most one real immediate drop can occur at this intervention boundary.
    immediate=bad>=3
    if immediate:
        state.update(drops=1,lr_bad=0,last_drop_epoch=before['progress']['epoch'],
                     lowest_validated=False,lowest_completed_epochs=0)
    return state,immediate

def main():
    begin=time.perf_counter()
    assert not (OUT/'migration.json').exists(),'Migration already applied; do not apply twice'
    state=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    assert state['status']=='stopped' and (HERE/'runs/STOP').exists()
    review=json.loads((OUT/'source_compatibility.json').read_text(encoding='utf-8'))
    assert reviewed_sources(review['old_sources'],sources())
    checks=check_rule()
    protected={str(x.relative_to(ROOT)):sha(x) for name in ('best.pt','latest.pt') for x in [CONTROL/'checkpoints'/name]}
    args=parser().parse_args(['--arm','cooperative','--output',str(RUN),'--max-hours','2','--max-epochs','200','--batch-size','32','--seed','0','--engine','cuda_graph','--resume'])
    identity=make_identity(args)
    archived=RUN/'history/20261030_remove_posttrain_protection'
    assert archived.resolve().is_relative_to(RUN.resolve())
    archived.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in ('latest.pt','best.pt'):
        path=RUN/'checkpoints'/name
        before=torch.load(path,map_location='cpu',weights_only=False)
        assert before['identity']['sources']==review['old_sources']
        diff={k for k in before['identity'].keys()|identity.keys() if before['identity'].get(k)!=identity.get(k)}
        assert diff=={'sources','scheduler_start_epoch'},diff
        original=archived/name;shutil.copy2(path,original)
        original_sha=sha(path);assert sha(original)==original_sha
        after=copy.deepcopy(before);scheduler,immediate=prime(before)
        after['identity']=identity;after['scheduler']=scheduler
        after['progress']['bad_checks']=scheduler['early_bad']
        if immediate:
            for group in after['optimizer']['param_groups']:group['lr']=.00025
        for k in ('model','rng','graph_step_control','elapsed_seconds','model_config'):
            same(before[k],after[k])
        progress_before=copy.deepcopy(before['progress']);progress_after=copy.deepcopy(after['progress'])
        progress_before.pop('bad_checks');progress_after.pop('bad_checks');same(progress_before,progress_after)
        if not immediate:same(before['optimizer'],after['optimizer'])
        checkpoint(path,after)
        check=torch.load(path,map_location='cpu',weights_only=False)
        same(check,after)
        records.append(dict(checkpoint=name,archived_original=str(original.relative_to(ROOT)),
            original_sha256=original_sha,migrated_sha256=sha(path),step=before['progress']['step'],
            completed_epochs=before['progress']['epoch'],cursor=before['progress']['cursor'],
            elapsed_seconds=before['elapsed_seconds'],scheduler_before=before['scheduler'],scheduler_after=scheduler,
            actual_immediate_drop=immediate,optimizer_lr=after['optimizer']['param_groups'][0]['lr'],
            preserved_model_optimizer_RNG_progress_and_budget=True))
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    snapshot(OUT/'source_snapshot.zip',identity)
    save(OUT/'continuation_config.json',dict(identity=identity,command='python -u -B -m experiments.E85_cooperative_space_refinement.pipeline --resume',original_config_unchanged=True))
    save(OUT/'migration.json',dict(status='applied_verified',utc=datetime.now(timezone.utc).isoformat(),
        authorization='User removes LR protection from post-training; from-scratch still protected',
        history_policy='Observed epoch0..11 losses prime meaningful-best and trailing counts; no fictional historical drops/lowest-tier epochs',
        records=records,completed_control_checkpoint_hashes=protected,completed_control_preserved=True,
        comparison_limit='Control finished with protection; cooperative continued without it. Final differences include schedule policy; no pure operator attribution.',
        scalar_policy_tests=checks,migration_seconds=time.perf_counter()-begin,test_read=False))
    oldcheck=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    save(OUT/'validation.json',dict(status='passed',sources=sources(),seconds=oldcheck['seconds'],
        original_algorithm_preflight_reused=True,reviewed_model_Loss_data_and_update_code_unchanged=True,
        scheduler_tests=checks,migration_checkpoint_tree_equality_verified=True,test_read=False))
    print(json.dumps({'status':'applied_verified','latest':records[0],'scalar_tests':checks},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
