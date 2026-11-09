"""Read-only five-case audit on a copied checkpoint; never touches live training."""
import importlib.util,json,time
from pathlib import Path
import torch
from experiments.E91_e74_32k_scratch.io import HERE,ROOT,sha,save
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from experiments.E91_e74_32k_scratch.batches import inputs_only
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.loss import per_case,losses
from experiments.E91_e74_32k_scratch.train import PlateauController,LR_LEVELS

def main():
    started=time.perf_counter();torch.set_num_threads(4)
    audit=Path(__file__).resolve().parent
    c=torch.load(audit/'inspection_checkpoint.pt',map_location='cpu',weights_only=False)
    ids=[1683,10027,5474,18140,25001]
    rows=load_rows('train',ids)
    host=ResidentRows(rows,'cpu',max_nodes=120)
    path=HERE/'runs/resource_diagnosis_20261031/before/resident.py'
    spec=importlib.util.spec_from_file_location('old_E91_resident_for_audit',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    reference=old.ResidentRows(rows,'cpu',max_nodes=120)
    for sequence in (list(range(5)),[4,2,0,1,3],[i%5 for i in range(20)]):
        a=host.batch(sequence);b=reference.batch(sequence)
        assert a.keys()==b.keys()
        for key in a:assert torch.equal(a[key],b[key]),key
    g=host.batch(list(range(5)))
    host.batch([4,3,2,1,0],out=g)
    expected=reference.batch([4,3,2,1,0])
    for key in g:assert torch.equal(g[key],expected[key])
    host.batch(list(range(5)),out=g)
    model=GeometryModel(**c['model_config']).eval().requires_grad_(False)
    model.load_state_dict(c['model'])
    with torch.no_grad():prediction=model(inputs_only(g))
    state=prediction.detach().requires_grad_()
    values=per_case(state,g,'simple');training=losses(state,g)['total']
    assert torch.equal(training,values['total'].mean())
    manual=(values['reference']+.08975162861857286*values['overlap_total']+.3336832108335229*values['task']).mean()
    assert torch.equal(training,manual)
    gradient=torch.autograd.grad(training,state,retain_graph=True)[0]
    manual_gradient=torch.autograd.grad(manual,state)[0]
    assert torch.equal(gradient,manual_gradient) and torch.isfinite(gradient).all()
    assert float(gradient.abs().max())>0
    changed={**g,'target':g['target']+2}
    public=inputs_only(g);changed_public=inputs_only(changed)
    for key in public:assert torch.equal(public[key],changed_public[key])
    changed_loss=per_case(prediction,changed,'simple')
    for key in ('overlap_total','task'):assert torch.equal(values[key].detach(),changed_loss[key])
    assert not torch.equal(values['reference'].detach(),changed_loss['reference'])
    opt=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))],lr=1e-3)
    schedule=PlateauController(opt,start_epoch=c['identity']['scheduler_start_epoch'])
    for record in c['progress']['history']:
        schedule.step(record['dev_monitor']['total'],record['epoch'])
        assert opt.param_groups[0]['lr']==record['lr_next']
    assert schedule.state_dict()==c['scheduler']
    assert c['optimizer']['param_groups'][0]['lr']==LR_LEVELS[c['scheduler']['drops']]
    keys=('model.py','features.py','batches.py','simple_loss.py','group_tree.py','loss.py','ops.py')
    equal={key:sha(HERE/key)==sha(ROOT/'experiments/E90_auxiliary_weight_placer'/key) for key in keys}
    assert all(equal.values())
    assert sha(HERE/'model.py')==sha(ROOT/'experiments/E74_collision_32k_posttrain/model.py')
    result=dict(status='passed',checkpoint_sha256=sha(audit/'inspection_checkpoint.pt'),
        checkpoint_step=c['progress']['step'],unique_real_train_cases=5,ids=ids,device='CPU only',
        different_batch_sizes_reorder_and_out_reuse_bit_exact=True,train_monitor_loss_and_state_gradient_bit_exact=True,
        public_model_inputs_independent_of_labels=True,public_loss_terms_independent_of_labels=True,
        scheduler_replayed_from_saved_history=True,all_Adam_LR_matches_scheduler=True,
        core_files_byte_equal_E90=equal,model_byte_equal_E74=True,
        frozen_snapshot_case_loss=float(training.detach()),
        formal_training_or_checkpoint_mutated=False,neural_parameter_updates=0,
        seconds=time.perf_counter()-started,test_read=False)
    save(audit/'cpu_checks.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
