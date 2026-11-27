"""CPU forward/configuration checks only; no training updates or solvers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
import sys,json,ast,time
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from datasets.reference_layouts import ReferenceLayouts
from experiments.E42_joint_regularization.model import GeometryModel
from experiments.E42_joint_regularization.arms import ARMS,model_config
from experiments.E42_joint_regularization.data import load_rows,collate
from experiments.E42_joint_regularization.batches import inputs_only
from experiments.E42_joint_regularization.losses import WEIGHTS
from experiments.E42_joint_regularization.train import make_identity,WEIGHT_DECAY
from experiments.E42_joint_regularization.pipeline import commands,control_identity,OUTPUT,COORDINATOR
from experiments.E42_joint_regularization.io import sources,save,sha
from experiments.E42_joint_regularization.cache import verify
from experiments.E39_rectangular_attention.model import GeometryModel as OriginalModel

def main():
    started=time.perf_counter();experiment=HERE.parents[1]
    assert not OUTPUT.exists() and not COORDINATOR.exists()
    for p in experiment.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    cfg=model_config('rectangle_ab');assert len(ARMS)==1 and cfg=={'width':96,'layers':12,'rectangle':True,'ffn_dropout':.05}
    torch.manual_seed(0);model=GeometryModel(**cfg)
    torch.manual_seed(0);original=OriginalModel(width=96,layers=12,rectangle=True)
    assert model.state_dict().keys()==original.state_dict().keys()
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in model.state_dict().items()),'Initialization changed beyond AB'
    assert sum(p.numel() for p in model.parameters())==1471539
    drops=[m for m in model.modules() if isinstance(m,torch.nn.Dropout)]
    assert len(drops)==12 and all(m.p==.05 for m in drops)
    old=ROOT/'experiments/E39_rectangular_attention/runs/20261023_rectangle_w96d12_r2_rectangle_seed0/checkpoints/best.pt'
    state=torch.load(old,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);original.load_state_dict(state['model'],strict=True)
    previous=ROOT/'experiments/E39_rectangular_attention/runs/20261023_training_entry_cuda/results.json'
    evidence=json.loads(previous.read_text(encoding='utf-8'));members=[tuple(x) for x in evidence['gradient_members']]
    metadata=ReferenceLayouts('train');indices={tuple(r['member']):i for i,r in enumerate(metadata.records)}
    train=load_rows('train',[indices[m] for m in members],use_cache=False)
    dev=load_rows('dev_monitor500',[0],use_cache=False)
    assert list(dev[0]['member'])==evidence['validation_members'][0]
    dg,_=collate(dev,'cpu');dg=inputs_only(dg)
    model.eval();original.eval()
    with torch.no_grad():a=model(dg);b=original(dg);again=model(dg)
    assert torch.equal(a,b) and torch.equal(a,again),'Eval parity failed'
    tg,_=collate(train,'cpu');tg=inputs_only(tg)
    counts=[]
    def hook(module,args,out):
        x=args[0];active=x!=0
        counts.append(float(((out==0)&active).sum()/active.sum()))
    hooks=[m.register_forward_hook(hook) for m in drops]
    model.train();saved_rng=torch.get_rng_state()
    with torch.no_grad():first=model(tg);second=model(tg)
    assert not torch.equal(first,second),'Dropout not active during train forward'
    torch.set_rng_state(saved_rng)
    with torch.no_grad():restored=model(tg)
    assert torch.equal(first,restored),'Dropout RNG restore failed'
    assert len(counts)==36 and all(.02<v<.08 for v in counts)
    for h in hooks:h.remove()
    assert torch.isfinite(first).all()
    args=SimpleNamespace(arm='rectangle_ab',device='cpu',seed=0,max_hours=4.,max_epochs=200,batch_size=32,scheduler_start_epoch=15)
    identity=make_identity(args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=identity['lr'],weight_decay=WEIGHT_DECAY)
    assert identity['parameters']==1471539 and identity['ffn_dropout']==.05
    assert identity['weight_decay']==.0003 and all(g['weight_decay']==.0003 for g in optimizer.param_groups)
    assert identity['data_augmentation'] is False and WEIGHTS['mib']==.1
    for name in ['loss.py','losses.py','occupancy.py','features.py','batches.py','data.py','cache.py','solver.py']:
        assert sha(experiment/name)==sha(ROOT/'experiments/E39_rectangular_attention'/name),name
    verify('train');verify('dev');control=control_identity();cmd=commands(False)
    assert cmd['train'][cmd['train'].index('--arm')+1]=='rectangle_ab' and '--baseline' not in cmd['final']
    result={'status':'prepared_not_started','model_config':cfg,'parameters':1471539,'dropout_modules':12,
        'dropout_location':'each FFN hidden, after SiLU before second Linear','dropout_observed_zero_fraction_mean':sum(counts)/len(counts),
        'same_seed_initial_parameter_exact_parity':True,'old_checkpoint_strict_load':True,'eval_output_exact_parity':True,
        'eval_deterministic':True,'train_forward_stochastic':True,'train_forward_RNG_restore_exact':True,
        'actual_optimizer_weight_decay':optimizer.param_groups[0]['weight_decay'],'identity_matches_optimizer':True,
        'loss_data_cache_solver_unchanged':True,'source_identity':sources(),'control_identity':control,'commands':cmd,
        'train_forward_members':[list(r['member']) for r in train],'eval_members':[list(r['member']) for r in dev],
        'device':'cpu','training_started':False,'optimizer_steps':0,'backward_calls':0,'solver_calls':0,'test_read':False,
        'data_augmentation':False,'formal_run_directories_created':False,'cuda_training_or_resume_checked':False,
        'seconds':time.perf_counter()-started}
    save(HERE/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_identity','control_identity','commands']},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
