"""Three-case CPU gradient/proxy checks, no optimizer steps or solvers."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
from pathlib import Path
import sys, json, ast, time
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
from datasets.reference_layouts import ReferenceLayouts
from experiments.E43_overlap_area_loss.data import load_rows, collate
from experiments.E43_overlap_area_loss.batches import inputs_only
from experiments.E43_overlap_area_loss.model import GeometryModel
from experiments.E43_overlap_area_loss.arms import ARMS,model_config
from experiments.E43_overlap_area_loss.loss import losses,weights
from experiments.E43_overlap_area_loss.area_loss import overlap_area,AREA_WEIGHT
from experiments.E43_overlap_area_loss.occupancy import components as occupancy
from experiments.E43_overlap_area_loss.io import HERE,save,sha,sources
from experiments.E43_overlap_area_loss.pipeline import OUTPUT,COORDINATOR,CONTROL,control_identity,commands
from experiments.E43_overlap_area_loss.cache import verify
from experiments.E43_overlap_area_loss.train import parser,make_identity,WEIGHT_DECAY
from experiments.E43_overlap_area_loss.report import summarize
from experiments.E43_overlap_area_loss.paired import paired
from experiments.E42_joint_regularization.loss import losses as old_losses
from experiments.E42_joint_regularization.model import GeometryModel as OldModel


def flat_gradient(value,params):
    gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])


def main():
    started=time.perf_counter();assert not OUTPUT.exists() and not COORDINATOR.exists()
    for p in HERE.rglob('*.py'):
        if 'runs' not in p.relative_to(HERE).parts:ast.parse(p.read_text(encoding='utf-8'))
    assert len(ARMS)==1 and model_config('rectangle_area')=={'width':96,'layers':12,'rectangle':True,'ffn_dropout':.05}
    torch.manual_seed(0);model=GeometryModel(**model_config('rectangle_area'))
    torch.manual_seed(0);old=OldModel(**model_config('rectangle_area'))
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters())==1471539
    args=parser().parse_args(['--output',str(OUTPUT),'--device','cpu'])
    identity=make_identity(args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=identity['lr'],weight_decay=WEIGHT_DECAY)
    assert identity['overlap_area_weight']==.25 and identity['batch_size']==32
    assert identity['weight_decay']==.0003 and identity['ffn_dropout']==.05
    assert all(group['weight_decay']==.0003 for group in optimizer.param_groups)
    for name in ('model.py','losses.py','occupancy.py','batches.py','features.py','data.py','cache.py','solver.py'):
        assert sha(HERE/name)==sha(ROOT/'experiments/E42_joint_regularization'/name), name
    evidence=json.loads((ROOT/'experiments/E39_rectangular_attention/runs/20261023_training_entry_cuda/results.json').read_text(encoding='utf-8'))
    metadata=ReferenceLayouts('train');lookup={tuple(r['member']):i for i,r in enumerate(metadata.records)}
    train=load_rows('train',[lookup[tuple(m)] for m in evidence['gradient_members']],use_cache=False)
    dev=load_rows('dev_monitor500',[0],use_cache=False)
    assert len(train)==2 and len(dev)==1
    measurements=[]
    state=torch.load(CONTROL/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    for label in ('initialization','E42_frozen_best'):
        if label=='E42_frozen_best':model.load_state_dict(state['model'],strict=True)
        model.eval();g,t=collate(train,'cpu');prediction=model(inputs_only(g))
        current=losses(prediction,g,t,'rectangle_area',2.,include_quality=False)
        baseline=old_losses(prediction,g,t,'rectangle_ab',2.,include_quality=False)
        expected=baseline['total']+AREA_WEIGHT*current['overlap_area']
        torch.testing.assert_close(current['total'],expected,rtol=0,atol=0)
        area=overlap_area(prediction,g);other=occupancy(prediction,g)['overlap_area']
        torch.testing.assert_close(area,other,rtol=0,atol=0)
        ga=flat_gradient(AREA_WEIGHT*area.mean(),list(model.parameters()))
        gb=flat_gradient(baseline['total'],list(model.parameters()))
        gp=flat_gradient(.5*baseline['overlap'],list(model.parameters()))
        gt=flat_gradient(current['total'],list(model.parameters()))
        torch.testing.assert_close(gt,ga+gb,rtol=2e-4,atol=2e-6)
        assert torch.isfinite(ga).all() and ga.norm()>0
        # Same expression and gradients even when full quality evaluation is on.
        qa=flat_gradient(other.mean(),list(model.parameters()))
        torch.testing.assert_close(ga,AREA_WEIGHT*qa,rtol=2e-4,atol=2e-6)
        record={'state':label,'train_members':[list(r['member']) for r in train],
            'base_loss':float(baseline['total']),'area':float(area.mean()),
            'weighted_area_loss':float(AREA_WEIGHT*area.mean()),
            'base_parameter_grad_norm':float(gb.norm()),'weighted_area_parameter_grad_norm':float(ga.norm()),
            'weighted_penetration_parameter_grad_norm':float(gp.norm()),
            'area_to_base_grad_ratio':float(ga.norm()/gb.norm()),
            'area_to_penetration_grad_ratio':float(ga.norm()/gp.norm()),
            'area_base_grad_cosine':float(torch.dot(ga,gb)/(ga.norm()*gb.norm()))}
        measurements.append(record)
        for epoch,coefficient in [(0,0.),(1,.125),(2,.25),(3,.25)]:
            a=losses(prediction,g,t,'rectangle_area',epoch,include_quality=False)
            b=old_losses(prediction,g,t,'rectangle_ab',epoch,include_quality=False)
            torch.testing.assert_close(a['total'],b['total']+coefficient*area.mean(),rtol=0,atol=0)
            assert weights('rectangle_area',epoch)['overlap_area']==coefficient
    model.eval();old.load_state_dict(state['model'],strict=True);old.eval()
    g,t=collate(dev,'cpu')
    with torch.no_grad():
        a=model(inputs_only(g));b=old(inputs_only(g));assert torch.equal(a,b)
        new=losses(a,g,t,'rectangle_area',2.);prior=old_losses(b,g,t,'rectangle_ab',2.)
        for key in prior:
            if key!='total':torch.testing.assert_close(new[key],prior[key],rtol=0,atol=0)
    # Analytic two-block layout: intersection .5*1 divided by total area2 = .25.
    toy={'area':torch.ones(1,3),'valid':torch.tensor([[True,True,False]]),
         'mask':torch.ones(1,3,3),'specs':torch.tensor([[[1.,1.,1.,1.],[1.5,1.,1.,1.],[1.,1.,1.,1.]]])}
    x=torch.tensor([[[1.5,1.5,0.],[2.,1.5,0.],[1.5,1.5,0.]]],requires_grad=True)
    assert overlap_area(x,toy).item()==.25
    gr=torch.autograd.grad(overlap_area(x,toy).sum(),x)[0]
    assert gr[0,1,0]<0 and torch.equal(gr[0,2],torch.zeros(3))
    toy['mask'][0,0]=0 # One preplaced, one free: still penalized.
    assert overlap_area(x,toy).item()==.25
    toy['mask'][0,1]=0 # Both immutable: no trainable remedy.
    assert overlap_area(x,toy).item()==0
    toy['mask'][:]=1
    with torch.no_grad():x[0,1,0]=2.5 # Exact edge contact incurs no area loss.
    assert overlap_area(x,toy).item()==0
    verify('train');verify('dev');control=control_identity()
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    for name,digest in provenance['source_sha256'].items():
        assert sha(ROOT/'experiments/E42_joint_regularization'/name)==digest,'E42 source changed: '+name
    # Read-only report functions on two saved records, plus a failed-record copy.
    records=json.loads((CONTROL/'terminal/results.json').read_text(encoding='utf-8'))['records'][:2]
    assert paired(records,records)['weighted_Q_delta']==0
    failure={'n':records[0]['n'],'seconds':0.,'error':'synthetic report check',
             'metrics':{'cost':10.,'is_feasible':False}}
    summary=summarize([records[0],failure])
    assert summary['count']==2 and summary['errors']==1 and summary['available_diagnostic_count']==1
    # Core experiment source must not import another experiment's algorithms.
    for p in HERE.rglob('*.py'):
        if 'runs' not in p.relative_to(HERE).parts:
            assert 'from experiments.E42' not in p.read_text(encoding='utf-8')
    result={'status':'prepared_not_started','coefficient':AREA_WEIGHT,'model_parameters':1471539,
        'gradient_calibration_train_only':measurements,'gradient_identity_passed':True,
        'area_matches_unchanged_occupancy_value_and_gradient':True,'two_epoch_ramp_passed':True,
        'selection_proxy_exact_parity':True,'same_seed_parameter_parity':True,'strict_checkpoint_load':True,
        'actual_entry_configuration_and_optimizer_passed':True,'saved_report_identity_and_failure_denominator_passed':True,
        'E42_source_preservation_passed':True,
        'padding_preplaced_contact_analytic_checks':True,'eval_member':list(dev[0]['member']),
        'sources':sources(),'control':control,'commands':commands(False),'device':'cpu',
        'training_started':False,'optimizer_steps':0,'solver_calls':0,'test_read':False,
        'formal_output_directories_created':False,'seconds':time.perf_counter()-started}
    save(Path(__file__).parent/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','control','commands')},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
