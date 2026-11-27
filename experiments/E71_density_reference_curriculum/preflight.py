"""Five unique training cases only; short implementation updates, no formal training."""
import gc
import hashlib
import json
import time
import torch
from .io import HERE,ROOT,save,sha,sources
from .initialization import verify_protected,metadata
from .arms import ARMS,model_config
from .model import GeometryModel
from .base_model import GeometryModel as OriginalModel
from .batches import inputs_only,collate as batch_collate
from .data import load_rows
from .resident import ResidentRows
from .loss import losses
from .losses import components,public_components,PUBLIC_TASK_KEYS,WEIGHTS,TASK_KEYS,contact_cost
from .group_tree import minimum_group_tree
from .cuda_step import CapturedStep
from .train import seed_all,parser,run,selection,PlateauController
from .curriculum import position_weight
from .density import density_energy


def same(a,b,path='state'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
    else:assert a==b,path


def clear():
    from . import data
    data._resident.clear();gc.collect();torch.cuda.empty_cache()


def geometry_fixtures():
    import numpy as np
    from .batches import sample
    def fixture(positions,groups=None,mib=None):
        r=torch.tensor(positions,dtype=torch.float32);n=len(r)
        c=np.zeros((n,5),dtype=np.float32)
        if groups is not None:c[:,3]=groups
        if mib is not None:c[:,2]=mib
        inputs=(n,r[:,2:].prod(1),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),c,np.full((n,4),-1.))
        row=sample(inputs,r);g=batch_collate([row],[0],'cuda');state=g['target'].clone().requires_grad_()
        return g,state
    g,state=fixture([[0,0,10,10],[3.1,4,2,2]])
    first=components(state,g);other=state.detach().clone();other[:,1,0]+=.1/g['scale'][0]
    second=components(other,g)
    assert first['penetration'].item()>0 and first['intersection_area'].item()>0
    torch.testing.assert_close(first['intersection_area'],second['intersection_area'])
    assert not torch.equal(first['penetration'],second['penetration'])
    gc,sc=fixture([[0,0,1,1],[1,1,1,1]],[1,1]);ge,se=fixture([[0,0,1,1],[1,0,1,1]],[1,1])
    assert components(sc,gc)['group_connectivity'].item()>0
    assert components(se,ge)['group_connectivity'].item()<1e-10
    ga,sa=fixture([[0,0,1,1],[1,0,1,1],[2,0,1,1]],[1,1,1],[1,1,1])
    gb,sb=fixture([[0,0,1,1],[1,0,1,1],[0,1,1,1]],[1,1,1],[1,1,1])
    assert components(sa,ga)['group_connectivity'].item()<1e-10 and components(sb,gb)['group_connectivity'].item()<1e-10
    assert components(sa,ga)['mib'].item()<1e-10
    from scipy.sparse.csgraph import minimum_spanning_tree
    raw=np.array([[0,2,7,9,8],[2,0,3,8,7],[7,3,0,8,9],[9,8,8,0,4],[8,7,9,4,0]],float)
    labels=np.array([1,1,1,2,2]);rel=(labels[:,None]==labels[None,:])&~np.eye(5,dtype=bool)
    expected=(float(minimum_spanning_tree(raw[:3,:3]).sum())+raw[3,4])/3
    cost=torch.tensor(raw,device='cuda',dtype=torch.float32)[None].requires_grad_()
    value=minimum_group_tree(cost,torch.tensor(rel,device='cuda')[None],torch.ones(1,5,device='cuda',dtype=torch.bool))
    torch.testing.assert_close(value,torch.tensor([expected],device='cuda',dtype=torch.float32));value.sum().backward()
    assert torch.isfinite(cost.grad).all()
    gn,sn=fixture([[0,0,1,1],[2,0,1,1]],[1,2],[1,2]);sn=sn.detach();sn[...,0]=-1;sn[...,2]=6;sn.requires_grad_()
    v=components(sn,gn)
    assert v['actual_wirelength'].item()==0 and v['group_connectivity'].item()==0 and v['mib'].item()==0
    dg=torch.autograd.grad(v['parameter_domain'].sum(),sn)[0]
    assert torch.isfinite(dg).all() and dg.abs().sum()>0
    return {'fixtures':5,'containment_depth_and_area_complementary':True,'point_contact_rejected':True,
            'alternative_group_topologies_accepted':True,'GPU_forest_matches_independent_MST':True,
            'zero_edges_singletons_and_domain_gradients':True}


def density_fixtures():
    # Exact tiling has zero non-DC rectangle moments. A displaced interior
    # block exposes a hole/overlap and has an interior center gradient.
    tiled=torch.tensor([[[x,y,1.,1.] for x in range(4) for y in range(4)]],device='cuda')
    a=torch.ones(1,16,device='cuda');valid=torch.ones_like(a,dtype=torch.bool)
    uniform=density_energy(tiled,a,valid)
    assert uniform.item()<1e-12
    moved=tiled.clone();moved[0,5,0]+=.2;moved.requires_grad_()
    value=density_energy(moved,a,valid)
    grad=torch.autograd.grad(value.sum(),moved)[0]
    assert value.item()>1e-7 and grad[0,5,:2].norm().item()>1e-6
    step=moved.detach().clone();step[0,5,:2]-=.05*grad[0,5,:2]/grad[0,5,:2].norm()
    assert density_energy(step,a,valid).item()<value.item()
    spread=tiled.clone();spread[...,:2]*=1.2
    expanded=density_energy(spread,a,valid)
    assert expanded.item()>uniform.item()+1e-6
    padded=density_energy(tiled,a,valid,bounds=(torch.zeros(1,2,device='cuda'),torch.full((1,2),5.,device='cuda')))
    assert padded.item()>uniform.item()+1e-6
    torch.testing.assert_close(density_energy(moved.detach()*3,a*9,valid),value.detach(),rtol=1e-4,atol=1e-7)
    # Validate both backward values and derivatives against differentiable
    # quadrature of the cosine basis in a fixed rectangle, independently.
    q=tiled.double();q[0,5,0]+=.2;q.requires_grad_()
    fixed=(torch.zeros(1,2,device='cuda',dtype=torch.float64),torch.full((1,2),4.,device='cuda',dtype=torch.float64))
    from .density import modes
    import math
    k,w=modes(q.device,q.dtype)
    t=(torch.arange(256,device=q.device,dtype=q.dtype)+.5)/256
    coords=q[...,None,:2]+q[...,None,2:]*t[None,None,:,None]
    factors=(coords[:,:,:,None,:]/4*k[None,None,None]*math.pi).cos().mean(2)
    c=(factors[...,0]*factors[...,1]).mean(1)
    numerical=(c.square()[:,None]*w[None]).sum(-1).mean(-1)
    analytic=density_energy(q,a.double(),valid,bounds=fixed)
    torch.testing.assert_close(analytic,numerical,rtol=3e-4,atol=1e-7)
    ga=torch.autograd.grad(analytic.sum(),q,retain_graph=True)[0]
    gn=torch.autograd.grad(numerical.sum(),q)[0]
    torch.testing.assert_close(ga,gn,rtol=5e-4,atol=1e-7)
    dummy=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.AdamW([dummy],lr=1e-3)
    controller=PlateauController(opt)
    for epoch in range(1,16):controller.step(1./epoch,epoch)
    assert controller.best is None and controller.drops==0
    for epoch in range(16,28):controller.step(1.,epoch)
    assert controller.drops==3 and controller.lowest_completed_epochs==2 and controller.should_stop()
    return {'uniform_tiling_energy':uniform.item(),'interior_displacement_energy':value.item(),
            'interior_center_gradient_norm':grad[0,5,:2].norm().item(),
            'center_step_reduces_energy':True,'spread_energy':expanded.item(),'padded_frame_energy':padded.item(),
            'quadrature_forward_and_gradient_agree':True,'physical_scale_invariant':True,
            'no_warmup_selection_or_LR_count':True,'three_drops_and_two_lowest_epochs':True,
            'limits':'Analytic fixtures only; finite modes can miss holes and density can still trade against bbox; no packing guarantee'}


def check_arm(arm,batch):
    from .inference import graph as inference_graph
    from scoring.official import official_evaluator
    seed_all(0);model=GeometryModel(**model_config(arm)).cuda().eval();pack=ResidentRows(batch,'cuda',120)
    g=pack.batch(list(range(32)));public=inputs_only(g)
    assert public['features'].shape[-1]==14 and 'wire_b2b' not in public
    with torch.no_grad():prediction=model(public)
    state=prediction.detach().requires_grad_();v=components(state,g)
    expected=sum(WEIGHTS[k]*v[k] for k in WEIGHTS)
    torch.testing.assert_close(v['total'],expected,rtol=1e-6,atol=1e-6)
    values=losses(state,g,None,arm);assert selection(values) is values['total']
    for epoch in (1,6,10,15,16,200):
        scheduled=losses(state,g,None,arm,epoch)
        expected_total=sum((position_weight(epoch) if k=='position' else WEIGHTS[k])*v[k].mean() for k in WEIGHTS)
        torch.testing.assert_close(scheduled['total'],expected_total)
    assert position_weight(1)==.375 and position_weight(15)==position_weight(200)==.25
    clean={k:g[k] for k in PUBLIC_TASK_KEYS};tasks=public_components(state,clean)
    no_ref={k:value for k,value in g.items() if k!='reference'}
    changed={**no_ref,'target':g['target']+1};altered=components(state,changed)
    for k in TASK_KEYS:assert torch.equal(v[k],altered[k]) and torch.equal(v[k],tasks[k]),k
    assert not torch.equal(v['total'],altered['total'])
    try:public_components(state,{**clean,'target':g['target']})
    except ValueError:pass
    else:raise AssertionError('Public task API must reject reference labels')
    cropped={k:(value[:,:113,:113] if k in ('adjacency','mib','group','wire_b2b') else
                value[:,:113] if value.ndim>=2 and value.shape[1]==120 else value) for k,value in g.items()}
    cropped_values=components(state[:,:113],cropped)
    for k in WEIGHTS:torch.testing.assert_close(v[k],cropped_values[k],rtol=2e-5,atol=2e-5)
    # Same physical weighted Manhattan costs as official, only normalized publicly.
    from .model import decode
    decoded=decode(state,g).detach();official=official_evaluator()
    wire_error=0.
    for i,row in enumerate(batch[:5]):
        n=row['graph']['n'];scale=float(g['scale'][i]);positions=(decoded[i,:n]*scale).cpu().tolist()
        inp=row['case'].solver_inputs
        wl=official.calculate_hpwl_b2b(positions,inp[2])+official.calculate_hpwl_p2b(positions,inp[3],inp[4])
        weights=sum(float(e[2]) for edges in (inp[2],inp[3]) for e in edges if float(e[2])>0)
        expected_wire=wl/(scale*weights) if weights else 0.
        wire_error=max(wire_error,abs(expected_wire-float(v['actual_wirelength'][i])))
        single={k:value.cuda() for k,value in inference_graph(inp).items()}
        with torch.no_grad():p=model(single)
        torch.testing.assert_close(p[0],prediction[i,:n],rtol=5e-5,atol=1e-5)
    assert wire_error<2e-5,wire_error
    state_norms={k:float(torch.autograd.grad((WEIGHTS[k]*v[k]).mean(),state,retain_graph=True)[0].norm()) for k in WEIGHTS}
    assert all(torch.isfinite(x).all() for x in v.values())
    # Same five train members only: verify internal gradient coverage at a
    # trained layout as well as at random initialization. No parameter update.
    historical=ROOT/'experiments/E70_reference_task_loss/runs/20261028_full_ports_taskloss_v1_seed0/checkpoints/best.pt'
    frozen=GeometryModel(**model_config(arm)).cuda().eval()
    frozen.load_state_dict(torch.load(historical,map_location='cpu',weights_only=False)['model'])
    with torch.no_grad():trained=frozen(public)
    trained=trained.detach().requires_grad_();tv=components(trained,g)
    dg=torch.autograd.grad((WEIGHTS['density']*tv['density']).sum(),trained,retain_graph=True)[0]
    bg=torch.autograd.grad((WEIGHTS['compactness']*tv['compactness']).sum(),trained)[0]
    coverage=[]
    for i in range(5):
        mutable=g['valid'][i] & (g['mask'][i,:,:2].sum(-1)>0)
        coverage.append({'n':int(g['valid'][i].sum()),'movable':int(mutable.sum()),
                         'density_nonzero_center_gradients':int(((dg[i,:,:2].norm(dim=-1)>1e-9)&mutable).sum()),
                         'bbox_nonzero_center_gradients':int(((bg[i,:,:2].norm(dim=-1)>1e-9)&mutable).sum()),
                         'density_value':float(tv['density'][i])})
    assert all(r['density_nonzero_center_gradients']>r['bbox_nonzero_center_gradients'] for r in coverage)
    del frozen,trained,tv,dg,bg
    initial={k:x.clone() for k,x in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-4)
    capture_start=time.perf_counter();captured=CapturedStep(model,optimizer,g,arm);capture_seconds=time.perf_counter()-capture_start
    assert not optimizer.state and all(torch.equal(x,initial[k]) for k,x in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay()
    except RuntimeError as e:assert 'Nonfinite' in str(e)
    else:raise AssertionError('Nonfinite objective must prevent Adam')
    assert not optimizer.state;captured.inputs['target'].copy_(g['target']);captured.replay()
    # Replay the same captured graph at three curriculum phases and compare
    # real Adam updates to an independent eager model, without replacing the
    # captured parameter gradient buffers.
    deltas={}
    for epoch in (1,10,16):
        model.load_state_dict(initial);optimizer.state.clear()
        captured.replay(epoch=epoch)
        eager_model=GeometryModel(**model_config(arm)).cuda().eval();eager_model.load_state_dict(initial)
        eager=torch.optim.AdamW(eager_model.parameters(),lr=1e-3,weight_decay=3e-4)
        eager.zero_grad(set_to_none=True);losses(eager_model(public),g,None,arm,epoch)['total'].backward()
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(),1.,error_if_nonfinite=True,foreach=True);eager.step()
        delta=max(float((x-eager_model.state_dict()[k]).abs().max()) for k,x in model.state_dict().items())
        assert delta<1e-6,(epoch,delta)
        deltas[str(epoch)]=delta
        del eager_model,eager
    torch.cuda.synchronize();tic=time.perf_counter()
    for _ in range(3):captured.replay(epoch=16)
    torch.cuda.synchronize();step_seconds=(time.perf_counter()-tic)/3
    return {'parameters':sum(p.numel() for p in model.parameters()),'input_channels':14,
            'label_independence_bit_exact':True,'fixed_total_and_padding':True,'public_inference_matches_batch':True,
            'trained_E70_same_five_gradient_coverage':coverage,
            'wire_max_abs_difference_to_official_normalized':wire_error,'component_initial_means':{k:float(v[k].mean()) for k in WEIGHTS},
            'weighted_output_gradient_norms':state_norms,'capture_eager_Adam_max_abs_delta_by_epoch':deltas,
            'CUDA_capture_seconds':capture_seconds,'captured_step_seconds_5case_batch32':step_seconds,
            'nonfinite_prevents_Adam':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}


def main():
    final=HERE/'runs/preflight/results.json'
    if final.exists():
        prior=json.loads(final.read_text())
        if prior['status']=='passed' and prior['sources']==sources():print('Current E71 preflight already passed.');return
    started=time.perf_counter();seed_all(0);protected=verify_protected();metadata()
    assert json.loads((HERE/'proposal.json').read_text())['weights']==WEIGHTS
    assert sha(HERE/'loss_reference.py')==sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    indices=[1683,10027,5474,9793,1814];rows=load_rows('train',indices,use_cache=False);batch=[rows[i%5] for i in range(32)]
    suffix=str(time.time_ns());checks={};fixtures=geometry_fixtures();density_checks=density_fixtures()
    for arm in ARMS:
        checks[arm]=check_arm(arm,batch);clear()
        args=parser().parse_args(['--arm',arm,'--output',str(HERE/f'runs/preflight_{suffix}_{arm}_resumed'),
                                 '--max-hours','.1','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        first=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        resumed=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
        args.resume=False;args.output=HERE/f'runs/preflight_{suffix}_{arm}_uninterrupted'
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        uninterrupted=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert first['progress']['step']==1 and resumed['progress']['step']==uninterrupted['progress']['step']==2
        for k in ('model','optimizer','scheduler'):same(resumed[k],uninterrupted[k],k)
        for k in ('torch','cuda'):same(resumed['rng'][k],uninterrupted['rng'][k],k)
        assert resumed['elapsed_seconds']>first['elapsed_seconds']
        assert first['identity']==resumed['identity']==uninterrupted['identity']
        checks[arm]['Dropout_Adam_resume_bit_exact']=True
        del first,resumed,uninterrupted;clear()
    assert verify_protected()==protected
    save(final,{'status':'passed','unique_train_cases':5,'indices':indices,'arms':checks,'sources':sources(),
                'formal_training_started':False,'protected_checkpoints_unchanged':protected,'test_read':False,
                'geometry_fixtures':fixtures,'density_and_schedule_checks':density_checks,'seconds':time.perf_counter()-started})
    print('E71 five-case public-task Loss/Adam/resume preflight passed; no formal training started.',flush=True)


if __name__=='__main__':main()
