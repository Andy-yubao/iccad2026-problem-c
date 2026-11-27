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
from .train import seed_all,parser,run,selection


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
    assert torch.equal(values['total'],losses(state,g,None,arm,200)['total'])
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
    initial={k:x.clone() for k,x in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-4)
    capture_start=time.perf_counter();captured=CapturedStep(model,optimizer,g,arm);capture_seconds=time.perf_counter()-capture_start
    assert not optimizer.state and all(torch.equal(x,initial[k]) for k,x in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay()
    except RuntimeError as e:assert 'Nonfinite' in str(e)
    else:raise AssertionError('Nonfinite objective must prevent Adam')
    assert not optimizer.state;captured.inputs['target'].copy_(g['target']);captured.replay()
    reference={k:x.clone() for k,x in model.state_dict().items()}
    # Small actual-update timing on the same32 rows, no new data or formal run.
    torch.cuda.synchronize();tic=time.perf_counter()
    for _ in range(3):captured.replay()
    torch.cuda.synchronize();step_seconds=(time.perf_counter()-tic)/3
    model.load_state_dict(initial);eager=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=3e-4)
    eager.zero_grad(set_to_none=True);losses(model(public),g,None,arm)['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True);eager.step()
    delta=max(float((x-reference[k]).abs().max()) for k,x in model.state_dict().items());assert delta<1e-6,delta
    return {'parameters':sum(p.numel() for p in model.parameters()),'input_channels':14,
            'label_independence_bit_exact':True,'fixed_total_and_padding':True,'public_inference_matches_batch':True,
            'wire_max_abs_difference_to_official_normalized':wire_error,'component_initial_means':{k:float(v[k].mean()) for k in WEIGHTS},
            'weighted_output_gradient_norms':state_norms,'capture_eager_Adam_max_abs_delta':delta,
            'CUDA_capture_seconds':capture_seconds,'captured_step_seconds_5case_batch32':step_seconds,
            'nonfinite_prevents_Adam':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}


def main():
    final=HERE/'runs/preflight/results.json'
    if final.exists():
        prior=json.loads(final.read_text())
        if prior['status']=='passed' and prior['sources']==sources():print('Current E70 preflight already passed.');return
    started=time.perf_counter();seed_all(0);protected=verify_protected();metadata()
    assert json.loads((HERE/'proposal.json').read_text())['weights']==WEIGHTS
    assert sha(HERE/'loss_reference.py')==sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    indices=[1683,10027,5474,9793,1814];rows=load_rows('train',indices,use_cache=False);batch=[rows[i%5] for i in range(32)]
    suffix=str(time.time_ns());checks={};fixtures=geometry_fixtures()
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
                'geometry_fixtures':fixtures,'seconds':time.perf_counter()-started})
    print('E70 five-case public-task Loss/Adam/resume preflight passed; no formal training started.',flush=True)


if __name__=='__main__':main()
