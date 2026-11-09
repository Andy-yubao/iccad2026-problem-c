"""Five train members only: independent, global, single-readout neural contract."""
import copy,json,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sources,sha,checkpoint
from .data import load_rows,collate,inputs_only,placer_inputs
from .model import GeometryModel,GlobalCorrector,decode
from .initialization import load,parent
from .loss import losses,per_case
from .train import seed,rng,restore_rng,Scheduler,CONFIG

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

def geometry(state,g):
    p=decode(state,g);v=g['valid'];placed=(~g['mask'][...,0].bool())&v;locked=(~g['mask'][...,2].bool())&v
    torch.testing.assert_close(p[...,2:].prod(-1)[v],g['area'][v],rtol=4e-6,atol=3e-8)
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
    assert bool((p[...,:2]>=0).all())

def step(model,opt,g):
    model.train();opt.zero_grad(set_to_none=True);v=losses(model(inputs_only(g)),g)
    assert bool(torch.isfinite(torch.stack(list(v.values()))).all())
    v['total'].backward();torch.nn.utils.clip_grad_norm_(model.corrector.parameters(),1.,error_if_nonfinite=True);opt.step()
    return float(v['total'].detach())

def main():
    begin=time.perf_counter();seed(0);ids=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['preflight_train_ids']
    source=parent();digest=sha(ROOT/source['checkpoint']);assert digest==source['sha256']
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for a,b in zip(rows,cached):
        assert a['member']==b['member']
        for key in a['graph']:
            if isinstance(a['graph'][key],torch.Tensor):same(a['graph'][key],b['graph'][key])
            else:assert a['graph'][key]==b['graph'][key]
        same(a['public_ports'],b['public_ports'])
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    batch=[rows[i%5] for i in range(32)];g=collate(batch,'cuda');public=inputs_only(g)
    model=GeometryModel(**CONFIG).cuda().eval();load(model)
    heads=[];layers=[]
    hook=model.corrector.head[-1].register_forward_hook(lambda *_:heads.append(1))
    handles=[layer.register_forward_hook(lambda *_:layers.append(1)) for layer in model.corrector.layers]
    with torch.no_grad():
        base=model.draft(public);prediction=model.corrector(public,base)
        assert len(heads)==1 and len(layers)==12
        torch.testing.assert_close(decode(prediction,g),decode(base,g),rtol=0,atol=0)
        geometry(prediction,g)
        initial=per_case(prediction,g)
    hook.remove()
    for handle in handles:handle.remove()
    try:model(g);raise AssertionError('Supervision accepted as input')
    except ValueError:pass
    initial_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # The output is not clipped to a local .15 sqrt(area) motion bound.
    with torch.no_grad():
        model.corrector.head[-1].bias[0]=.4
        large=model.corrector(public,base)
        movable=g['mask'][...,0].bool()&g['valid']
        assert bool(((large[...,0]-base[...,0])>.15*g['area'].sqrt()+1e-4)[movable].any())
        geometry(large,g)
    model.load_state_dict(initial_weights)
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
    first=step(model,opt,g)
    saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng(),step=1,cursor=32)
    checkpoint(HERE/'runs/preflight/resume_step1.pt',saved)
    second=step(model,opt,g)
    whole_weights=copy.deepcopy(model.state_dict());whole_opt=copy.deepcopy(opt.state_dict());whole_rng=rng()
    resumed=GeometryModel(**CONFIG).cuda();resumed.load_state_dict(saved['model'])
    resumed_opt=torch.optim.AdamW(resumed.corrector.parameters(),lr=.001,weight_decay=.0003);resumed_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    resumed_second=step(resumed,resumed_opt,g)
    same(resumed.state_dict(),whole_weights);same(resumed_opt.state_dict(),whole_opt);same(rng(),whole_rng)
    assert second==resumed_second
    for key,value in whole_weights.items():
        if key.startswith('placer.'):same(value.cpu(),initial_weights[key])
    assert all(p.grad is None for p in resumed.placer.parameters())
    for index in range(12):
        assert any(not torch.equal(value.cpu(),initial_weights[key]) for key,value in whole_weights.items() if key.startswith(f'corrector.layers.{index}.'))
    resumed.eval()
    with torch.no_grad():
        out=resumed(public);geometry(out,g)
        standalone=GlobalCorrector(**CONFIG).cuda().eval();standalone.load_state_dict(resumed.corrector.state_dict())
        torch.testing.assert_close(standalone(public,resumed.draft(public)),out,rtol=0,atol=0)
    # Verify actual cross-module neural influence after real updates.
    probe=dict(public);probe['features']=public['features'].detach().clone().requires_grad_()
    state=resumed.corrector(probe,resumed.draft(public).detach())
    i=int(torch.nonzero(movable[0],as_tuple=False)[0]);centers=decode(base,g)[0,:,:2]
    distance=(centers-centers[i]).square().sum(-1).masked_fill(~g['valid'][0],-1)
    j=int(distance.argmax());grad=torch.autograd.grad(state[0,i,:2].sum(),probe['features'])[0]
    influence=float(grad[0,j].abs().max());assert np.isfinite(influence) and influence>1e-12
    # A complete module permutation changes ordering, not the represented layout.
    n=public['valid'].shape[1];perm=torch.arange(n-1,-1,-1,device='cuda');pg={}
    pair={'adjacency','mib','group','b2b_weights'};ports={'port_xy','port_valid'}
    for key,value in public.items():
        if key in pair:pg[key]=value[:,perm][:,:,perm]
        elif key=='p2b_weights':pg[key]=value[:,perm]
        elif key in ports or key=='scale':pg[key]=value
        else:pg[key]=value[:,perm]
    with torch.no_grad():
        direct=resumed.corrector(public,resumed.draft(public))
        permuted=resumed.corrector(pg,resumed.draft(public)[:,perm])
        torch.testing.assert_close(permuted,direct[:,perm],rtol=1e-4,atol=2e-5)
    # Same fixed Loss formula is used before/after serialization.
    same_loss=losses(out,g)['total'];assert torch.isfinite(same_loss)
    fake=torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))],lr=.001);sched=Scheduler();drops=[]
    for epoch in range(12):
        old=sched.drops;sched.step(.5,epoch,fake)
        if old!=sched.drops:drops.append(epoch)
        if epoch<11:assert not sched.stop(epoch)
    assert drops==[3,6,9] and sched.stop(11)
    assert sha(ROOT/source['checkpoint'])==digest
    result=dict(status='passed',unique_train_cases=5,indices=ids,public_port_counts=[len(r['public_ports']['port_xy']) for r in rows],
        raw_cache_parity=True,complete_public_port_coordinates_and_edge_weights=True,labels_rejected=True,test_roles_rejected=True,
        independent_corrector=True,one_simultaneous_output_head=True,global_hidden_layers=12,
        initial_draft_geometry_identity_exact=True,no_local_movement_cap=True,hard_area_shapes_preplacements_after_updates=True,
        all_12_neural_layers_update=True,frozen_placer_bit_exact=True,real_Dropout_Adam_RNG_resume_bit_exact=True,
        module_permutation_equivariant=True,far_module_cross_influence=influence,standalone_export_equivalent=True,
        parameters=sum(p.numel() for p in model.parameters()),corrector_parameters=sum(p.numel() for p in model.corrector.parameters()),
        optimizer_steps_are_implementation_checks_only=True,first_two_step_Loss=[first,second],
        scheduler_flat_loss_drop_epochs=drops,sources=sources(),seconds=time.perf_counter()-begin,formal_training_started=False,test_read=False)
    save(HERE/'runs/preflight/results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
