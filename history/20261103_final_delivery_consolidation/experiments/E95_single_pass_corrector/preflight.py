"""Five train members only: independent, global, single-readout neural contract."""
import copy,json,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sources,sha,checkpoint
from .data import load_rows,collate,inputs_only,placer_inputs
from .model import GeometryModel,GlobalCorrector,decode
from .initialization import load,parent
from .loss import losses,per_case,intent_components,BOX_WEIGHT
from .runtime import seed,rng,restore_rng,CONFIG,checked_update,charge
from .schedule import Scheduler

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
    public=inputs_only(g)
    values,_=checked_update(model,opt,public,model.draft(public))
    return values['total']

def bbox_check(base,g):
    state=base.detach().clone().requires_grad_()
    values=per_case(state,g,base)
    restored=values['total']-BOX_WEIGHT*values['bbox_over_A']+.05*values['compactness']
    torch.testing.assert_close(restored,values['v2_total_diagnostic'],rtol=2e-5,atol=2e-6)
    p=decode(state,g);lo=p[...,:2].masked_fill(~g['valid'][...,None],float('inf')).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~g['valid'][...,None],-float('inf')).amax(1)
    size=hi-lo;independent=size[:,0]*size[:,1]/(g['area']*g['valid']).sum(1)
    torch.testing.assert_close(values['bbox_over_A'],independent,rtol=0,atol=0)
    gb=torch.autograd.grad(values['bbox_over_A'].sum(),state,retain_graph=True)[0]*g['mask']
    # Fixed coefficient scales the true area derivative, with no tolerance band.
    weighted=torch.autograd.grad(values['bbox_cost'].sum(),state)[0]*g['mask']
    torch.testing.assert_close(weighted,BOX_WEIGHT*gb,rtol=1e-5,atol=2e-6)
    i,j,k=(int(v) for v in torch.nonzero(gb.abs()==gb.abs().max(),as_tuple=False)[0])
    epsilon=1e-4;plus=base.clone();minus=base.clone();plus[i,j,k]+=epsilon;minus[i,j,k]-=epsilon
    with torch.no_grad():
        finite_difference=float((per_case(plus,g,base)['bbox_over_A'][i]-per_case(minus,g,base)['bbox_over_A'][i])/(2*epsilon))
    analytic=float(gb[i,j,k]);assert abs(finite_difference-analytic)<=.01*max(abs(analytic),1.)
    return dict(other_loss_reconstruction_matches=True,decoded_bbox_matches=True,
        exact_coefficient_gradient=True,finite_difference=finite_difference,analytic=analytic,
        box_weight=BOX_WEIGHT,unique_cases=5)


def contract():
    begin=time.perf_counter();seed(0);ids=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['preflight_train_ids']
    source=parent();digest=sha(ROOT/source['checkpoint']);assert digest==source['sha256']
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for a,b in zip(rows,cached):
        assert a['member']==b['member']
        for key in a['graph']:
            if isinstance(a['graph'][key],torch.Tensor):same(a['graph'][key],b['graph'][key])
            else:assert a['graph'][key]==b['graph'][key]
        same(a['public_ports'],b['public_ports'])
    print('Five-case public/raw/cache parity passed',flush=True)
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
        from .data import placer_inputs
        torch.testing.assert_close(decode(base,g),decode(model.placer(placer_inputs(public)),g),rtol=0,atol=0)
        geometry(prediction,g)
        initial=per_case(prediction,public,base)
    hook.remove()
    for handle in handles:handle.remove()
    bbox_result=bbox_check(base,public)
    print('Direct bbox objective, remaining Loss and finite-difference check passed',flush=True)
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
    checkpoint(HERE/'runs/preflight_engine/resume_step1.pt',saved)
    saved=torch.load(HERE/'runs/preflight_engine/resume_step1.pt',map_location='cpu',weights_only=False)
    # RNG byte tensors must be restored from CPU even when model tensors use CUDA.
    saved['rng']['torch']=saved['rng']['torch'].cpu()
    saved['rng']['cuda']=[x.cpu() for x in saved['rng']['cuda']]
    second=step(model,opt,g)
    whole_weights=copy.deepcopy(model.state_dict());whole_opt=copy.deepcopy(opt.state_dict());whole_rng=rng()
    resumed=GeometryModel(**CONFIG).cuda();resumed.load_state_dict(saved['model'])
    resumed_opt=torch.optim.AdamW(resumed.corrector.parameters(),lr=.001,weight_decay=.0003);resumed_opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    resumed_second=step(resumed,resumed_opt,g)
    same(resumed.state_dict(),whole_weights);same(resumed_opt.state_dict(),whole_opt);same(rng(),whole_rng)
    assert second==resumed_second
    print('Eager Dropout/Adam/RNG resume passed',flush=True)
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
    same_loss=losses(out,public,resumed.draft(public))['total'];assert torch.isfinite(same_loss)
    fake=torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))],lr=.001);sched=Scheduler();drops=[]
    for epoch in range(12):
        old=sched.drops;sched.step(.5,epoch,fake)
        if old!=sched.drops:drops.append(epoch)
        if epoch<11:assert not sched.stop(epoch)
    assert drops==[3,6,9] and sched.stop(11)
    assert sha(ROOT/source['checkpoint'])==digest
    try:per_case(out,g,base);raise AssertionError('Loss accepted reference fields')
    except ValueError:pass
    with torch.no_grad():
        zero=intent_components(base,base,public)
        assert float(zero['intent'].abs().max())<1e-8
        # Calling the zero-head F repeatedly must remain an exact identity.
        standalone.load_state_dict({k[len('corrector.'):]:v for k,v in initial_weights.items() if k.startswith('corrector.')})
        original=base.clone();identity_out=base
        for _ in range(8):identity_out=standalone(public,identity_out)
        torch.testing.assert_close(decode(identity_out,g),decode(original,g),rtol=0,atol=0)
        # Match actual public edge/port weighted wirelength, including duplicate
        # edges and complete individual ports rather than centroid surrogates.
        actual=per_case(base,public,base)['wirelength'].cpu().numpy()
        rect=decode(base,g).cpu().numpy()
        for i,row in enumerate(batch):
            n,areas,b2b,p2b,pins,_,_=row['case'].solver_inputs
            r=rect[i,:int(n)];c=r[:,:2]+r[:,2:]/2;total=0.;weight=0.
            for a,b,w in np.asarray(b2b):
                if a<0:continue
                total+=float(w)*np.abs(c[int(a)]-c[int(b)]).sum();weight+=float(w)
            for a,b,w in np.asarray(p2b):
                if a<0:continue
                total+=float(w)*np.abs(np.asarray(pins)[int(a),:2]/float(g['scale'][i])-c[int(b)]).sum();weight+=float(w)
            expected=total/(max(weight,1e-12)*float((g['area'][i]*g['valid'][i]).sum().sqrt()))
            assert abs(float(actual[i])-expected)<1e-5
    # A nonfinite draft must fail before Adam mutates weights or moments.
    before=copy.deepcopy(resumed.state_dict());before_opt=copy.deepcopy(resumed_opt.state_dict())
    invalid=resumed.draft(public).clone();invalid[0,0,0]=float('nan')
    try:checked_update(resumed,resumed_opt,public,invalid);raise AssertionError('Nonfinite update accepted')
    except RuntimeError:pass
    same(before,resumed.state_dict());same(before_opt,resumed_opt.state_dict())
    from .evaluate import STEADY
    from .evaluate import existing_source_terminal
    assert existing_source_terminal().exists()
    from .solver import optimize
    from .diagnostics import geometry_checks
    resumed.eval()
    with torch.no_grad():probe_rect=decode(resumed(public),g)[0,:rows[0]['graph']['n']].cpu().double().numpy()*float(g['scale'][0])
    final,stats=optimize(probe_rect,rows[0]['case'].solver_inputs,**STEADY)
    _,hard,_=geometry_checks(final,rows[0]['case'].solver_inputs)
    assert hard['is_feasible']
    print('Neural contract, finite rejection and one-case terminal passed; checking packed engine',flush=True)
    result=dict(status='passed',unique_train_cases=5,indices=ids,public_port_counts=[len(r['public_ports']['port_xy']) for r in rows],
        raw_cache_parity=True,complete_public_port_coordinates_and_edge_weights=True,labels_rejected=True,test_roles_rejected=True,
        independent_corrector=True,one_simultaneous_output_head=True,global_hidden_layers=12,
        initial_draft_geometry_identity_exact=True,source_decode_identity_exact=True,no_local_movement_cap=True,hard_area_shapes_preplacements_after_updates=True,
        all_12_neural_layers_update=True,frozen_placer_bit_exact=True,real_Dropout_Adam_RNG_resume_bit_exact=True,
        module_permutation_equivariant=True,far_module_cross_influence=influence,standalone_export_equivalent=True,
        parameters=sum(p.numel() for p in model.parameters()),corrector_parameters=sum(p.numel() for p in model.corrector.parameters()),
        optimizer_steps_are_implementation_checks_only=True,first_two_step_Loss=[first,second],bbox_objective=bbox_result,
        no_reference_corrector_objective=True,identity_intent_zero=True,complete_wirelength_matches_raw_edges=True,
        zero_head_eight_calls_no_numerical_drift=True,source_terminal_protocol_matches=True,
        nonfinite_update_rejected_before_Adam=True,one_train_case_frozen_flash_hard_legal=True,
        scheduler_flat_loss_drop_epochs=drops,sources=sources(),seconds=time.perf_counter()-begin,formal_training_started=False,test_read=False)
    return result

def main():
    import gc
    result=contract();gc.collect();torch.cuda.empty_cache()
    proof=HERE/'runs/20261101_engine_repair/check.json'
    result['engine']=json.loads(proof.read_text(encoding='utf-8'))
    if result['engine']['status']!='passed':raise RuntimeError('Stopped-checkpoint graph checks required')
    if result['engine']['old_checkpoint_sha256']!=sha(HERE/'runs/20261101_direct_bbox846_seed0/checkpoints/latest.pt'):
        raise RuntimeError('Stopped checkpoint changed after engine verification')
    result['engine_proof_sha256']=sha(proof)
    result['sources']=sources()
    save(HERE/'runs/preflight_engine/results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:charge(time.perf_counter()-started)

