"""Only the original five train cases; bounded implementation updates, not training."""
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from .io import HERE,save,checkpoint,sources
from .runtime import ARMS,model_config,seed,rng,restore_rng,charge,preflight_dir
from .data import load_rows,collate,inputs_only
from .model import GeometryModel,decode
from .initialization import load
from .compression import network_view
from .loss import per_case,configuration
from .step_engine import PackedStep
from .graph_step import CapturedStep
from .packed import PublicPacker

IDS=[1683,10027,5474,10140,15001]

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def geometry(state,g):
    p=decode(state,g);valid=g['valid'];placed=(~g['mask'][...,0].bool())&valid;locked=(~g['mask'][...,2].bool())&valid
    torch.testing.assert_close(p[...,2:].prod(-1)[valid],g['area'][valid],rtol=4e-6,atol=3e-8)
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
    assert bool((p[...,:2]>=0).all())

def objective_probes(g,draft):
    """Read-only geometric probes of the same five cases, never network updates."""
    g={k:v[:5] for k,v in g.items()};draft=draft[:5].detach()
    weights=configuration()['weights']
    assert weights==dict(overlap=1.,bbox=.5,wirelength=.05,soft=.2,intent=.1)
    state=draft.clone().requires_grad_();v=per_case(state,g,draft)
    expected=v['overlap_total']+.5*v['bbox_over_A']+.05*v['wirelength']+.2*v['soft']+.1*v['intent']
    torch.testing.assert_close(v['total'],expected,rtol=0,atol=0)
    grad=torch.autograd.grad(v['overlap_total'].sum(),state)[0]*g['mask']
    local=g['area'].sqrt();unit=grad/grad.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    origin=(draft[...,:2]*g['valid'][...,None]).sum(1)/g['valid'].sum(1).clamp_min(1)[:,None]
    probes={}
    for name,factor in (('contract_5pct',.95),('expand_5pct',1.05)):
        delta=torch.zeros_like(draft);delta[...,:2]=(draft[...,:2]-origin[:,None])*(factor-1)
        probes[name]=draft+delta*g['mask']
    probes['local_overlap_gradient_probe']=draft-.02*local[...,None]*unit*g['mask']
    keys=('overlap_total','bbox_over_A','wirelength','soft','intent','intent_neighbour','total')
    with torch.no_grad():
        baseline=per_case(draft,g,draft);records=[]
        for name,probe in probes.items():
            geometry(probe,g);values=per_case(probe,g,draft)
            assert all(bool(torch.isfinite(x).all()) for x in values.values())
            records.append(dict(probe=name,per_case=[dict(train_id=index,
                delta={k:float((values[k]-baseline[k])[i].cpu()) for k in keys}) for i,index in enumerate(IDS)]))
    return dict(weights=weights,probes=records,optimizer_updates=0,
        note='five-train geometry diagnostics only; gradient probe is not part of F or training, no coefficient selection')

def contract(arm,device):
    seed();rows=load_rows('train',IDS);selected=[rows[i%5] for i in range(32)]
    g=inputs_only(collate(selected,device));cfg=model_config(arm,device)
    model=GeometryModel(**cfg).to(device).eval();load(model);initial_digest=digest(model)
    with torch.no_grad():draft=model.draft(g)
    for i,row in enumerate(rows):row['draft']=draft[i,:row['graph']['n']].detach().cpu().clone()
    view=network_view(g,arm);assert torch.equal(view['features'],g['features'])
    assert view['port_xy'].shape[1]==(0 if arm=='centroid' else 32)
    total=g['p2b_weights'].sum(-1)
    center=(g['p2b_weights'][...,None]*g['port_xy'][:,None]).sum(2)/total.clamp_min(1e-12)[...,None]
    torch.testing.assert_close(g['features'][...,-3],total.log1p(),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(g['features'][...,-2:],center,rtol=1e-5,atol=1e-6)
    if arm=='regional32':
        torch.testing.assert_close(view['p2b_weights'].sum(-1),total,rtol=1e-5,atol=1e-5)
        original=(g['port_xy']*g['p2b_weights'].sum(1)[...,None]).sum(1)
        compressed=(view['port_xy']*view['p2b_weights'].sum(1)[...,None]).sum(1)
        torch.testing.assert_close(original,compressed,rtol=1e-5,atol=1e-5)
    # Port identity/order is irrelevant; empty public connectivity must stay finite.
    reverse=dict(g);reverse.update(port_xy=g['port_xy'].flip(1),port_valid=g['port_valid'].flip(1),p2b_weights=g['p2b_weights'].flip(2))
    reversed_view=network_view(reverse,arm)
    for k in ('port_xy','p2b_weights'):torch.testing.assert_close(reversed_view[k],view[k],rtol=1e-5,atol=1e-6)
    empty=dict(g);empty['p2b_weights']=torch.zeros_like(g['p2b_weights']);ev=network_view(empty,arm)
    assert bool(torch.isfinite(ev['port_xy']).all()) and not bool(ev['port_valid'].any())
    with torch.no_grad():
        calls=[];hook=model.corrector.head[-1].register_forward_hook(lambda *_:calls.append(1))
        out=model.corrector(g,draft);hook.remove();assert len(calls)==1
        torch.testing.assert_close(decode(out,g),decode(draft,g),rtol=0,atol=0);geometry(out,g)
        baseline=per_case(draft,g,draft);values=per_case(out,g,draft)
        torch.testing.assert_close(values['total'],baseline['total'],rtol=0,atol=0)
    try:model.corrector(dict(g,reference=torch.zeros(1,device=device)),draft);raise AssertionError('Labels accepted')
    except ValueError:pass
    from .dataset_view import ReferenceLayouts
    try:ReferenceLayouts('test');raise AssertionError('Test role opened')
    except PermissionError:pass
    probes=objective_probes(g,draft)
    # Real updates with actual Dropout and Adam, then restore the next update.
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003);engine=PackedStep(model,opt)
    initial_model=cpu(model.state_dict());initial_rng=rng()
    first_values,_,_=engine.replay(g,draft)
    first_values={k:float(v.detach().cpu()) for k,v in first_values.items()}
    progress=dict(cursor=32,step=1,order=np.random.default_rng(0).permutation(16000))
    folder=preflight_dir(arm,device)
    checkpoint(folder/'step1.pt',dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),rng=rng(),progress=progress,loss=first_values,
        initial_model=initial_model,initial_rng=initial_rng))
    engine.replay(g,draft);whole=cpu(model.state_dict());whole_opt=cpu(opt.state_dict());whole_rng=rng()
    first=torch.load(folder/'step1.pt',map_location='cpu',weights_only=False)
    restored=GeometryModel(**cfg).to(device);restored.load_state_dict(first['model'])
    restored_opt=torch.optim.AdamW(restored.corrector.parameters(),lr=.001,weight_decay=.0003);restored_opt.load_state_dict(first['optimizer']);restore_rng(first['rng'])
    PackedStep(restored,restored_opt).replay(g,draft)
    exact(whole,restored.state_dict());exact(whole_opt,restored_opt.state_dict());exact(whole_rng,rng());exact(progress,first['progress'])
    with torch.no_grad():geometry(restored.corrector(g,draft),g)
    assert all(p.grad is None for p in restored.placer.parameters())
    result=dict(status='passed',arm=arm,device=device,unique_train_cases=5,indices=IDS,initial_model_sha256=initial_digest,
        token_slots=view['port_xy'].shape[1],natural_module_slots=g['valid'].shape[1],legacy_module_summary_exact=True,port_order_invariant=True,
        empty_connectivity_finite=True,labels_and_test_rejected=True,zero_head_identity_exact=True,
        original_full_port_Loss_preserved=True,one_simultaneous_output_head=True,public_hard_geometry_preserved=True,
        real_Adam_Dropout_next_update_restore_bit_exact=True,serialized_cursor_order_preserved=True,
        frozen_L_has_no_gradients=True,objective_probes=probes,implementation_updates_only=True,formal_training_started=False,test_read=False)
    return result,rows,first

def graph_check(arm,rows,eager_first):
    import gc
    seed();torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info()
    if free<2.5*2**30:raise RuntimeError('Need2.5GiB free for short GPU check; existing training is not stopped')
    cfg=model_config(arm);model=GeometryModel(**cfg).cuda().eval();model.load_state_dict(eager_first['initial_model'])
    packer=PublicPacker(rows[0],p=480);h,d,_=packer.pack([rows[i%5] for i in range(32)]);g={k:v.cuda() for k,v in h.items()};d=d.cuda()
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
    # Compare identical padding. Natural CPU/GPU batches can have fewer than120
    # module slots, so their Dropout masks must not be compared bit for bit.
    restore_rng(eager_first['initial_rng']);reference,_,_=PackedStep(model,opt).replay(g,d)
    reference_values={k:float(v.detach().cpu()) for k,v in reference.items()};reference_model=cpu(model.state_dict())
    del opt,model;gc.collect();torch.cuda.empty_cache()
    model=GeometryModel(**cfg).cuda();model.load_state_dict(eager_first['initial_model'])
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003)
    restore_rng(eager_first['initial_rng'])
    initial=cpu(model.state_dict());before_rng=rng();engine=CapturedStep(model,opt,g,d)
    exact(initial,model.state_dict());assert not opt.state;exact(before_rng,rng())
    engine.replay()
    parameter_error=max(float((v.detach().cpu()-reference_model[k]).abs().max()) for k,v in model.state_dict().items())
    loss_error=max(abs(float(v.detach().cpu())-reference_values[k]) for k,v in engine.values.items())
    print('Eager/graph parity',arm,'parameter',parameter_error,'Loss',loss_error,flush=True)
    assert parameter_error<3e-5 and loss_error<2e-5
    folder=preflight_dir(arm,'cuda')
    checkpoint(folder/'graph_step1.pt',dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),rng=rng()))
    engine.replay();whole=cpu(model.state_dict());moments=cpu(opt.state_dict());random=rng()
    peak=torch.cuda.max_memory_allocated()/2**20
    del engine,opt,model;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(folder/'graph_step1.pt',map_location='cpu',weights_only=False)
    model=GeometryModel(**cfg).cuda();model.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(model.corrector.parameters(),lr=.001,weight_decay=.0003);opt.load_state_dict(saved['optimizer'])
    engine=CapturedStep(model,opt,g,d);restore_rng(saved['rng']);engine.replay()
    exact(whole,model.state_dict());exact(moments,opt.state_dict());exact(random,rng())
    before=cpu(model.state_dict());before_opt=cpu(opt.state_dict());bad=d.clone();bad[0,0,0]=float('nan')
    try:engine.replay(draft=bad);raise AssertionError('Nonfinite accepted')
    except RuntimeError:pass
    exact(before,model.state_dict());exact(before_opt,opt.state_dict())
    return dict(graph_reconstruction_resume_model_Adam_RNG_bit_exact=True,warmup_no_Adam_updates=True,
        same_shape_eager_graph_update_max_abs=parameter_error,same_shape_eager_graph_loss_max_abs=loss_error,
        nonfinite_rejected_before_Adam=True,peak_allocated_MiB=peak,raw_public_shape=[32,120,480],network_tokens=120+(0 if arm=='centroid' else 32))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=(*ARMS,'both'),default='both');p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    results=[]
    for arm in ARMS if a.arm=='both' else (a.arm,):
        begin=time.perf_counter()
        try:
            result,rows,eager_first=contract(arm,a.device)
            if a.device=='cuda':
                import gc
                gc.collect();torch.cuda.empty_cache();result['graph']=graph_check(arm,rows,eager_first)
            result.update(seconds=time.perf_counter()-begin,sources=sources())
            save(preflight_dir(arm,a.device)/'results.json',result);results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2),flush=True)
        finally:charge(time.perf_counter()-begin)
    if len(results)==2:assert results[0]['initial_model_sha256']==results[1]['initial_model_sha256']

if __name__=='__main__':main()
